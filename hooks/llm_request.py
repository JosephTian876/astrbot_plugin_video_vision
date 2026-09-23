"""``on_llm_request`` 钩子：把视频作为原生模态注入请求。

为什么是钩子而不是 provider 适配器
----------------------------------
AstrBot 主对话轮次的上下文由 **``ProviderRequest.assemble_context()``**
（``core/provider/entities.py``）组装，而 runner 调用 provider 时传的是
``contexts=`` 且 **``prompt=None``**（``tool_loop_agent_runner.py``）。
而 ``ProviderGoogleGenAI.text_chat`` 只在 ``prompt is not None`` 时才调用
provider 侧的 ``assemble_context``（``gemini_source.py``）。

因此覆写 provider 的 ``assemble_context`` **在主对话轮次上永远不会被调用** ——
只有图片描述、TTS 后处理等辅助调用会走到那里。

正确做法是在 ``on_llm_request`` 阶段把视频块追加进
``req.extra_user_content_parts``。该字段的内容会被
``ProviderRequest.assemble_context()`` 原样透传，最终在
``gemini_source._prepare_conversation`` 的 ``else`` 分支里交给
``process_audio_url`` 构造成 ``Part.from_bytes(mime_type="video/mp4")``。

三个必须遵守的约束（来自对陪伴插件的兼容性审计）
------------------------------------------------
1. **必须标记为非持久化**（``mark_as_temp()``）。否则多兆的 base64 会写进
   对话历史，并在之后**每一轮**被重发 —— token 与成本爆炸、数据库膨胀。
   ``message.py`` 的 ``dump_messages_with_checkpoints`` 会丢弃 ``_no_save`` 的 part。
2. **用 ``AudioURLPart`` 对象而不是裸 dict**。provider 侧的 ``assemble_context``
   对非 ``TextPart``/``ImageURLPart``/``AudioURLPart`` 的 part 会 **raise ValueError**；
   而 ``ProviderRequest`` 侧两者都接受。用对象两边都安全。
3. **不要缓存 ``req.extra_user_content_parts`` 的列表引用**。陪伴插件会在多个
   钩子里把它替换成新列表；每次读写都重新取一次属性。
"""

from __future__ import annotations

import asyncio
import os
from typing import Any

from astrbot.api import logger

from ..core.config import VideoConfig
from ..core.injector import prepare_video
from ..core.placeholder import iter_placeholders, parse_placeholder_text

try:
    from astrbot.core.agent.message import AudioURLPart
except Exception:  # pragma: no cover - 版本差异兜底
    AudioURLPart = None  # type: ignore[assignment]

#: 判定为「Gemini 系」的 provider 类型 —— 只有它们才会把 audio_url 块按 MIME
#: 构造成媒体 Part（``process_audio_url`` 是 MIME 无关的）。其他 provider 会把
#: 视频当音频处理，因此默认拦住。
GEMINI_PROVIDER_TYPES: frozenset[str] = frozenset(
    {
        "googlegenai_chat_completion",
        "gemini_video_chat_completion",  # 兼容早期设计/第三方实现
    }
)

#: 挂在 req 上的去重标记，避免同一请求重复注入（前缀与陪伴插件区隔）
_DONE_ATTR = "_video_vision_injected"
_warned_modality: set[str] = set()
_warned_gate: set[str] = set()


def _short_reason(reason: str) -> str:
    """把失败原因压缩成不含本机路径的短语，供注入给模型。

    注意匹配顺序：更具体的原因要排在更宽泛的之前。
    例如「无法读取时长，拒绝盲压缩（体积超闸门）」应归类为时长问题，
    而不是体积问题 —— 否则模型会收到误导性的说明。
    """
    low = reason.lower()
    if "ffmpeg" in low:
        return "ffmpeg is not available"
    if "不存在" in reason or "not exist" in low or "不是普通文件" in reason:
        return "the file could not be read"
    if "时长" in reason:
        return "its duration could not be read"
    if "压缩" in reason and "拒绝" not in reason:
        return "compression failed"
    if "拒绝" in reason or "闸门" in reason or "too large" in low:
        return "it exceeds the size limit"
    return "it could not be processed"


def _parts_of(req: Any) -> list[Any]:
    """每次重新读取，绝不缓存列表引用。"""
    parts = getattr(req, "extra_user_content_parts", None)
    return parts if isinstance(parts, list) else []


def _temp_video_part(data_uri: str) -> Any:
    """构造一个**不持久化**的视频内容块。

    借道 ``audio_url`` 是刻意为之：``process_audio_url`` 从 data URI 里取 MIME
    再交给 ``Part.from_bytes``，因此 ``data:video/mp4`` 会被构造成视频 Part。
    """
    if AudioURLPart is None:
        # 极端兜底：至少带上 _no_save，避免污染历史
        return {
            "type": "audio_url",
            "audio_url": {"url": data_uri},
            "_no_save": True,
        }
    part = AudioURLPart(audio_url=AudioURLPart.AudioURL(url=data_uri))
    return part.mark_as_temp()


def _temp_text_part(text: str) -> Any:
    try:
        from astrbot.core.agent.message import TextPart

        return TextPart(text=text).mark_as_temp()
    except Exception:  # pragma: no cover
        return {"type": "text", "text": text, "_no_save": True}


def _path_key(path: str) -> str:
    """把路径规范化为去重用的 key（大小写、相对/绝对、分隔符归一）。"""
    try:
        return os.path.normcase(os.path.abspath(path))
    except Exception:  # noqa: BLE001
        return path


async def build_video_blocks(
    parts: list[Any],
    *,
    config: VideoConfig,
    cache_root: str,
) -> tuple[list[Any], set[int], list[str]]:
    """扫描占位符并准备视频内容块。

    **每个视频的处理都放到线程池执行**。``prepare_video`` 是同步的，内部会跑
    ffmpeg 子进程并做文件 I/O；若直接在事件循环里调用，一次最长
    ``max_compress_seconds``（默认 45 秒）的压缩会把整个 AstrBot 卡住 ——
    期间所有会话的消息都不被处理。

    Returns:
        (待追加的内容块列表, 已成功注入的占位符下标, 失败原因短语列表)
    """
    found = iter_placeholders(parts)
    if not found:
        return [], []

    total_found = len(found)

    # ---- 闸门 0：File 类型附件按扩展名筛选 ----
    # 实测：QQ/NapCat 把用户发的视频当作 File 组件投递，核心于是产出
    # [File Attachment: ...] 而非 [Video Attachment: ...]。必须兼容，
    # 否则插件在真实 QQ 环境下永不触发。
    def _is_video_candidate(ph: Any) -> bool:
        if not ph.is_file_kind:
            return True  # Video 组件来的，无条件处理
        return bool(config.treat_file_as_video) and ph.looks_like_video()

    rejected = [(i, p) for i, p in found if not _is_video_candidate(p)]
    if rejected:
        logger.debug(
            "[VideoVision] 跳过 %d 个非视频附件: %s",
            len(rejected),
            [p.name for _, p in rejected][:5],
        )
    found = [(i, p) for i, p in found if _is_video_candidate(p)]
    if not found:
        return [], []

    # ---- 闸门 1：引用视频 ----
    # 先按策略过滤，再套用数量上限 —— 否则被跳过的引用视频会白占配额，
    # 可能把真正的视频挤掉。
    if not config.handle_quoted_video:
        found = [(i, p) for i, p in found if not p.quoted]
        if not found:
            logger.debug("[VideoVision] 仅有引用视频且配置为不处理，跳过")
            return [], []

    limit = config.max_video_count
    if limit <= 0:
        logger.debug("[VideoVision] max_video_count=0，跳过全部视频")
        return [], []

    selected = found[:limit]
    if len(found) > len(selected):
        logger.info(
            "[VideoVision] 本条消息含 %d 个待处理视频，按 max_video_count=%d 只处理前 %d 个",
            len(found),
            limit,
            len(selected),
        )
    elif total_found > len(found):
        logger.debug(
            "[VideoVision] 共 %d 个占位符，按策略过滤掉 %d 个引用视频",
            total_found,
            total_found - len(found),
        )

    blocks: list[Any] = []
    failures: list[str] = []
    consumed: set[int] = set()
    seen_paths: set[str] = set()

    for index, placeholder in selected:
        # 同一文件出现多次（例如既引用又重发）时只送一次，省带宽与 token
        key = _path_key(placeholder.path)
        if key in seen_paths:
            logger.info("[VideoVision] 跳过重复视频: %s", placeholder.name)
            continue
        seen_paths.add(key)

        outcome = await asyncio.to_thread(
            prepare_video,
            placeholder,
            config=config,
            cache_root=cache_root,
            logger=logger,
            debug=config.debug_log,
        )
        if outcome.ok and outcome.injection is not None:
            inj = outcome.injection
            blocks.append(_temp_video_part(inj.data_uri))
            consumed.add(index)
            logger.info(
                "[VideoVision] 视频已注入: %s | %s | %.2fMB -> %.2fMB | %.1fs | %s%s",
                placeholder.name,
                inj.action,
                _safe_size(inj.source_path),
                inj.size_mb,
                inj.duration_sec,
                inj.note or "正常",
                "（File 组件按视频处理）" if placeholder.is_file_kind else "",
            )
        else:
            failures.append(_short_reason(outcome.reason))
            logger.warning(
                "[VideoVision] 视频未注入: %s | %s", placeholder.name, outcome.reason
            )

    return blocks, consumed, failures


def _neutralize_placeholders(parts: list[Any], consumed: set[int]) -> int:
    """把已成功注入的占位文本改写成**不含本机路径**的形式。

    实测动机：核心产出的是 ``[File Attachment: name X, path C:\\...\\X.mp4]``，
    把本机绝对路径直接喂给了模型。模型是 agentic 的 —— 看到路径就会自己去跑
    ``astrbot_execute_python`` + ``cv2`` 抽帧、grep 知识库、web search，
    而**它本来已经拿到视频了**。实测日志：

        05:04:17  [VideoVision] 视频已注入 ... 6.16MB
        05:05:45  使用工具：astrbot_execute_python {'code': 'import cv2...'}
        05:05:46  Result: FPS: 30.0, Frame count: 87

    改写后模型仍知道「有个视频、叫什么名字」，但不再被路径诱导去自己动手。
    仅改写**已成功注入**的那些占位符；注入失败的原样保留（那时路径反而有用）。

    第二轮实测补充：只去掉路径还不够。模型**仍在按文件名搜整个磁盘**：

        target = "<视频文件名>"
        for root, dirs, files in os.walk(<用户主目录>): ...

    说明它把 ``[Video: X.mp4]`` 理解成「有个文件，我得去读它」，而没意识到
    **视频内容已经作为原生模态直接给它了**。因此改写文本必须同时说明这一点。
    """
    if not consumed:
        return 0
    changed = 0
    for index in sorted(consumed):
        if not (0 <= index < len(parts)):
            continue
        part = parts[index]
        text = getattr(part, "text", None)
        is_dict = isinstance(part, dict)
        if is_dict:
            text = part.get("text")
        if not isinstance(text, str):
            continue
        placeholder = parse_placeholder_text(text)
        if placeholder is None:
            continue
        name = placeholder.name or "attachment"
        replacement = (
            f"[视频内容已随本消息直接提供（文件名 {name}）。"
            "你已经在本次请求中看到了这段视频，无需再用任何工具去查找或读取该文件。]"
        )
        try:
            if is_dict:
                part["text"] = replacement
            else:
                part.text = replacement
            changed += 1
        except Exception:  # noqa: BLE001
            continue
    return changed


def _safe_size(path: str) -> float:
    try:
        return os.path.getsize(path) / 1024 / 1024
    except OSError:
        return 0.0


async def _resolve_provider(plugin: Any, event: Any) -> Any:
    try:
        return await plugin.context.get_using_provider_async(
            umo=getattr(event, "unified_msg_origin", None)
        )
    except Exception as exc:  # noqa: BLE001
        logger.debug("[VideoVision] 获取当前 provider 失败: %s", exc)
        return None


async def handle_llm_request(plugin: Any, event: Any, req: Any) -> None:
    """钩子入口。任何异常都在内部消化，绝不打断对话。"""
    try:
        await _handle(plugin, event, req)
    except Exception as exc:  # noqa: BLE001
        logger.error("[VideoVision] 注入视频时异常，已跳过: %s", exc, exc_info=True)


async def _handle(plugin: Any, event: Any, req: Any) -> None:
    config: VideoConfig = getattr(plugin, "config", None) or VideoConfig()
    if not config.enable:
        return
    if getattr(req, _DONE_ATTR, False):
        return

    # 便宜的预检：没有视频占位符就完全不做 await
    if not iter_placeholders(_parts_of(req)):
        return

    provider = await _resolve_provider(plugin, event)
    provider_config = getattr(provider, "provider_config", None) or {}
    provider_type = str(provider_config.get("type") or "")

    # ---- 闸门 1：必须是 Gemini 系 provider ----
    if config.require_gemini_provider and provider_type not in GEMINI_PROVIDER_TYPES:
        if provider_type not in _warned_gate:
            _warned_gate.add(provider_type)
            logger.warning(
                "[VideoVision] 当前 provider 类型 %r 不是 Gemini 系，已跳过视频注入。"
                "若该 provider 同样支持按 MIME 构造媒体 Part，"
                "可在插件配置里关闭 require_gemini_provider。",
                provider_type or "(未知)",
            )
        return

    # ---- 闸门 2：模型必须声明 audio 模态 ----
    # 视频块借道 audio_url，而 modality 清洗会把未声明的 audio 块替换成
    # "[Audio]" 文本，那样视频就白传了。
    modalities = provider_config.get("modalities")
    if isinstance(modalities, list) and modalities and "audio" not in modalities:
        key = f"{provider_type}:{provider_config.get('id') or ''}"
        if key not in _warned_modality:
            _warned_modality.add(key)
            logger.warning(
                "[VideoVision] 模型 %s 的 modalities 未包含 audio，视频块会被清洗成文本。"
                "请在 WebUI 给该模型勾选「音频」模态后重试。",
                provider_config.get("id") or provider_type,
            )
        return

    setattr(req, _DONE_ATTR, True)

    # ---- 准备（同步，无 await）----
    cache_root = str(getattr(plugin, "cache_root", "") or "")
    if config.debug_log:
        logger.info(
            "[VideoVision][debug] 钩子触发 | umo=%s provider=%s 候选占位符=%d",
            getattr(event, "unified_msg_origin", "?"),
            provider_type,
            len(iter_placeholders(_parts_of(req))),
        )
    blocks, consumed, failures = await build_video_blocks(
        _parts_of(req), config=config, cache_root=cache_root
    )

    # ---- 重新读取后再写入，避免期间列表被替换 ----
    target = _parts_of(req)
    if blocks:
        # 注入成功后抹掉占位文本里的本机路径，避免诱导模型自己去跑 Python/cv2
        if config.hide_local_path:
            changed = _neutralize_placeholders(target, consumed)
            if changed and config.debug_log:
                logger.info(
                    "[VideoVision][debug] 已抹除 %d 处占位文本中的本机路径", changed
                )
        target.extend(blocks)
    if failures and config.notify_on_failure:
        for reason in dict.fromkeys(failures):
            target.append(
                _temp_text_part(f"[Video attachment was skipped: {reason}]")
            )
    if config.debug_log:
        logger.info(
            "[VideoVision][debug] 注入完成 | 成功=%d 失败=%d 列表长度=%d",
            len(blocks),
            len(failures),
            len(target),
        )
