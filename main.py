"""「基米看得见视频」—— AstrBot 插件入口。

两种工作模式（由 ``provider_mode`` 配置切换）：

- ``gemini_native``：在 ``on_llm_request`` 阶段把视频转成内容块注入当前请求，
  让 Gemini 原生"看见"画面（要求主对话 provider 是 Gemini 系）。
- ``openai_compat``：插件直接把视频发给配置里的 OpenAI 兼容视觉服务做理解，
  再把理解结果以文字注入对话 —— 主对话模型可以是任意模型。

为什么不覆写 provider 的 ``assemble_context``：主对话轮次的上下文由
``ProviderRequest.assemble_context()`` 组装，runner 传 ``contexts=`` 且
``prompt=None``，而 provider 侧方法只在 ``prompt is not None`` 时才被调用。
详见 ``hooks/llm_request.py`` 的模块说明。
"""

from __future__ import annotations

import asyncio
import os
import time
from typing import Any

from astrbot.api import logger
from astrbot.api.event import AstrMessageEvent, filter
from astrbot.api.star import Context, Star, register

try:
    from astrbot.api.message_components import File, Video
except ImportError:  # pragma: no cover
    Video = None  # type: ignore[assignment]
    File = None  # type: ignore[assignment]

try:  # 单独 import：老版本缺 Reply 时不该把 Video/File 一起带下水
    from astrbot.api.message_components import Reply
except ImportError:  # pragma: no cover
    Reply = None  # type: ignore[assignment]

try:
    from astrbot.core.star.filter.command import GreedyStr
except ImportError:  # pragma: no cover
    GreedyStr = str  # type: ignore[assignment,misc]

from .core.config import VideoConfig
from .core.decide import (
    ACTION_LABELS,
    URI_LIMIT_MB,
    URI_LIMIT_RAW_MB,
    classify,
    describe,
)
from .core.injector import prepare_video
from .core.media import (
    OUTPUT_MIME,
    file_size_mb,
    guess_video_mime,
    probe_duration,
    resolve_ffmpeg_tools,
    suffix_of,
)
from .core.openai_vision import analyze_video, build_vision_prompt
from .core.placeholder import VIDEO_SUFFIXES, VideoPlaceholder
from .hooks.llm_request import (
    GEMINI_PROVIDER_TYPES,
    handle_llm_request,
    mark_degradation_warned,
    resolve_provider,
)

PLUGIN_NAME = "astrbot_plugin_video_vision"
PLUGIN_REPO = "https://github.com/JosephTian876/astrbot_plugin_video_vision"


def _read_plugin_version() -> str:
    """从 metadata.yaml 读取版本号，避免与发布版本脱节。"""
    try:
        from pathlib import Path

        text = (Path(__file__).parent / "metadata.yaml").read_text(encoding="utf-8")
        for line in text.splitlines():
            if line.startswith("version:"):
                return line.split(":", 1)[1].strip().strip("\"'")
    except Exception:  # noqa: BLE001 - 读不到就用占位值，不影响功能
        pass
    return "unknown"


PLUGIN_VERSION = _read_plugin_version()

#: 比其他插件都晚执行（数值越小越晚），确保陪伴插件改写完请求后我们再追加。
HOOK_PRIORITY = -300000

# ``_ACTION_LABELS`` 已搬进 ``core/decide.py``：动作词表是判决的一部分，
# ``/videotest`` 的判定行与「动作:」行都查同一张表（见 ``decide.describe``）。

#: ``/videotest`` 的用法。三条入口只有前两条在任何平台都成立，第三条看平台：
#:
#: * ① 显式路径 —— 永远是最后的手段；
#: * ② 引用（回复）含视频的消息 —— **唯一**让「视频」和「命令」落在同一条
#:   事件里的自然方式。被引用消息的消息段在 ``Reply.chain`` 里
#:   （``aiocqhttp_platform_adapter.py:328-337`` 收到 ``reply`` 段时先 ``get_msg``
#:   再把被引用消息整条重新解析成消息链），``Reply.message_str`` 只是文本；
#: * ③ 视频和命令写在同一条消息里（平台支持图文混排时）。
#:
#: 0.2.1 的旧文案写「或先发一个视频/视频文件，再发送 /videotest」，这句话是
#: **做不到的**：那是两条独立消息、两个独立事件，处理 ``/videotest`` 时
#: ``event.message_obj.message`` 里只有后一条消息，插件没有任何途径回头看上一条
#: 消息里的视频。真机证据（2026-10-26 11:34）：11:34:35 发视频、11:34:40 发
#: ``/videotest`` → 拿到的正是这段泛泛的用法提示；11:34:52 改成**引用**那条视频
#: 再发 ``/videotest`` → 命令命中并产出了诊断（引用链里的视频
#: ``media_video_20261026113452316_48f6.mp4``）。
_USAGE_TEXT = (
    "用法（任选一种）：\n"
    "  1) /videotest <视频文件绝对路径>\n"
    "  2) 引用（回复）那条视频消息，再发 /videotest\n"
    "  3) 视频和 /videotest 写在同一条消息里（平台支持时）\n"
    "注意：「先单独发一个视频、再单独发 /videotest」不行 —— 那是两条消息、"
    "两个事件，插件看不到上一条消息里的视频；请用 2) 的引用方式。\n"
    "路径含空格时请用引号包起来。"
)

#: 找不到视频时的原因码 → 一句具体的话。
#:
#: 0.2.1 只有一段泛泛的用法提示：不管是「没给参数」「引用的消息里没有视频」还是
#: 「平台根本没把被引用消息的内容交出来」，用户看到的字一模一样 —— 于是"命令不
#: 生效"和"你引用错了"在他的视角完全无法区分。
_VIDEO_REASON_TEXT = {
    "no-media": "本条消息和引用的消息里都没有视频",
    "quote-content-missing": (
        "引用了消息，但平台没有把被引用消息的内容交给插件"
        "（aiocqhttp 的 get_msg 失败时就是这样），看不到里面的视频"
    ),
    "quote-no-video": "引用的那条消息里没有视频",
    "video-unreadable": "找到了视频，但没能取到本地文件（链接可能已过期）",
}


def _videotest_usage(reason: str = "") -> str:
    """用法提示；``reason`` 非空时把「本次为什么没找到视频」一起说出来。"""
    note = _VIDEO_REASON_TEXT.get(reason, "")
    if not note:
        return _USAGE_TEXT
    head, _, tail = _USAGE_TEXT.partition("\n注意：")
    return f"{head}\n本次没找到视频：{note}\n注意：{tail}"


def _resolve_cache_root() -> str:
    """压缩缓存目录，放在 AstrBot 数据目录下的 plugin_data 里。"""
    try:
        from astrbot.core.utils.astrbot_path import get_astrbot_data_path

        base = get_astrbot_data_path()
    except Exception:  # pragma: no cover - 版本差异兜底
        base = os.path.join(os.path.expanduser("~"), ".astrbot", "data")
    return os.path.join(base, "plugin_data", PLUGIN_NAME, "cache")


def _missing_credentials(cfg: VideoConfig) -> list[str]:
    """视觉服务三要素里缺哪几项（凭据齐全性诊断用）。"""
    missing: list[str] = []
    if not cfg.openai_base_url:
        missing.append("openai_base_url")
    if not cfg.openai_api_key:
        missing.append("openai_api_key")
    if not cfg.openai_model:
        missing.append("openai_model")
    return missing


def _credentials_summary(cfg: VideoConfig) -> str:
    """一行"凭据是否齐全"，供 /videotest 使用。"""
    missing = _missing_credentials(cfg)
    if not missing:
        return " ✅ base_url / api_key / model 齐全"
    return f" ❌ 缺 {'、'.join(missing)}"


def _unknown_container_note(path: str) -> str:
    """显式路径 + 未知后缀时的警示句；后缀是已知视频容器时返回空串。

    ``/videotest`` 走的是**显式路径**，刻意跳过闸门 ⓪ 的「File 组件扩展名
    筛选」；而 ``guess_video_mime`` 对未知后缀回落到 ``video/mp4``。于是
    ``/videotest anim.gif`` 会显示「未超限，直接送 / ✅ 准备成功 /
    mime: video/mp4」—— 诊断说一切正常，真发出去却是假 MIME，provider 侧才
    会炸（真机证据见 LIVE-TEST §10.2③）。

    这里只把事实写进报告，**不改任何行为**：正常消息流仍由闸门 ⓪ 负责筛选。
    """
    suffix = suffix_of(path)
    if suffix in VIDEO_SUFFIXES:
        return ""
    return (
        f"⚠️ 扩展名 {suffix or '（无）'} 不在已知视频容器内：显式路径不校验扩展名，"
        f"将以 {OUTPUT_MIME} 发送，provider 可能拒绝"
    )


def _mime_line(path: str, action: str) -> str:
    """``MIME   :`` 行：报告实际声明给 provider 的 MIME。

    - 压缩/截断产物一定是 ffmpeg 产出的 ``.mp4``，MIME 是确定的；
    - 直接送时 MIME 由源文件后缀决定；后缀不在已知容器里就只能回落到
      ``video/mp4``，这一行必须标 ``(推断)``，不能和真实探测结果混为一谈。
    """
    if action != "as-is":
        return f"MIME   : {OUTPUT_MIME}（压缩产物，固定为 mp4）"
    if suffix_of(path) in VIDEO_SUFFIXES:
        return f"MIME   : {guess_video_mime(path)}"
    return f"MIME   : {OUTPUT_MIME} (推断)"


@register(
    PLUGIN_NAME,
    "JosephTian876",
    "让模型真正看见视频：Gemini 原生注入 / 任意 OpenAI 兼容视觉服务理解，双模式。",
    PLUGIN_VERSION,
    PLUGIN_REPO,
)
class VideoVisionPlugin(Star):
    """视频原生输入插件。"""

    def __init__(self, context: Context, config: dict | None = None):
        super().__init__(context)

        self.raw_config: dict = dict(config or {})
        # from_mapping_detailed 额外返回「未知 / 拼错的键」与「被上下界收敛的值」
        # —— 0.2.0 对这两类完全静默，用户只会奇怪"为什么配置没生效"。
        resolved, notices = VideoConfig.from_mapping_detailed(self.raw_config)
        resolved, warnings = resolved.normalized()
        self.config: VideoConfig = resolved
        #: 启动时记下的两类提示，``/videotest`` 会原样带出来 —— 用户看诊断时才
        #: 知道自己写的 ``compress_max_mb=15.0`` 被夹成了多少（LIVE-TEST §10.2①）。
        #: 两类分开存：``notices`` 是"你的输入被收敛/被忽略"（提示），
        #: ``warnings`` 是"normalized() 改了值"（修正）—— 混成一个前缀会把
        #: "忽略未知配置键"也说成"已修正"。
        self.config_notices: list[str] = list(notices)
        self.config_warnings: list[str] = list(warnings)
        for notice in notices:
            logger.warning("[VideoVision] 配置提示: %s", notice)
        for warning in warnings:
            logger.warning("[VideoVision] 配置已自动修正: %s", warning)

        self.cache_root: str = _resolve_cache_root()

        ffmpeg, ffprobe = resolve_ffmpeg_tools(self.config.ffmpeg_path)
        self.ffmpeg = ffmpeg
        self.ffprobe = ffprobe

        logger.info(
            "[VideoVision] v%s 已加载 | 启用=%s | 体积闸门=%.1fMB | "
            "压缩上限=%.1fMB | CRF=%s | 超限策略=%s | 仅 Gemini=%s",
            PLUGIN_VERSION,
            self.config.enable,
            self.config.max_size_mb,
            self.config.compress_max_mb,
            self.config.compress_crf,
            ACTION_LABELS.get(
                self.config.oversize_action, self.config.oversize_action
            ),
            self.config.require_gemini_provider,
        )
        logger.info(
            "[VideoVision] ffmpeg=%s | ffprobe=%s | 缓存=%s",
            ffmpeg or "未找到",
            ffprobe or "未找到",
            self.cache_root if self.config.cache_enabled else "已关闭",
        )
        effective_mode = self.config.effective_provider_mode
        if effective_mode == "openai_compat":
            key = self.config.openai_api_key
            # 只露后 4 位。密钥是敏感信息，日志经常被贴到 issue 里，
            # 前 6 位足以还原出「sk-xxx」这类前缀，没必要泄露。
            masked = f"…{key[-4:]}" if len(key) >= 8 else ""
            if not masked:
                masked = "已配置" if key else "未配置"
            logger.info(
                "[VideoVision] 模式=openai_compat | 视觉服务=%s | 模型=%s | "
                "key=%s | 超时=%.0fs",
                self.config.openai_base_url or "未配置",
                self.config.openai_model,
                masked,
                self.config.openai_timeout_sec,
            )
        else:
            logger.info(
                "[VideoVision] 模式=gemini_native | 仅 Gemini provider=%s",
                self.config.require_gemini_provider,
            )
        if self.config.provider_mode != effective_mode:
            logger.warning(
                "[VideoVision] provider_mode=openai_compat 但 openai_base_url / "
                "openai_api_key / openai_model 未配齐，已降级为 gemini_native "
                "（老配置零改动仍可继续用）。要使用 OpenAI 兼容视觉服务，"
                "请把这三项补齐。"
            )
            # 承诺是"只告警一次"：启动这条信息最完整，钩子里不再重复刷
            mark_degradation_warned()
        if not ffmpeg:
            logger.warning(
                "[VideoVision] 未找到 ffmpeg：超限视频将无法压缩，会被直接跳过。"
                "请安装 ffmpeg 或在插件配置里指定 ffmpeg_path。"
            )

    # ------------------------------------------------------------------ 钩子
    @filter.on_llm_request(priority=HOOK_PRIORITY)
    async def on_llm_request(
        self, event: AstrMessageEvent, req: object
    ) -> None:
        """把视频内容块注入本次 LLM 请求。"""
        await handle_llm_request(self, event, req)

    # ------------------------------------------------------------ 调试指令
    @filter.command("videotest", alias={"视频测试"})
    async def cmd_videotest(self, event: AstrMessageEvent, path: GreedyStr):
        """逐步诊断视频处理链路（仅管理员）。

        **``path`` 绝对不能写默认值**。框架的 ``CommandFilter.init_handler_md``
        （``astrbot/core/star/filter/command.py:76-79``）在参数**有默认值**时存的
        是默认值本身（``""``），而 ``validate_and_convert_params``
        （``:101-102``）用 ``param_type_or_default_val is GreedyStr`` 判断贪婪
        参数 —— 所以 0.2.0 的 ``path: GreedyStr = ""`` 让 ``is_greedy`` 恒为
        ``False``，只取**第一个 token**：
        ``/videotest "C:\\Program Files\\a.mp4"`` 会变成 ``C:\\Program`` →
        "文件不存在"。

        现在按框架自身的写法（``astrbot/core/builtin_stars/builtin_commands/
        main.py:40``）去掉默认值；无参数时框架把 ``params[i:]`` 的空列表 join
        成空串传进来，行为与原来的默认值一致。

        无参数时视频只可能来自**本条事件**的消息链，共三种入口（见
        :data:`_USAGE_TEXT`）：同一条消息里的视频、引用（回复）的视频消息
        （``Reply.chain``）、显式路径。**"先单独发视频、再发 /videotest" 不可能
        成立** —— 两条消息是两个独立事件，这条命令执行时上一条消息早已处理完。
        """
        if str(getattr(event, "role", "") or "") != "admin":
            yield event.plain_result("该指令仅管理员可用。")
            return

        target = str(path or "").strip().strip('"').strip("'")
        reason = ""
        if not target:
            target, reason = await self._find_video_in_event(event)

        if not target:
            yield event.plain_result(_videotest_usage(reason))
            return

        try:
            report = await self._diagnose(target, event=event)
        except Exception as exc:  # noqa: BLE001
            logger.error("[VideoVision] /videotest 诊断异常: %s", exc, exc_info=True)
            report = f"❌ 诊断过程异常: {type(exc).__name__}: {exc}"
        yield event.plain_result(report)

    async def _find_video_in_event(self, event: AstrMessageEvent) -> tuple[str, str]:
        """从当前消息或引用消息里找一个视频的本地路径。

        同时处理两种组件形态：``Video``，以及**被当作文件投递的视频**
        （``File``）—— 实测 QQ/NapCat 走的是后者。

        「引用视频」的数据来源是 ``Reply.chain``，不是 ``Reply.message_str``：
        aiocqhttp 适配器收到 ``reply`` 段时会先 ``get_msg``，把被引用消息**整条**
        重新解析成消息链塞进 ``Reply.chain``
        （``aiocqhttp_platform_adapter.py:310-337``）。
        顺带说明为什么引用不影响命令匹配：``CommandFilter.filter`` 只看
        ``event.get_message_str()``（``command.py:191-206``），而带引用的
        ``message_str`` 与不带引用时**完全一样**（适配器只把 ``text`` 段拼进它，
        ``aiocqhttp_platform_adapter.py:245-252,422``）。

        Returns:
            ``(路径, 原因码)``。找到时路径非空、原因码为空串；找不到时路径为空串，
            原因码是 :data:`_VIDEO_REASON_TEXT` 的键 —— 用户因此能拿到一句具体的
            话（"引用的消息里没有视频" / "平台没给被引用内容"），而不是一段无论
            什么原因都一样的用法提示。
        """
        try:
            chain = list(getattr(event.message_obj, "message", None) or [])
        except Exception:  # noqa: BLE001
            return "", "no-media"

        replies = [
            comp for comp in chain if Reply is not None and isinstance(comp, Reply)
        ]
        candidates = list(chain)
        for comp in chain:
            inner = getattr(comp, "chain", None)
            if inner:
                candidates.extend(inner)

        # 先找真正的 Video 组件（语义最明确）
        saw_video = False
        if Video is not None:
            for comp in candidates:
                if not isinstance(comp, Video):
                    continue
                saw_video = True
                try:
                    return await comp.convert_to_file_path(), ""
                except Exception as exc:  # noqa: BLE001
                    logger.debug("[VideoVision] /videotest 解析视频路径失败: %s", exc)

        # 再找「看起来是视频」的 File 组件
        if File is not None and self.config.treat_file_as_video:
            for comp in candidates:
                if not isinstance(comp, File):
                    continue
                name = str(getattr(comp, "name", "") or "")
                if suffix_of(name) not in VIDEO_SUFFIXES:
                    continue
                try:
                    path = await comp.get_file()
                except Exception as exc:  # noqa: BLE001
                    logger.debug("[VideoVision] /videotest 解析文件路径失败: %s", exc)
                    continue
                if path and os.path.exists(path):
                    logger.debug(
                        "[VideoVision] /videotest 从 File 组件找到视频: %s", name
                    )
                    return str(path), ""

        # 引用消息的 chain 为空（适配器 get_msg 失败）或里面没有能用的视频：
        # 按 OneBot 的 get_msg 自己再取一次 ——「引用视频不生效」唯一还有救的分支。
        for reply in replies:
            recovered = await self._resolve_quoted_video_via_onebot(event, reply)
            if recovered:
                return recovered, ""

        if saw_video:
            return "", "video-unreadable"
        if any(not getattr(reply, "chain", None) for reply in replies):
            return "", "quote-content-missing"
        if replies:
            return "", "quote-no-video"
        return "", "no-media"

    async def _resolve_quoted_video_via_onebot(
        self, event: AstrMessageEvent, reply: Any
    ) -> str:
        """``Reply.chain`` 拿不到内容时，按 OneBot 的 ``get_msg`` 再取一次视频。

        适配器在收消息时已经 ``get_msg`` 过一次
        （``aiocqhttp_platform_adapter.py:310-326``）；偶发失败（NapCat 超时、
        视频 CDN 链接还没就绪）时 ``Reply.chain`` 是空的，于是引用里的视频对插件
        **完全不可见** —— 用户看到的表现就是"引用视频发 /videotest 没反应"。

        ``event.bot.api.call_action`` 正是框架自己的 ``OneBotClient`` 用的入口
        （``core/utils/quoted_message/onebot_client.py:35-41``），所以这里不需要
        任何平台判断：不是 OneBot 平台就取不到 ``call_action``，直接跳过。

        只认 ``video`` 段：``file`` 段的取址要额外的 ``get_private_file_url`` /
        ``get_group_file_url``，而那是适配器已经替我们试过的一步。

        Args:
            event: 当前消息事件。
            reply: 引用组件（只用它的 ``id``）。

        Returns:
            视频本地路径；取不到时返回空串（绝不抛异常、绝不拖慢命令）。
        """
        call_action = getattr(
            getattr(getattr(event, "bot", None), "api", None), "call_action", None
        )
        message_id = getattr(reply, "id", None)
        if not callable(call_action) or message_id in (None, "") or Video is None:
            return ""

        payload: Any = None
        for params in ({"message_id": message_id}, {"id": message_id}):
            try:
                payload = await call_action("get_msg", **params)
            except Exception as exc:  # noqa: BLE001
                logger.debug("[VideoVision] /videotest 取引用消息失败: %s", exc)
                payload = None
                continue
            if isinstance(payload, dict):
                break
        if not isinstance(payload, dict):
            return ""

        data = payload.get("data")
        data = data if isinstance(data, dict) else payload
        segments = data.get("message")
        if not isinstance(segments, list):
            return ""

        for segment in segments:
            if not isinstance(segment, dict) or segment.get("type") != "video":
                continue
            info = segment.get("data")
            info = info if isinstance(info, dict) else {}
            ref = str(info.get("file") or info.get("url") or "")
            if not ref:
                continue
            try:
                video = Video(
                    file=ref,
                    url=str(info.get("url") or ""),
                    path=str(info.get("path") or ""),
                )
                path = await video.convert_to_file_path()
            except Exception as exc:  # noqa: BLE001
                logger.debug("[VideoVision] /videotest 重取引用视频失败: %s", exc)
                continue
            if path:
                logger.info("[VideoVision] /videotest 从引用消息重新取到视频: %s", ref)
                return str(path)
        return ""

    async def _diagnose(self, target: str, event: Any | None = None) -> str:
        """跑一遍完整链路并输出逐步报告。

        gemini_native 模式不发请求；openai_compat 模式会真实调用视觉服务。

        两个耗时段（``probe_duration`` 与 ``prepare_video``）都通过
        ``asyncio.to_thread`` 执行：它们是阻塞子进程，直接在事件循环里跑会把
        整个 AstrBot 卡住（实测同步诊断 0.5s 会把一个在飞的钩子从 0.30s
        拖到 0.50s，心跳最大间隔 625ms）。

        Args:
            target: 视频文件路径。
            event: 当前消息事件，用于查 provider 类型 / 模态；可为 None。

        Returns:
            多行可读报告（直接回给管理员）。
        """
        cfg = self.config
        lines: list[str] = ["🔍 视频链路诊断"]

        if not os.path.exists(target):
            lines.append(f"❌ 文件不存在: {target}")
            return "\n".join(lines)
        if not os.path.isfile(target):
            lines.append(f"❌ 不是文件: {target}")
            return "\n".join(lines)

        size_mb = file_size_mb(target)
        duration = await asyncio.to_thread(probe_duration, target, self.ffprobe)

        lines.append(f"文件: {os.path.basename(target)}")
        lines.append(f"体积: {size_mb:.2f} MB")
        lines.append(f"时长: {duration:.1f} s" if duration > 0 else "时长: 读取失败")
        lines.append("")
        media_ok = "❌ 未找到"
        lines.append(
            f"ffmpeg : {'✅ ' + self.ffmpeg if self.ffmpeg else media_ok}"
        )
        lines.append(
            f"ffprobe: {'✅ ' + self.ffprobe if self.ffprobe else media_ok}"
        )
        lines.append(
            f"配置   : 闸门={cfg.max_size_mb:.1f}MB 上限={cfg.compress_max_mb:.1f}MB "
            f"CRF={cfg.compress_crf} "
            f"策略={ACTION_LABELS.get(cfg.oversize_action, cfg.oversize_action)}"
        )
        # 配置被自动修正过时，把"你写的原值 → 生效值"一起打出来。
        for notice in getattr(self, "config_notices", []):
            lines.append(f"配置提示: {notice}")
        for warning in getattr(self, "config_warnings", []):
            lines.append(f"配置修正: {warning}")
        # 与体积闸门无关的第二道体积限制：data URI 编码上限。
        lines.append(
            f"编码上限: data URI {URI_LIMIT_MB:.0f}MB ≈ 原始 {URI_LIMIT_RAW_MB:.1f}MB"
            "（编码后超过必失败，与体积闸门无关）"
        )

        # ---- 模式 / 凭据（0.2.0 只报模式）----
        effective_mode = cfg.effective_provider_mode
        degraded = cfg.provider_mode != effective_mode
        lines.append(
            f"模式   : {effective_mode}"
            + (
                f"（provider_mode={cfg.provider_mode} 三项未配齐，运行时降级）"
                if degraded
                else ""
            )
        )
        lines.append(f"凭据   : 视觉服务{_credentials_summary(cfg)}")

        # ---- provider 与三道闸门（0.2.0 完全不诊断，而这正是最常见的失败点）----
        lines.extend(await self._diagnose_gates(cfg, event, effective_mode))

        # ---- 显式路径的扩展名风险（闸门 ⓪ 在本指令里不适用）----
        container_note = _unknown_container_note(target)
        if container_note:
            lines.append(f"提醒   : {container_note}")

        # ---- 体积 / 时长判定：只渲染 decide 的判决，绝不在这里重算 ----
        # 0.2.1 这段是**手工预测** injector 会怎么判（注释自己写着"必须与 injector
        # 完全一致"，corrupt / 编码上限 / overlong 的 duration>0 守卫都是靠人肉同步
        # 的）。现在两侧吃同一个 MediaPlan：判定行与实际结果结构上不可能再分叉
        # （真机证据见 LIVE-TEST §10.2②）。
        plan = classify(
            size_mb=size_mb,
            duration_sec=duration,
            config=cfg,
            mime_or_suffix=suffix_of(target),
            has_ffmpeg=bool(self.ffmpeg),
            has_ffprobe=bool(self.ffprobe),
        )
        lines.append(f"判定   : {describe(plan)}")

        # ---- 将要发给视觉服务的 prompt 预览（截断）----
        lines.append(f"prompt : {self._prompt_preview(cfg, effective_mode)}")

        lines.append("")
        started = time.monotonic()
        outcome = await asyncio.to_thread(
            prepare_video,
            VideoPlaceholder(name=os.path.basename(target), path=target),
            config=cfg,
            cache_root=self.cache_root,
            logger=logger,
            debug=cfg.debug_log,
        )
        elapsed = time.monotonic() - started

        if not outcome.ok or outcome.injection is None:
            lines.append(f"❌ 准备失败（{elapsed:.2f}s）")
            if outcome.kind:
                lines.append(f"分类: {outcome.kind}")
            lines.append(f"原因: {outcome.reason}")
            return "\n".join(lines)

        inj = outcome.injection
        uri_mb = len(inj.block["audio_url"]["url"]) / 1024 / 1024
        est_upload = inj.size_mb * 8  # 实测约 7-9 秒/MB
        is_openai = effective_mode == "openai_compat"
        budget = cfg.openai_timeout_sec if is_openai else 180.0
        budget_name = "视觉服务预算" if is_openai else "provider 预算"

        lines.append(f"✅ 准备成功（{elapsed:.2f}s）")
        lines.append(f"动作: {ACTION_LABELS.get(inj.action, inj.action)}")
        lines.append(f"产物: {inj.size_mb:.2f} MB / {inj.duration_sec:.1f} s")
        lines.append(f"编码: data URI {uri_mb:.2f} MB")
        # 声明给 provider 的 MIME：未知后缀回落 video/mp4 时标 (推断)，
        # 免得"✅ 准备成功 + video/mp4"被读成"这个 gif 真的会以视频发出去"。
        lines.append(_mime_line(inj.used_path, inj.action))
        if inj.note:
            lines.append(f"说明: {inj.note}")
        lines.append("")
        lines.append(
            f"预计上传耗时: 约 {est_upload:.0f}s（{budget_name} {budget:.0f}s）"
        )
        if est_upload > budget:
            hint = "openai_timeout_sec" if is_openai else "provider timeout"
            lines.append(f"⚠️ 可能超时，建议调低体积闸门或提高 {hint}")
        else:
            lines.append("✅ 在预算内")

        # openai_compat：真实调用一次视觉服务，端到端验证配置与网络
        if is_openai:
            lines.append("")
            prompt = build_vision_prompt(cfg, "", logger=logger)
            result = await analyze_video(
                inj.data_uri,
                os.path.basename(target),
                prompt,
                config=cfg,
                logger=logger,
            )
            if result.ok:
                lines.append(
                    f"✅ 视觉调用成功（{result.elapsed_sec:.1f}s | "
                    f"token {result.prompt_tokens}+{result.completion_tokens}）"
                )
                excerpt = result.analysis[:400] + (
                    "…" if len(result.analysis) > 400 else ""
                )
                lines.append("— 视觉模型回复节选 —")
                lines.append(excerpt)
            else:
                lines.append(f"❌ 视觉调用失败: {result.reason}")
        return "\n".join(lines)

    async def _diagnose_gates(
        self, cfg: VideoConfig, event: Any | None, mode: str
    ) -> list[str]:
        """输出 provider 信息与三道闸门的判定结果。

        判定逻辑与 ``hooks/llm_request._handle`` **完全一致**（同样的允许列表、
        同样的 modalities 规则），避免"诊断说的"和"实际跑的"不一样。

        Args:
            cfg: 当前配置。
            event: 消息事件，用于 ``get_using_provider_async(umo=...)``。
            mode: ``effective_provider_mode``。

        Returns:
            若干行文本。
        """
        gate0 = "⓪ File 组件扩展名筛选 → 本指令用显式路径，不适用"
        if mode == "openai_compat":
            return [
                "provider: 不适用（插件直连视觉服务，不经过主对话 provider）",
                f"闸门   : {gate0}",
                "         ① Gemini 系 provider 校验 → openai_compat 不适用",
                "         ② audio 模态校验 → openai_compat 不适用",
                f"         ③ 视觉服务凭据{_credentials_summary(cfg)}",
            ]

        provider, provider_error = await resolve_provider(self, event)
        provider_config = getattr(provider, "provider_config", None) or {}
        provider_type = str(provider_config.get("type") or "")
        modalities = provider_config.get("modalities")
        declared = (
            ", ".join(str(item) for item in modalities)
            if isinstance(modalities, list) and modalities
            else "未声明"
        )
        lines = [
            f"provider: type={provider_type or '未声明'} "
            f"id={provider_config.get('id') or '未声明'} "
            f"modalities=[{declared}]"
        ]
        allowed = GEMINI_PROVIDER_TYPES | set(cfg.additional_provider_types)
        if provider is None:
            gate1 = f"① provider 解析失败 ❌（{provider_error or '未知'}）"
        elif not cfg.require_gemini_provider:
            gate1 = (
                f"① require_gemini_provider=false → 放行"
                f"（类型 {provider_type or '未声明'}，不校验 MIME 兼容性）"
            )
        elif provider_type in allowed:
            gate1 = f"① 类型 {provider_type} ✅ 在允许列表"
        else:
            gate1 = (
                f"① 类型 {provider_type or '未声明'} ❌ 不在允许列表 "
                f"{sorted(allowed)}"
            )
        if not isinstance(modalities, list) or not modalities:
            gate2 = "② modalities 未声明 → ⚠️ 放行（无法校验，视频块可能被清洗）"
        elif "audio" in modalities:
            gate2 = "② modalities 含 audio ✅"
        else:
            gate2 = "② modalities 不含 audio ❌（视频块会被清洗成 [Audio]）"
        gate3 = f"③ 视觉服务凭据{_credentials_summary(cfg)}（本模式不使用）"
        return [
            *lines,
            f"闸门   : {gate0}",
            f"         {gate1}",
            f"         {gate2}",
            f"         {gate3}",
        ]

    @staticmethod
    def _prompt_preview(cfg: VideoConfig, mode: str) -> str:
        """将要发送（或注入）的 prompt 预览，压成一行并截断。"""
        if mode != "openai_compat":
            return "（gemini_native 不调用视觉服务，无 prompt）"
        prompt = build_vision_prompt(cfg, "", logger=logger)
        flat = " ".join(prompt.split())
        suffix = ""
        if "{question}" not in (cfg.vision_qa_prompt or ""):
            suffix = "（⚠️ vision_qa_prompt 缺 {question}，用户问题会被追加到末尾）"
        return flat[:200] + ("…" if len(flat) > 200 else "") + suffix

    async def terminate(self) -> None:
        """插件被停用/卸载时的清理。"""
        logger.info("[VideoVision] 插件已终止")
