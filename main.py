"""「基米看得见视频」—— AstrBot 插件入口。

两种工作模式（由 ``provider_mode`` 配置切换）：

- ``gemini_native``：在 ``on_llm_request`` 阶段把视频转成内容块注入当前请求，
  让 Gemini 原生"看见"画面（要求主对话 provider 是 Gemini 系）。
- ``openai_compat``：插件直接把视频发给配置里的 OpenAI 兼容视觉服务做理解，
  再把理解结果以文字注入对话 —— 主对话模型可以是任意模型。

本文件只保留**插件壳**：插件类、命令注册、事件/配置装配，以及把事实喂给
``core.diagnostics``。判决在 ``core/decide.py``，执行在 ``core/injector.py``，
所有用户可见文本在 ``core/diagnostics.py``。

为什么不覆写 provider 的 ``assemble_context``：详见 ``hooks/llm_request.py``
的模块说明（那里是唯一一份完整论证）。
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

from .core import diagnostics
from .core.config import VideoConfig
from .core.decide import classify
from .core.injector import prepare_video
from .core.media import (
    file_size_mb,
    probe_duration,
    resolve_ffmpeg_tools,
    suffix_of,
)
from .core.openai_vision import analyze_video, build_vision_prompt
from .core.placeholder import VIDEO_SUFFIXES, VideoPlaceholder
from .hooks.llm_request import (
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


def _resolve_cache_root() -> str:
    """压缩缓存目录，放在 AstrBot 数据目录下的 plugin_data 里。"""
    try:
        from astrbot.core.utils.astrbot_path import get_astrbot_data_path

        base = get_astrbot_data_path()
    except Exception:  # pragma: no cover - 版本差异兜底
        base = os.path.join(os.path.expanduser("~"), ".astrbot", "data")
    return os.path.join(base, "plugin_data", PLUGIN_NAME, "cache")


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
        #: 知道自己写的 ``compress_max_mb=15.0`` 最终生效成了多少。
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

        # 启动日志的**文本**在 diagnostics（纯函数，可单测），按级别打出去与
        # 下面的状态变更留在这里。
        for level, message in diagnostics.startup_lines(
            self.config,
            version=PLUGIN_VERSION,
            ffmpeg=ffmpeg or "",
            ffprobe=ffprobe or "",
            cache_root=self.cache_root,
        ):
            if level == "warning":
                logger.warning(message)
            else:
                logger.info(message)
        if self.config.provider_mode != self.config.effective_provider_mode:
            # 承诺是"只告警一次"：启动这条信息最完整，钩子里不再重复刷
            mark_degradation_warned()

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

        **``path`` 绝对不能写默认值**：框架的 ``CommandFilter.init_handler_md``
        （``astrbot/core/star/filter/command.py:76-79``）在参数**有默认值**时存的
        是默认值本身（``""``），而 ``validate_and_convert_params``（``:101-102``）
        用 ``param_type_or_default_val is GreedyStr`` 判断贪婪参数 —— 于是
        ``path: GreedyStr = ""`` 让 ``is_greedy`` 恒为 ``False``，只取第一个
        token，带空格的路径被切碎（0.2.0 的 bug，两条用例锁在
        ``tests/test_main.py``）。无参数时框架把空列表 join 成空串传进来，
        行为与原来的默认值一致。

        无参数时视频只可能来自**本条事件**的消息链，三条入口见
        :data:`core.diagnostics.USAGE_TEXT`。
        """
        if str(getattr(event, "role", "") or "") != "admin":
            yield event.plain_result("该指令仅管理员可用。")
            return

        target = str(path or "").strip().strip('"').strip("'")
        reason = ""
        if not target:
            target, reason = await self._find_video_in_event(event)

        if not target:
            yield event.plain_result(diagnostics.videotest_usage(reason))
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

        Returns:
            ``(路径, 原因码)``。找到时路径非空、原因码为空串；找不到时路径为空串，
            原因码是 :data:`core.diagnostics.VIDEO_REASON_TEXT` 的键 —— 用户因此能
            拿到一句具体的话（"引用的消息里没有视频" / "平台没给被引用内容"），
            而不是一段无论什么原因都一样的用法提示。
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

        适配器在收消息时已经 ``get_msg`` 过一次；偶发失败（NapCat 超时、视频
        CDN 链接还没就绪）时 ``Reply.chain`` 是空的，于是引用里的视频对插件
        **完全不可见** —— 用户看到的表现就是"引用视频发 /videotest 没反应"。

        ``event.bot.api.call_action`` 正是框架自己的 ``OneBotClient`` 用的入口，
        所以这里不需要任何平台判断：不是 OneBot 平台就取不到 ``call_action``，
        直接跳过。只认 ``video`` 段：``file`` 段的取址要额外的
        ``get_private_file_url`` / ``get_group_file_url``，而那是适配器已经替
        我们试过的一步。

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

    # -------------------------------------------------------------- 诊断
    async def _diagnose(self, target: str, event: Any | None = None) -> str:
        """跑一遍完整链路并输出逐步报告（渲染全部在 ``core.diagnostics``）。

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
        # 前置条件先拦掉：文件都不存在就没有"事实"可诊断，报告只有两行。
        if not os.path.exists(target):
            return f"{diagnostics.TITLE}\n❌ 文件不存在: {target}"
        if not os.path.isfile(target):
            return f"{diagnostics.TITLE}\n❌ 不是文件: {target}"

        size_mb = file_size_mb(target)
        duration = await asyncio.to_thread(probe_duration, target, self.ffprobe)
        mode = cfg.effective_provider_mode
        # provider 解析留在 hooks（要 plugin.context），结果作为纯数据交给渲染层
        provider = await self._provider_facts(event, mode)
        probe = diagnostics.ProbeFacts(
            path=target,
            size_mb=size_mb,
            duration_sec=duration,
            ffmpeg=self.ffmpeg or "",
            ffprobe=self.ffprobe or "",
        )
        # 判定与实际执行吃同一个 MediaPlan（``core/decide.py``）：判定行结构上
        # 不可能再与下面的真实结果分叉（真机证据见 LIVE-TEST §10.2②）。
        plan = classify(
            size_mb=size_mb,
            duration_sec=duration,
            config=cfg,
            mime_or_suffix=suffix_of(target),
            has_ffmpeg=bool(self.ffmpeg),
            has_ffprobe=bool(self.ffprobe),
        )

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

        vision = None
        # openai_compat：真实调用一次视觉服务，端到端验证配置与网络
        if mode == "openai_compat" and outcome.ok and outcome.injection is not None:
            prompt = build_vision_prompt(cfg, "", logger=logger)
            vision = diagnostics.VisionReport.of(
                await analyze_video(
                    outcome.injection.data_uri,
                    os.path.basename(target),
                    prompt,
                    config=cfg,
                    logger=logger,
                )
            )

        return diagnostics.render_report(
            cfg=cfg,
            mode=mode,
            probe=probe,
            plan=plan,
            provider=provider,
            notices=self.config_notices,
            warnings=self.config_warnings,
            outcome=outcome,
            elapsed_sec=elapsed,
            vision=vision,
        )

    async def _provider_facts(
        self, event: Any | None, mode: str
    ) -> diagnostics.ProviderFacts | None:
        """把「当前 provider」翻译成渲染层要的纯数据。

        provider 解析留在 ``hooks``（需要 ``plugin.context``），``core`` 只接收
        三个纯数据字段 —— 层次方向见 ARCH-REVIEW §2.4。
        """
        if mode == "openai_compat":
            return None  # 本模式不经过主对话 provider，三道闸门都不适用
        provider, error = await resolve_provider(self, event)
        if provider is None:
            return diagnostics.ProviderFacts(error=error or "未知")
        provider_config = getattr(provider, "provider_config", None) or {}
        modalities = provider_config.get("modalities")
        return diagnostics.ProviderFacts(
            provider_type=str(provider_config.get("type") or ""),
            provider_id=str(provider_config.get("id") or ""),
            modalities=modalities if isinstance(modalities, list) else None,
        )

    async def terminate(self) -> None:
        """插件被停用/卸载时的清理。"""
        logger.info("[VideoVision] 插件已终止")
