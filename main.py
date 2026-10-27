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
from astrbot.api.star import Context, Star

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
from .core.injector import PrepareOutcome, prepare_video
from .core.limits import NULL_LIMITER, QueueTimeout, SlotLimiter, format_seconds
from .core.media import (
    file_size_mb,
    probe_duration,
    resolve_ffmpeg_tools,
    suffix_of,
)
from .core.openai_vision import analyze_video, build_vision_prompt
from .core.placeholder import VIDEO_SUFFIXES, VideoPlaceholder
from .core.vision_cache import VisionCache
from .hooks.llm_request import (
    handle_llm_request,
    mark_degradation_warned,
    resolve_provider,
)

PLUGIN_NAME = "astrbot_plugin_video_vision"
#: 插件的显示名 / 简介 / 版本 / 仓库 / 作者**只在** ``metadata.yaml`` 里维护
#: （0.3.0 删掉 ``@register`` 之后，代码里不再有第二份副本可以漂移）。


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


#: 插件的身份**只**来自 ``metadata.yaml``（AstrBot 4.29.1 起不需要 ``@register``）：
#: ``Star.__init_subclass__`` 在类创建时就把本模块登记进 ``star_map``，
#: ``StarManager.load()`` 再用 ``metadata.yaml`` 的字段逐个覆盖
#: （``star_manager.py:1182-1196``，源码注释原文"yaml 文件的元数据优先"）。
#:
#: 为什么不留那个装饰器：``core/star/register/star.py:6-46`` 文件头就写着
#: ``[DEPRECATED]``，并会在首次使用时 ``warnings.warn(DeprecationWarning)``。
#: 唯一回归面：``metadata.yaml`` 整个丢失时身份退化为 ``unknown/unknown``
#: （插件仍能加载）—— 所以 ``tests/test_docs.py`` 锁住四个必填字段。
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

        #: ffmpeg 并发限流（**进程内全局、跨会话**）。
        #: ``ffmpeg_max_concurrency=0`` -> 不建这个对象，钩子用 ``NULL_LIMITER``
        #: （不建令牌队列、不加任何 await、日志零新增）—— 逐字节回到 0.2.1。
        self.limiter: SlotLimiter | None = (
            SlotLimiter(
                self.config.ffmpeg_max_concurrency,
                queue_timeout=self.config.ffmpeg_queue_timeout_sec,
                logger=logger,
                debug=self.config.debug_log,
            )
            if self.config.ffmpeg_max_concurrency > 0
            else None
        )
        if self.limiter is not None:
            logger.info(
                "[VideoVision] ffmpeg 并发上限=%d，排队上限=%s"
                "（原先的上限只是默认线程池的 min(32, cpu+4)）",
                self.limiter.limit,
                format_seconds(self.limiter.queue_timeout),
            )

        #: ``ffmpeg_singleflight=true`` 但没有压缩缓存 = 自相矛盾：没有可共享的
        #: 产物，flight 只能被跳过（``core/injector._prepare_video`` 里逐视频
        #: 打一条 debug）。**启动时告警一次**：逐视频刷 WARNING 会淹没日志，
        #: 静默跳过又会让用户以为"开了但没生效"（AUDIT-030 §C.1-7）。
        self.singleflight_effective = bool(
            self.config.ffmpeg_singleflight
            and self.config.cache_enabled
            and self.cache_root
        )
        if self.config.ffmpeg_singleflight and not self.singleflight_effective:
            logger.warning(
                "[VideoVision] ffmpeg_singleflight=true，但没有可用的压缩缓存"
                "（cache_enabled=%s，cache_root=%s）：同一文件的并发转码**不会**被"
                "合并（每次各压一份，与 0.2.1 相同）。要生效请开启 cache_enabled；"
                "想消除这条提示就把 ffmpeg_singleflight 设为 false。",
                self.config.cache_enabled,
                "已配置" if self.cache_root else "空",
            )

        ffmpeg, ffprobe = resolve_ffmpeg_tools(self.config.ffmpeg_path)
        self.ffmpeg = ffmpeg
        self.ffprobe = ffprobe

        #: 视觉理解结果缓存（**纯内存**、TTL + LRU）。
        #: ``vision_cache_enabled=false`` -> 不建对象，钩子拿不到缓存 ->
        #: 不查不写、``content_key`` 也不算 —— 逐字节回到 0.2.1。
        #: 只对"完全相同"的重复投递生效：同一份视频字节 + 同一最终 prompt +
        #: 同一端点 + 同一模型 + 同一 max_tokens（追问换了提问不会命中，这是设计）。
        self.vision_cache: VisionCache | None = None
        if self.config.vision_cache_enabled:
            self.vision_cache = VisionCache(
                ttl_sec=self.config.vision_cache_ttl_sec,
                max_entries=self.config.vision_cache_max_entries,
            )
            # 文案必须诚实：gemini_native 走「视频块交给主 provider 亲眼看」，
            # 根本不产生可缓存的文本结果 —— 不能在这种模式下宣称"缓存已启用"
            # （VERIFY-B P3：实测该模式投递一次后 hits/misses/stores 全 0）。
            if self.config.effective_provider_mode == "openai_compat":
                logger.info(
                    "[VideoVision] 视觉结果缓存已启用：TTL=%s，最多 %d 条"
                    "（纯内存、重启即失效；只对完全相同的重复投递命中，"
                    "同一 key 的并发投递只发一次请求）",
                    format_seconds(self.vision_cache.ttl_sec),
                    self.vision_cache.max_entries,
                )
            else:
                logger.info(
                    "[VideoVision] 视觉结果缓存已配置，但当前模式=%s 不使用它"
                    "（仅 openai_compat 生效：gemini_native 没有文本结果可缓存）",
                    self.config.effective_provider_mode,
                )

        #: ``vision_cache_enabled=true`` + ``cache_enabled=false``：缓存**仍然生效**
        #: （key 取"源内容哈希 + 压缩参数"，不依赖压缩缓存，ARCH-VV-C2 P1-1），
        #: 但需要压缩的视频**每次投递都要重新压一遍** —— 缓存省掉的是上传 + 推理，
        #: 省不掉本地 ffmpeg。启动时告警一次（逐视频刷 WARNING 会淹没日志，静默又
        #: 会让用户以为"开了但没用"，与上面 singleflight 那条同一个模板）。
        if (
            self.config.vision_cache_enabled
            and not self.config.cache_enabled
            and self.config.effective_provider_mode == "openai_compat"
        ):
            logger.warning(
                "[VideoVision] vision_cache_enabled=true，但压缩缓存已关闭"
                "（cache_enabled=%s，cache_root=%s）：视觉结果缓存**仍然生效**"
                "（key 取「源内容哈希 + 压缩参数」，不依赖压缩缓存），但需要压缩的"
                "视频**每次投递都要重新压一遍**（缓存省掉的是上传 + 推理，省不掉"
                "本地压缩）。想连本地压缩一起省掉请开启 cache_enabled；想消除这条"
                "提示就把 vision_cache_enabled 设为 false。",
                self.config.cache_enabled,
                "已配置" if self.cache_root else "空",
            )

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
        # 判决与实际执行吃同一个 MediaPlan（``core/decide.py``）：结论行结构上
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
        slots = self.limiter if self.limiter is not None else NULL_LIMITER
        try:
            # 计时起点在**取槽之前**：报告里的"准备成功（X s）"必须包含排队时间，
            # 否则高峰期会出现"报告 1.8s、用户实际等了 12s"的静默谎言。
            # 还槽绑定到 worker **真正**结束（同 hooks：取消不许提前还槽）。
            async with slots.slot(name=os.path.basename(target)) as slot:
                outcome = await slot.wait_for_worker(
                    asyncio.to_thread(
                        prepare_video,
                        VideoPlaceholder(name=os.path.basename(target), path=target),
                        config=cfg,
                        cache_root=self.cache_root,
                        logger=logger,
                        debug=cfg.debug_log,
                    )
                )
        except QueueTimeout as exc:
            outcome = PrepareOutcome(
                False,
                # 同 hooks：文案走 ``QueueTimeout.short_reason``，不各写一份格式串
                reason=exc.short_reason(),
                kind="busy",
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

        limiter_stats = getattr(self.limiter, "stats", None)
        queue_summary = (
            limiter_stats.summary() if limiter_stats is not None else ""
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
            # 有排队/超时才多一行（``LimiterStats.summary()`` 为空串 = 不渲染）。
            # 用 getattr：限流器可能是鸭子类型的替身（测试/外部注入），没有 stats
            # 时"没有可观测数据"= 不渲染，绝不能让诊断因此失败。
            queue_summary=queue_summary,
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
