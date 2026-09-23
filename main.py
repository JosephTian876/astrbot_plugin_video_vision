"""「基米看得见视频」—— AstrBot 插件入口。

AstrBot 核心目前只把视频写成一行文本占位符（``[Video Attachment: name ..., path ...]``）
交给模型，模型看不到任何画面。本插件在 ``on_llm_request`` 阶段把视频转成内容块
追加进请求，使 Gemini 真正"看见"画面。

为什么不覆写 provider 的 ``assemble_context``：主对话轮次的上下文由
``ProviderRequest.assemble_context()`` 组装，runner 传 ``contexts=`` 且
``prompt=None``，而 provider 侧方法只在 ``prompt is not None`` 时才被调用。
详见 ``hooks/llm_request.py`` 的模块说明。
"""

from __future__ import annotations

import os
import time

from astrbot.api import logger
from astrbot.api.event import AstrMessageEvent, filter
from astrbot.api.star import Context, Star, register

try:
    from astrbot.api.message_components import File, Video
except ImportError:  # pragma: no cover
    Video = None  # type: ignore[assignment]
    File = None  # type: ignore[assignment]

try:
    from astrbot.core.star.filter.command import GreedyStr
except ImportError:  # pragma: no cover
    GreedyStr = str  # type: ignore[assignment,misc]

from .core.config import VideoConfig
from .core.injector import prepare_video
from .core.media import file_size_mb, probe_duration, resolve_ffmpeg_tools
from .core.placeholder import VIDEO_SUFFIXES, VideoPlaceholder
from .hooks.llm_request import handle_llm_request

PLUGIN_NAME = "astrbot_plugin_video_vision"
PLUGIN_VERSION = "0.1.0"
PLUGIN_REPO = "https://github.com/JosephTian876/astrbot_plugin_video_vision"

#: 比其他插件都晚执行（数值越小越晚），确保陪伴插件改写完请求后我们再追加。
HOOK_PRIORITY = -300000

_ACTION_LABELS = {
    "compress": "压缩",
    "reject": "拒绝",
    "truncate": "截断",
    "as-is": "直接送（未压缩）",
    "compressed": "已压缩",
    "truncated": "已截断",
    "rejected": "已拒绝",
}


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
    "让 AstrBot 把视频作为原生模态直接交给 Gemini 理解，而不是抽取关键帧。",
    PLUGIN_VERSION,
    PLUGIN_REPO,
)
class VideoVisionPlugin(Star):
    """视频原生输入插件。"""

    def __init__(self, context: Context, config: dict | None = None):
        super().__init__(context)

        self.raw_config: dict = dict(config or {})
        resolved, warnings = VideoConfig.from_mapping(self.raw_config).normalized()
        self.config: VideoConfig = resolved
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
            _ACTION_LABELS.get(
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
    async def cmd_videotest(self, event: AstrMessageEvent, path: GreedyStr = ""):
        """逐步诊断视频处理链路（仅管理员）。"""
        if str(getattr(event, "role", "") or "") != "admin":
            yield event.plain_result("该指令仅管理员可用。")
            return

        target = str(path or "").strip().strip('"').strip("'")
        if not target:
            target = await self._find_video_in_event(event)

        if not target:
            yield event.plain_result(
                "用法：\n"
                "  /videotest <视频文件绝对路径>\n"
                "或先发一个视频/视频文件，再发送 /videotest"
            )
            return

        try:
            report = self._diagnose(target)
        except Exception as exc:  # noqa: BLE001
            logger.error("[VideoVision] /videotest 诊断异常: %s", exc, exc_info=True)
            report = f"❌ 诊断过程异常: {type(exc).__name__}: {exc}"
        yield event.plain_result(report)

    async def _find_video_in_event(self, event: AstrMessageEvent) -> str:
        """从当前消息或引用消息里找一个视频的本地路径。

        同时处理两种组件形态：``Video``，以及**被当作文件投递的视频**
        （``File``）—— 实测 QQ/NapCat 走的是后者。
        """
        try:
            chain = list(getattr(event.message_obj, "message", None) or [])
        except Exception:
            return ""

        candidates = list(chain)
        for comp in chain:
            inner = getattr(comp, "chain", None)
            if inner:
                candidates.extend(inner)

        # 先找真正的 Video 组件（语义最明确）
        if Video is not None:
            for comp in candidates:
                if not isinstance(comp, Video):
                    continue
                try:
                    return await comp.convert_to_file_path()
                except Exception as exc:  # noqa: BLE001
                    logger.debug("[VideoVision] /videotest 解析视频路径失败: %s", exc)

        # 再找「看起来是视频」的 File 组件
        if File is not None and self.config.treat_file_as_video:
            for comp in candidates:
                if not isinstance(comp, File):
                    continue
                name = str(getattr(comp, "name", "") or "")
                if os.path.splitext(name)[1].lower() not in VIDEO_SUFFIXES:
                    continue
                try:
                    path = await comp.get_file()
                except Exception as exc:  # noqa: BLE001
                    logger.debug("[VideoVision] /videotest 解析文件路径失败: %s", exc)
                    continue
                if path and os.path.exists(path):
                    logger.debug("[VideoVision] /videotest 从 File 组件找到视频: %s", name)
                    return str(path)
        return ""

    def _diagnose(self, target: str) -> str:
        """跑一遍完整链路并输出逐步报告（不真正发起模型请求）。"""
        cfg = self.config
        lines: list[str] = ["🔍 视频链路诊断"]

        if not os.path.exists(target):
            lines.append(f"❌ 文件不存在: {target}")
            return "\n".join(lines)
        if not os.path.isfile(target):
            lines.append(f"❌ 不是文件: {target}")
            return "\n".join(lines)

        size_mb = file_size_mb(target)
        duration = probe_duration(target, self.ffprobe)

        lines.append(f"文件: {os.path.basename(target)}")
        lines.append(f"体积: {size_mb:.2f} MB")
        lines.append(f"时长: {duration:.1f} s" if duration > 0 else "时长: 读取失败")
        lines.append("")
        lines.append(f"ffmpeg : {'✅ ' + self.ffmpeg if self.ffmpeg else '❌ 未找到'}")
        lines.append(f"ffprobe: {'✅ ' + self.ffprobe if self.ffprobe else '❌ 未找到'}")
        lines.append(
            f"配置   : 闸门={cfg.max_size_mb:.1f}MB 上限={cfg.compress_max_mb:.1f}MB "
            f"CRF={cfg.compress_crf} "
            f"策略={_ACTION_LABELS.get(cfg.oversize_action, cfg.oversize_action)}"
        )

        oversize = size_mb > cfg.max_size_mb
        overlong = cfg.max_duration_sec > 0 and duration > cfg.max_duration_sec
        if not oversize and not overlong:
            lines.append("判定   : 未超限，直接送")
        else:
            why = []
            if oversize:
                why.append(f"体积 {size_mb:.1f}MB > 闸门 {cfg.max_size_mb:.1f}MB")
            if overlong:
                why.append(f"时长 {duration:.1f}s > 上限 {cfg.max_duration_sec:.1f}s")
            label = _ACTION_LABELS.get(cfg.oversize_action, cfg.oversize_action)
            lines.append(f"判定   : {'；'.join(why)} → {label}")

        lines.append("")
        started = time.monotonic()
        outcome = prepare_video(
            VideoPlaceholder(name=os.path.basename(target), path=target),
            config=cfg,
            cache_root=self.cache_root,
            logger=logger,
        )
        elapsed = time.monotonic() - started

        if not outcome.ok or outcome.injection is None:
            lines.append(f"❌ 准备失败（{elapsed:.2f}s）")
            lines.append(f"原因: {outcome.reason}")
            return "\n".join(lines)

        inj = outcome.injection
        uri_mb = len(inj.block["audio_url"]["url"]) / 1024 / 1024
        est_upload = inj.size_mb * 8  # 实测约 7-9 秒/MB
        budget = 180.0

        lines.append(f"✅ 准备成功（{elapsed:.2f}s）")
        lines.append(f"动作: {_ACTION_LABELS.get(inj.action, inj.action)}")
        lines.append(f"产物: {inj.size_mb:.2f} MB / {inj.duration_sec:.1f} s")
        lines.append(f"编码: data URI {uri_mb:.2f} MB")
        if inj.note:
            lines.append(f"说明: {inj.note}")
        lines.append("")
        lines.append(f"预计上传耗时: 约 {est_upload:.0f}s（provider 预算 {budget:.0f}s）")
        if est_upload > budget:
            lines.append("⚠️ 可能超时，建议调低体积闸门或提高 provider timeout")
        else:
            lines.append("✅ 在预算内")
        return "\n".join(lines)

    async def terminate(self) -> None:
        """插件被停用/卸载时的清理。"""
        logger.info("[VideoVision] 插件已终止")
