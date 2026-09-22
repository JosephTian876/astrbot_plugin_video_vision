"""视频处理配置模型。

刻意不依赖 AstrBot，便于日后上移到核心。
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, replace
from typing import Any

#: 实测经中转站上传约 7-9 秒/MB（原始体积）。13MB 对应约 60-90 秒，
#: 需留在 provider timeout（默认 120s，本机已调至 180s）之内。
DEFAULT_MAX_SIZE_MB = 13.0

VALID_OVERSIZE_ACTIONS = ("compress", "reject", "truncate")
VALID_PRESETS = ("ultrafast", "veryfast", "faster", "fast", "medium", "slow")


def _as_bool(value: Any, default: bool) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "on", "是", "开"}
    return default


def _as_float(value: Any, default: float, *, minimum: float | None = None) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return default
    if result != result:  # NaN
        return default
    if minimum is not None and result < minimum:
        return minimum
    return result


def _as_int(value: Any, default: int, *, minimum: int | None = None) -> int:
    try:
        result = int(float(value))
    except (TypeError, ValueError):
        return default
    if minimum is not None and result < minimum:
        return minimum
    return result


def _as_choice(value: Any, default: str, choices: tuple[str, ...]) -> str:
    text = str(value or "").strip().lower()
    return text if text in choices else default


@dataclass(frozen=True)
class VideoConfig:
    """插件运行配置。所有字段都可由用户在 WebUI 调整。"""

    enable: bool = True
    require_gemini_provider: bool = True
    max_size_mb: float = DEFAULT_MAX_SIZE_MB
    compress_max_mb: float = 8.0
    compress_crf: int = 30
    compress_preset: str = "veryfast"
    max_duration_sec: float = 60.0
    oversize_action: str = "compress"
    max_video_count: int = 2
    handle_quoted_video: bool = True
    treat_file_as_video: bool = True
    hide_local_path: bool = True
    notify_on_failure: bool = True
    max_compress_seconds: float = 45.0
    ffmpeg_path: str = ""
    cache_enabled: bool = True
    cache_max_mb: int = 512
    debug_log: bool = False

    @classmethod
    def from_mapping(cls, raw: Mapping[str, Any] | None) -> VideoConfig:
        """从 WebUI 配置字典构造，容错处理缺失/非法值。"""
        data: Mapping[str, Any] = raw if isinstance(raw, Mapping) else {}
        return cls(
            enable=_as_bool(data.get("enable"), True),
            require_gemini_provider=_as_bool(
                data.get("require_gemini_provider"), True
            ),
            max_size_mb=_as_float(
                data.get("max_size_mb"), DEFAULT_MAX_SIZE_MB, minimum=0.5
            ),
            compress_max_mb=_as_float(
                data.get("compress_max_mb"), 8.0, minimum=0.5
            ),
            compress_crf=_as_int(data.get("compress_crf"), 30, minimum=0),
            compress_preset=_as_choice(
                data.get("compress_preset"), "veryfast", VALID_PRESETS
            ),
            max_duration_sec=_as_float(
                data.get("max_duration_sec"), 60.0, minimum=0.0
            ),
            oversize_action=_as_choice(
                data.get("oversize_action"), "compress", VALID_OVERSIZE_ACTIONS
            ),
            max_video_count=_as_int(data.get("max_video_count"), 2, minimum=0),
            handle_quoted_video=_as_bool(data.get("handle_quoted_video"), True),
            treat_file_as_video=_as_bool(data.get("treat_file_as_video"), True),
            hide_local_path=_as_bool(data.get("hide_local_path"), True),
            notify_on_failure=_as_bool(data.get("notify_on_failure"), True),
            max_compress_seconds=_as_float(
                data.get("max_compress_seconds"), 45.0, minimum=1.0
            ),
            ffmpeg_path=str(data.get("ffmpeg_path") or "").strip(),
            cache_enabled=_as_bool(data.get("cache_enabled"), True),
            cache_max_mb=_as_int(data.get("cache_max_mb"), 512, minimum=16),
            debug_log=_as_bool(data.get("debug_log"), False),
        )

    def normalized(self) -> tuple[VideoConfig, list[str]]:
        """修正自相矛盾的配置并返回 (新配置, 警告列表)。

        典型问题：压缩体积上限 >= 体积闸门 —— 那样压缩完仍然超限。
        """
        warnings: list[str] = []
        ceiling = self.compress_max_mb
        if ceiling >= self.max_size_mb:
            ceiling = max(0.5, self.max_size_mb * 0.6)
            warnings.append(
                f"compress_max_mb({self.compress_max_mb}) 不小于 "
                f"max_size_mb({self.max_size_mb})，已自动下调为 {ceiling:.1f}"
            )

        crf = self.compress_crf
        if crf > 51:
            crf = 51
            warnings.append("compress_crf 超过 H.264 上限 51，已收敛为 51")

        if ceiling == self.compress_max_mb and crf == self.compress_crf:
            return self, warnings

        return replace(self, compress_max_mb=ceiling, compress_crf=crf), warnings
