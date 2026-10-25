"""视频处理配置模型。

刻意不依赖 AstrBot，便于日后上移到核心。
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field, replace
from typing import Any

#: 实测经中转站上传约 7-9 秒/MB（原始体积）。13MB 对应约 60-90 秒，
#: 需留在 provider timeout（默认 120s，本机已调至 180s）之内。
DEFAULT_MAX_SIZE_MB = 13.0

VALID_OVERSIZE_ACTIONS = ("compress", "reject", "truncate")
VALID_PRESETS = ("ultrafast", "veryfast", "faster", "fast", "medium", "slow")

#: 视频走哪条路送给模型：
#: - ``gemini_native``：注入 AstrBot 当前的 Gemini 系 provider（借道 audio_url，
#:   由 provider 侧 process_audio_url 按 MIME 构造媒体 Part）；
#: - ``openai_compat``：由插件直接调用配置里的 OpenAI 兼容接口做视觉理解，
#:   再把理解结果以文字注入对话。主对话 provider 可以是任意模型。
VALID_PROVIDER_MODES = ("openai_compat", "gemini_native")

#: 用户没附带文字问题时，视觉服务用的通用描述提示词。
DEFAULT_DESCRIBE_PROMPT = (
    "请详细描述这段视频：场景、人物/物体、动作、画面中的文字、字幕与关键时间点。"
    "你的描述会被转交给另一个看不到视频的模型，请客观具体、保留关键细节。"
)

#: 用户附带了文字问题时用的提示词模板，``{question}`` 会被替换为用户原话。
DEFAULT_QA_PROMPT = (
    "用户发来一段视频，并问：「{question}」\n"
    "请认真观看视频后直接、具体地回答用户的问题。"
    "如果问题与视频内容无关，请简要说一句，再描述视频的主要内容。"
)


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


def _as_str_list(value: Any) -> list[str]:
    """把 WebUI 的 list 配置规范化：去空白、去空项、保序去重。"""
    if isinstance(value, str):
        value = [value]
    if not isinstance(value, (list, tuple)):
        return []
    result: list[str] = []
    for item in value:
        text = str(item or "").strip()
        if text and text not in result:
            result.append(text)
    return result


@dataclass(frozen=True)
class VideoConfig:
    """插件运行配置。所有字段都可由用户在 WebUI 调整。"""

    enable: bool = True
    require_gemini_provider: bool = True
    #: 在 GEMINI_PROVIDER_TYPES 之外额外允许注入视频的 provider type。
    #: 默认空列表 —— 行为与旧版完全一致。
    additional_provider_types: list[str] = field(default_factory=list)
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

    # -------- 视觉服务商（provider_mode=openai_compat 时生效）--------
    provider_mode: str = "openai_compat"
    openai_base_url: str = ""
    openai_api_key: str = ""
    openai_model: str = "gemini-3.8-flash-high"
    openai_timeout_sec: float = 180.0
    openai_max_tokens: int = 2048
    vision_with_question: bool = True
    vision_describe_prompt: str = DEFAULT_DESCRIBE_PROMPT
    vision_qa_prompt: str = DEFAULT_QA_PROMPT
    vision_verify_ssl: bool = True

    @property
    def openai_service_ready(self) -> bool:
        """OpenAI 兼容视觉服务三要素是否齐全。"""
        return bool(
            self.openai_base_url and self.openai_api_key and self.openai_model
        )

    @property
    def effective_provider_mode(self) -> str:
        """实际生效的模式。

        ``provider_mode=openai_compat`` 但三要素没配齐时降级为 ``gemini_native``。
        原因见 ``hooks/llm_request._effective_mode``：老配置文件里没有
        ``provider_mode`` 键，落到默认值后如果直接跳过注入，升级即等于视频功能
        静默失效。钩子、启动日志、``/videotest`` 都读这个属性，保证三处一致。
        """
        if self.provider_mode == "openai_compat" and not self.openai_service_ready:
            return "gemini_native"
        return self.provider_mode

    @classmethod
    def from_mapping(cls, raw: Mapping[str, Any] | None) -> VideoConfig:
        """从 WebUI 配置字典构造，容错处理缺失/非法值。"""
        data: Mapping[str, Any] = raw if isinstance(raw, Mapping) else {}
        return cls(
            enable=_as_bool(data.get("enable"), True),
            require_gemini_provider=_as_bool(
                data.get("require_gemini_provider"), True
            ),
            additional_provider_types=_as_str_list(
                data.get("additional_provider_types")
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
            provider_mode=_as_choice(
                data.get("provider_mode"), "openai_compat", VALID_PROVIDER_MODES
            ),
            openai_base_url=str(data.get("openai_base_url") or "")
            .strip()
            .rstrip("/"),
            openai_api_key=str(data.get("openai_api_key") or "").strip(),
            openai_model=str(
                data.get("openai_model") or "gemini-3.8-flash-high"
            ).strip(),
            openai_timeout_sec=_as_float(
                data.get("openai_timeout_sec"), 180.0, minimum=10.0
            ),
            openai_max_tokens=_as_int(
                data.get("openai_max_tokens"), 2048, minimum=64
            ),
            vision_with_question=_as_bool(
                data.get("vision_with_question"), True
            ),
            vision_describe_prompt=str(
                data.get("vision_describe_prompt") or DEFAULT_DESCRIBE_PROMPT
            ),
            vision_qa_prompt=str(
                data.get("vision_qa_prompt") or DEFAULT_QA_PROMPT
            ),
            vision_verify_ssl=_as_bool(data.get("vision_verify_ssl"), True),
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
