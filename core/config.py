"""视频处理配置模型。

刻意不依赖 AstrBot，便于日后上移到核心。
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field, fields, replace
from typing import Any

#: 实测经中转站上传约 7-9 秒/MB（原始体积）。13MB 对应约 60-90 秒，
#: 需留在 provider timeout（默认 120s，本机已调至 180s）之内。
DEFAULT_MAX_SIZE_MB = 13.0

#: ``compress_max_mb`` 的下界（与 ``from_mapping`` 的 minimum 保持一致）
MIN_COMPRESS_MB = 0.5

#: 数值配置的硬边界。0.2.0 只设下界，``inf`` / ``1e999`` / 手滑多打几个零都
#: 会被原样接受（``max_size_mb=inf`` 实测让 73.54MB 文件直接走 as-is 编码，
#: 峰值内存 269.6MB）。收益极小、风险极大的项一律收敛。
MAX_SIZE_MB_CEILING = 2000.0
CACHE_MAX_MB_CEILING = 65536
MAX_VIDEO_COUNT_CEILING = 10
MAX_COMPRESS_SECONDS_CEILING = 600.0
OPENAI_TIMEOUT_SEC_CEILING = 3600.0
OPENAI_MAX_TOKENS_CEILING = 100000

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


def _as_float(
    value: Any,
    default: float,
    *,
    minimum: float | None = None,
    maximum: float | None = None,
) -> float:
    """把配置值转成 float，非法值回落默认值，越界值收敛到边界。

    ``inf`` / ``1e999`` 是**合法的 JSON 数值**，会被 float() 接受；此处由
    上下界收敛（0.2.0 没有上界，``max_size_mb=inf`` 会一路走到 data URI 编码，
    实测 73.54MB 的文件峰值内存 269.6MB）。
    """
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError):
        return default
    if result != result:  # NaN
        return default
    if minimum is not None and result < minimum:
        result = minimum
    if maximum is not None and result > maximum:
        result = maximum
    return result


def _as_int(
    value: Any,
    default: int,
    *,
    minimum: int | None = None,
    maximum: int | None = None,
) -> int:
    """把配置值转成 int，非法值回落默认值，越界值收敛到边界。

    必须捕 ``OverflowError``：``int(float("inf"))`` 会抛它，而 ``1e999`` 是合法
    JSON —— 0.2.0 里一个 ``{"max_video_count": 1e999}`` 就足以让
    ``from_mapping`` 穿出异常、插件**加载失败**，报错还对用户毫无意义。
    """
    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError):
        return default
    if number != number:  # NaN
        return default
    try:
        result = int(number)
    except (OverflowError, ValueError):
        # ±inf（或超出 int 范围）：按方向收敛到已知边界
        if number > 0 and maximum is not None:
            result = maximum
        elif number < 0 and minimum is not None:
            result = minimum
        else:
            return default
    if minimum is not None and result < minimum:
        result = minimum
    if maximum is not None and result > maximum:
        result = maximum
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


#: 数值字段的 (下限, 上限)，用于把「被收敛的值」变成一条可读告警。
#: 与 ``from_mapping_detailed`` 里的实际边界一一对应。
_NUMERIC_BOUNDS: dict[str, tuple[float, float]] = {
    "max_size_mb": (0.5, MAX_SIZE_MB_CEILING),
    "compress_max_mb": (MIN_COMPRESS_MB, MAX_SIZE_MB_CEILING),
    "cache_max_mb": (16, CACHE_MAX_MB_CEILING),
    "max_video_count": (0, MAX_VIDEO_COUNT_CEILING),
    "max_compress_seconds": (1.0, MAX_COMPRESS_SECONDS_CEILING),
    "openai_timeout_sec": (10.0, OPENAI_TIMEOUT_SEC_CEILING),
    "openai_max_tokens": (64, OPENAI_MAX_TOKENS_CEILING),
}


def _range_notices(data: Mapping[str, Any]) -> list[str]:
    """把「超出上下界、已被收敛」的字段列成告警（不报错）。"""
    notices: list[str] = []
    for key, (low, high) in _NUMERIC_BOUNDS.items():
        if key not in data:
            continue
        try:
            number = float(data[key])
        except (TypeError, ValueError, OverflowError):
            continue
        if number != number:  # NaN 会回落默认值，不属于"收敛"
            continue
        if number < low:
            notices.append(f"{key}={number:g} 低于下限 {low:g}，已收敛为 {low:g}")
        elif number > high:
            notices.append(f"{key}={number:g} 高于上限 {high:g}，已收敛为 {high:g}")
    return notices


def _unknown_key_notices(data: Mapping[str, Any]) -> list[str]:
    """未知 / 拼错的配置键（0.2.0 完全静默）。"""
    known = {item.name for item in fields(VideoConfig)}
    unknown = sorted(str(key) for key in data if key not in known)
    if not unknown:
        return []
    return ["忽略未知配置键（请检查拼写）: " + ", ".join(unknown)]


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
        config, _notices = cls.from_mapping_detailed(raw)
        return config

    @classmethod
    def from_mapping_detailed(
        cls, raw: Mapping[str, Any] | None
    ) -> tuple[VideoConfig, list[str]]:
        """同 :meth:`from_mapping`，另外返回可读的配置告警。

        告警覆盖两类默认完全静默的问题：

        1. **未知 / 拼错的配置键**（0.2.0 里拼错 ``max_size_mb`` 会无声地
           回到默认值，用户只会奇怪"为什么配置没生效"）；
        2. **被上下界收敛的数值**（``inf`` / ``1e999`` / 多打几个零）。

        Returns:
            (配置, 告警列表)。告警只用于日志，**不阻塞**插件加载。
        """
        data: Mapping[str, Any] = raw if isinstance(raw, Mapping) else {}
        notices = _unknown_key_notices(data) + _range_notices(data)
        # compress_max_mb 的上界跟着体积闸门走：压缩产物必须小于闸门，
        # 否则压完仍然超限（normalized() 会再兜一次并告警）。
        max_size = _as_float(
            data.get("max_size_mb"),
            DEFAULT_MAX_SIZE_MB,
            minimum=0.5,
            maximum=MAX_SIZE_MB_CEILING,
        )
        return cls(
            enable=_as_bool(data.get("enable"), True),
            require_gemini_provider=_as_bool(
                data.get("require_gemini_provider"), True
            ),
            additional_provider_types=_as_str_list(
                data.get("additional_provider_types")
            ),
            max_size_mb=max_size,
            compress_max_mb=_as_float(
                data.get("compress_max_mb"),
                8.0,
                minimum=MIN_COMPRESS_MB,
                maximum=max_size,
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
            max_video_count=_as_int(
                data.get("max_video_count"),
                2,
                minimum=0,
                maximum=MAX_VIDEO_COUNT_CEILING,
            ),
            handle_quoted_video=_as_bool(data.get("handle_quoted_video"), True),
            treat_file_as_video=_as_bool(data.get("treat_file_as_video"), True),
            hide_local_path=_as_bool(data.get("hide_local_path"), True),
            notify_on_failure=_as_bool(data.get("notify_on_failure"), True),
            max_compress_seconds=_as_float(
                data.get("max_compress_seconds"),
                45.0,
                minimum=1.0,
                maximum=MAX_COMPRESS_SECONDS_CEILING,
            ),
            ffmpeg_path=str(data.get("ffmpeg_path") or "").strip(),
            cache_enabled=_as_bool(data.get("cache_enabled"), True),
            cache_max_mb=_as_int(
                data.get("cache_max_mb"),
                512,
                minimum=16,
                maximum=CACHE_MAX_MB_CEILING,
            ),
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
                data.get("openai_timeout_sec"),
                180.0,
                minimum=10.0,
                maximum=OPENAI_TIMEOUT_SEC_CEILING,
            ),
            openai_max_tokens=_as_int(
                data.get("openai_max_tokens"),
                2048,
                minimum=64,
                maximum=OPENAI_MAX_TOKENS_CEILING,
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
        ), notices

    def normalized(self) -> tuple[VideoConfig, list[str]]:
        """修正自相矛盾的配置并返回 (新配置, 警告列表)。

        能自动修正的：

        - ``compress_max_mb >= max_size_mb`` —— 那样压缩完仍然超限；
        - ``compress_crf > 51`` —— 超出 H.264 上限。

        不能自动修正的**显式告警**，交给用户决定：

        - ``max_size_mb`` 太小（0.83MB 以下时 ``compress_max_mb`` 无论如何都
          不小于它，旧代码 ``max(0.5, max_size_mb * 0.6)`` 在 0.5MB 处会原地
          打转却只报"已自动下调为 0.5"，等于错报成功）；
        - ``oversize_action="truncate"`` 但 ``max_duration_sec<=0`` —— 截断永远
          不生效，超限视频只会被压缩；
        - ``compress_max_mb >= cache_max_mb`` —— 压缩产物放不进缓存。
        """
        warnings: list[str] = []
        ceiling = self.compress_max_mb
        if ceiling >= self.max_size_mb:
            candidate = self.max_size_mb * 0.6
            if candidate < MIN_COMPRESS_MB:
                warnings.append(
                    f"max_size_mb({self.max_size_mb:g}) 过小：compress_max_mb "
                    f"必须小于它，而下限是 {MIN_COMPRESS_MB:g}，无法自动修正，"
                    "请把体积闸门调到 1.0 以上"
                )
            else:
                ceiling = candidate
                warnings.append(
                    f"compress_max_mb({self.compress_max_mb}) 不小于 "
                    f"max_size_mb({self.max_size_mb})，已自动下调为 {ceiling:.1f}"
                )

        if self.oversize_action == "truncate" and self.max_duration_sec <= 0:
            warnings.append(
                "oversize_action=truncate 但 max_duration_sec<=0：截断不可能生效，"
                "超限视频只会走压缩。请把 max_duration_sec 设为正数，"
                "或把 oversize_action 改成 compress"
            )

        if self.cache_enabled and self.compress_max_mb >= self.cache_max_mb:
            warnings.append(
                f"compress_max_mb({self.compress_max_mb:g}) 不小于 "
                f"cache_max_mb({self.cache_max_mb:g})：超过缓存预算的压缩产物"
                "不会被缓存（每次都会重新压缩）"
            )

        crf = self.compress_crf
        if crf > 51:
            crf = 51
            warnings.append("compress_crf 超过 H.264 上限 51，已收敛为 51")

        if ceiling == self.compress_max_mb and crf == self.compress_crf:
            return self, warnings

        return replace(self, compress_max_mb=ceiling, compress_crf=crf), warnings
