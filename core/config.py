"""视频处理配置模型。

刻意不依赖 AstrBot，便于日后上移到核心。
"""

from __future__ import annotations

import math
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

#: 0.3.0：ffmpeg 并发限流。默认 2 的依据是**本机 2 核实测**（钉 2 个逻辑核 +
#: 6 线程默认执行器 + 8 个内容各不相同的 720p/18s 源，每组 3 次取中位）：
#: 单个 ffmpeg 只用掉约 44%-77% 的 CPU，2 个槽位大体喂满；限量到 2 之后
#: makespan 与"不限"持平（15.09s vs 14.91s，+1.3%），单个 ffmpeg 的槽内时长
#: 11.70s → 3.77s，峰值 ffmpeg 进程 6 → 2；而 limit=1 会让总耗时 +22%
#: （单个 ffmpeg 用不满 2 个核，串行等于让一半机器空转）。
#: **不写成 cpu_count 公式**：配置默认值必须可预测、可文档化、可被
#: ``test_schema_defaults_match_dataclass_defaults`` 逐字校验。
FFMPEG_CONCURRENCY_CEILING = 8

#: 单个视频**等待槽位**的上限（不含转码本身）。取 30s 的理由：
#: openai_compat 下一次成功请求的预算是"本地（排队 + 压缩）+ 上传（7-9s/MB ×
#: 压缩目标 8MB ≈ 56-72s）+ 推理"，再叠加 ``max_compress_seconds=45``；排队上限
#: 若也给 60s，最坏情况会顶穿 ``openai_timeout_sec=180``。
QUEUE_TIMEOUT_CEILING = 600.0

#: 0.3.0：视觉理解结果缓存。TTL 上界取 24h —— 再长就等于"重启前永远命中"，
#: 而端点侧模型/prompt 的真实变化用户是感知不到的（缓存只在**完全相同**的
#: 重复投递上命中，长 TTL 不会扩大命中面，只会推迟"重新验证"的时刻）。
VISION_CACHE_TTL_CEILING = 86400.0

#: 条目数上界：按 ``openai_vision.MAX_ANALYSIS_CHARS``（16KB）算，1024 条
#: 最坏约 16MB 常驻 —— 已经超出"顺手加个缓存"的合理成本，所以上界就到这里。
VISION_CACHE_ENTRIES_CEILING = 1024

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


def _as_float_or_none(value: Any) -> float | None:
    """同 :func:`_as_float` 的容错，但**不**做上下界收敛；取不到数字返回 None。

    用途只有一个：分辨"用户真的写了一个数（比如 0.1）"和"用户写了垃圾（比如
    ``"abc"``）"。前者要报「低于下限、已收敛」，后者只是回落默认值 —— 与其它
    数值键一致，不发提示。
    """
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    return None if result != result else result  # NaN 视为"没有值"


#: ``compress_max_mb`` 的默认值（``_conf_schema.json`` 同值）
DEFAULT_COMPRESS_MB = 8.0

#: ``compress_max_mb`` 相对体积闸门保留的余量：压缩目标 = ``min(用户值, 闸门 × 0.9)``。
#:
#: 为什么是 0.9（而不是"砍到一半"的 0.6）：
#:
#: 1. 目标是"贴着闸门下方"：0.6 会让 13MB 闸门只剩 7.8MB，而码率预算本来就是
#:    从这个目标反推的（``media.budget_kbits`` 按 max_mb 算），等于白白牺牲一半画质；
#: 2. 压缩器自己已按 max_mb 反推码率并给容器与音频留了 8% 余量
#:    （``media.py: budget_kbits = max_mb * 8192 * 0.92``），这 8% 与这里的 10%
#:    是**叠加**的，常规产物落在闸门的 ~83%；
#: 3. 真正的硬上限是 ``-fs``"写满即停"，实测溢出 1.03x（0.5MB 预算 -> 532KB）；
#:    越小的预算里固定开销占比越高，最坏 1.28x 出现在 0.2MB 这种极小预算上
#:    （见 FIX-REPORT §3.3-1）。10% 余量在 0.5MB 以上这一档吸收得掉；
#: 4. 万一越过闸门，injector 压缩后会复核体积并记下"仍超闸门，已尽力"，
#:    不会静默宣称成功。
#:
#: 系数 < 1 保证 ``min()`` 的结果严格小于闸门，因此 :func:`_resolve_compress_mb`
#: **幂等**（再喂回去得到同一个值），``normalized()`` 对它的产出零改动零告警。
COMPRESS_GATE_RATIO = 0.9


def _mb_text(value: float) -> str:
    """体积数字的可读形式：至少带一位小数（15 → 15.0），但不丢有效小数（0.75）。

    告警里要出现**用户写的那个数**，``str(15.0)`` / ``f"{15.0:g}"`` 分别是
    "15.0" 与 "15" —— 两种都跟 ``11.7`` 这种一位小数的最终值不好对齐，
    所以统一成"最短但至少一位小数"。

    **非有限值单独处理**：JSON 里的 ``1e999`` 会变成 ``inf``，手写 ``Infinity``
    会变成 ``nan``/``inf``。旧实现把 ``inf`` 当成"没有小数点"补成 ``inf.0``，
    于是提示写成 ``compress_max_mb=inf.0 超过体积闸门 …``（VERIFY2 P3-4）。
    现在分别输出 ``∞`` / ``-∞`` / ``非数值``。
    """
    if not math.isfinite(value):
        if math.isnan(value):
            return "非数值"
        return "∞" if value > 0 else "-∞"
    text = f"{value:g}"
    if "." in text or "e" in text or "E" in text:
        return text
    return f"{text}.0"


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
#:
#: ``compress_max_mb`` **刻意不在这张表里**：它的上界不是常量而是体积闸门，
#: 且它的提示要同时写出"用户原值 → 最终值"。两者都由
#: :func:`_resolve_compress_mb` 一次算完（单趟归一），不再走这里的通用循环。
_NUMERIC_BOUNDS: dict[str, tuple[float, float]] = {
    "max_size_mb": (0.5, MAX_SIZE_MB_CEILING),
    "cache_max_mb": (16, CACHE_MAX_MB_CEILING),
    "max_video_count": (0, MAX_VIDEO_COUNT_CEILING),
    "max_compress_seconds": (1.0, MAX_COMPRESS_SECONDS_CEILING),
    "openai_timeout_sec": (10.0, OPENAI_TIMEOUT_SEC_CEILING),
    "openai_max_tokens": (64, OPENAI_MAX_TOKENS_CEILING),
    # ---- 0.3.0 ----
    #: 0 = 关闭限流（逐字节回到 0.2.1）
    "ffmpeg_max_concurrency": (0, FFMPEG_CONCURRENCY_CEILING),
    "ffmpeg_queue_timeout_sec": (1.0, QUEUE_TIMEOUT_CEILING),
    "vision_cache_ttl_sec": (60.0, VISION_CACHE_TTL_CEILING),
    "vision_cache_max_entries": (1, VISION_CACHE_ENTRIES_CEILING),
}


def _range_notices(data: Mapping[str, Any]) -> list[str]:
    """把「超出上下界、已被收敛」的字段列成告警（不报错）。

    每个字段最多一条，格式统一为「``key=原值 高于/低于 边界，已收敛为 X``」。
    ``compress_max_mb`` 的闸门上界与"原值 → 最终值"由
    :func:`_resolve_compress_mb` 单独负责（0.2.1 曾在这里为它开特例，
    结果同一个字段打出两个"最终值"）。

    Args:
        data: 用户的原始配置字典。
    """
    notices: list[str] = []
    for key, (low, high) in _NUMERIC_BOUNDS.items():
        if key not in data:
            continue
        number = _as_float_or_none(data[key])
        if number is None:  # 垃圾值会回落默认值，不属于"收敛"
            continue
        if number < low:
            notices.append(f"{key}={number:g} 低于下限 {low:g}，已收敛为 {low:g}")
        elif number > high:
            notices.append(f"{key}={number:g} 高于上限 {high:g}，已收敛为 {high:g}")
    return notices


def _resolve_compress_mb(raw: Any, gate_mb: float) -> tuple[float, str]:
    """把 ``compress_max_mb`` **一次**算成最终生效值，并给出至多一条提示。

    最终值 = ``min(用户值, 闸门 × :data:`COMPRESS_GATE_RATIO`)``，且不低于
    :data:`MIN_COMPRESS_MB`。

    为什么必须一次算完：0.2.1 分两步改同一个字段 —— ``from_mapping_detailed``
    先按体积闸门把 15.0 夹成 13.0（第一条提示），``normalized()`` 再降到 11.7
    （第二条提示）。于是同一个旋钮在 ``/videotest`` 里出现三个数字
    （15.0 / 13.0 / 11.7），两条提示都在说"已修正"，生效值却等于哪一条都没写。
    现在只有一次改写、一条提示，原值与最终值出现在**同一句**里。

    幂等性由构造保证：``min(x, gate × 0.9)`` 再喂回来必然得到同一个值，所以
    :meth:`VideoConfig.normalized` 对 ``from_mapping`` 的产出是"零改动零告警"。

    闸门太小（``gate × 0.9 < MIN_COMPRESS_MB``）时**无解**：返回
    ``min(用户值, 闸门)`` 且**不发提示** —— 那种情况交给 ``normalized()`` 发一条
    "无法自动修正"的显式告警，绝不能假装修好了。

    Args:
        raw: 用户写的原始值（任何 JSON 类型）。
        gate_mb: **已经收敛过**的体积闸门。

    Returns:
        ``(生效值, 提示或空串)``。
    """
    asked = _as_float(raw, DEFAULT_COMPRESS_MB, minimum=MIN_COMPRESS_MB)
    candidate = gate_mb * COMPRESS_GATE_RATIO
    if candidate < MIN_COMPRESS_MB:
        return min(asked, gate_mb), ""
    final = min(asked, candidate)
    number = _as_float_or_none(raw)
    if number is not None and number < MIN_COMPRESS_MB:
        return final, (
            f"compress_max_mb={number:g} 低于下限 {MIN_COMPRESS_MB:g}，"
            f"已收敛为 {_mb_text(final)}"
        )
    if final >= asked:  # 用户值本身就贴着闸门下方，没被改
        return final, ""
    return final, (
        f"compress_max_mb={_mb_text(asked)} 超过体积闸门 "
        f"max_size_mb={_mb_text(gate_mb)}，已按闸门的 "
        f"{COMPRESS_GATE_RATIO:.0%} 收敛为 {_mb_text(final)}"
    )


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
    compress_max_mb: float = DEFAULT_COMPRESS_MB
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

    # -------- 0.3.0：ffmpeg 并发限流（进程内全局、跨会话）--------
    #: 同时进行的 ffmpeg 转码数。**0 = 关闭限流**（逐字节回到 0.2.1：
    #: 上限只剩默认线程池的 ``min(32, cpu+4)``）。
    #: 排队发生在**事件循环侧**（``core/limits.SlotLimiter``），不占用默认
    #: 线程池 —— 那个池是全进程共享的，2 核只有 6 个线程。
    ffmpeg_max_concurrency: int = 2
    #: 单个视频等待槽位的上限（不含转码本身）。超时 -> ``kind="busy"``。
    ffmpeg_queue_timeout_sec: float = 30.0
    #: 同一内容哈希的并发转码只做一次。依赖 ``cache_enabled``：
    #: 缓存关掉就没有可共享的产物，followers 只能自己压（会打一条 DEBUG）。
    ffmpeg_singleflight: bool = True

    # -------- 0.3.0：视觉理解结果缓存（进程内、纯内存、TTL+LRU）--------
    #: **只对"完全相同"的重复投递生效**：同一份视频字节 + 同一个最终 prompt +
    #: 同一端点 + 同一模型 + 同一 max_tokens。追问时提问变了 -> 本来就不该命中
    #: （AUDIT-030 §B.2 证伪了"连续追问每次重调"这个动机的前半段）。
    #: 关闭时**完全不建缓存对象**，行为逐字节回到 0.2.1。
    vision_cache_enabled: bool = True
    #: 条目存活时长（60-86400）。命中**不**续期：否则热门条目永不过期。
    vision_cache_ttl_sec: float = 600.0
    #: LRU 容量（1-1024）。按 ``MAX_ANALYSIS_CHARS``=16KB 算，64 条约 1MB。
    vision_cache_max_entries: int = 64

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
        2. **被上下界收敛的数值**（``inf`` / ``1e999`` / 多打几个零），以及
           被体积闸门收敛的 ``compress_max_mb``。

        ``compress_max_mb`` 在这里就**一次算成最终值**（见
        :func:`_resolve_compress_mb`）：出一个数、发一条提示，提示里同时有原值
        与最终值。因此 :meth:`normalized` 不再二次改动它。

        Returns:
            (配置, 告警列表)。告警只用于日志，**不阻塞**插件加载。
        """
        data: Mapping[str, Any] = raw if isinstance(raw, Mapping) else {}
        max_size = _as_float(
            data.get("max_size_mb"),
            DEFAULT_MAX_SIZE_MB,
            minimum=0.5,
            maximum=MAX_SIZE_MB_CEILING,
        )
        # 压缩目标的上界跟着体积闸门走（压完必须小于闸门），且要一次算完 ——
        # 闸门必须先算出来。
        compress_max_mb, compress_notice = _resolve_compress_mb(
            data.get("compress_max_mb"), max_size
        )
        notices = _unknown_key_notices(data) + _range_notices(data)
        if compress_notice:
            notices.append(compress_notice)
        config = cls(
            enable=_as_bool(data.get("enable"), True),
            require_gemini_provider=_as_bool(
                data.get("require_gemini_provider"), True
            ),
            additional_provider_types=_as_str_list(
                data.get("additional_provider_types")
            ),
            max_size_mb=max_size,
            compress_max_mb=compress_max_mb,
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
            ffmpeg_max_concurrency=_as_int(
                data.get("ffmpeg_max_concurrency"),
                2,
                minimum=0,
                maximum=FFMPEG_CONCURRENCY_CEILING,
            ),
            ffmpeg_queue_timeout_sec=_as_float(
                data.get("ffmpeg_queue_timeout_sec"),
                30.0,
                minimum=1.0,
                maximum=QUEUE_TIMEOUT_CEILING,
            ),
            ffmpeg_singleflight=_as_bool(data.get("ffmpeg_singleflight"), True),
            vision_cache_enabled=_as_bool(data.get("vision_cache_enabled"), True),
            vision_cache_ttl_sec=_as_float(
                data.get("vision_cache_ttl_sec"),
                600.0,
                minimum=60.0,
                maximum=VISION_CACHE_TTL_CEILING,
            ),
            vision_cache_max_entries=_as_int(
                data.get("vision_cache_max_entries"),
                64,
                minimum=1,
                maximum=VISION_CACHE_ENTRIES_CEILING,
            ),
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
        )
        return config, notices

    def normalized(self) -> tuple[VideoConfig, list[str]]:
        """修正自相矛盾的配置并返回 (新配置, 警告列表)。

        能自动修正的：

        - ``compress_max_mb >= max_size_mb`` —— 那样压缩完仍然超限，下调到
          **闸门的 90%**（理由见 :data:`COMPRESS_GATE_RATIO`）。这条只是
          **不变量守卫**：``from_mapping_detailed`` 已经用同一个公式
          （:func:`_resolve_compress_mb`）算好了最终值，那里产出的配置再跑本函数
          一定"零改动零告警"（幂等性由 ``min()`` 的构造保证）。守卫仍然保留，是为了
          兜住直接 ``VideoConfig(max_size_mb=13, compress_max_mb=13)`` 这种构造 ——
          那条路径没有经过 ``from_mapping``；
        - ``compress_crf > 51`` —— 超出 H.264 上限。

        不能自动修正的**显式告警**，交给用户决定：

        - ``max_size_mb`` 太小（0.56MB 以下时 ``compress_max_mb`` 无论如何都
          不小于它，旧代码 ``max(0.5, max_size_mb * 0.6)`` 在 0.5MB 处会原地
          打转却只报"已自动下调为 0.5"，等于错报成功）；
        - ``oversize_action="truncate"`` 但 ``max_duration_sec<=0`` —— 截断永远
          不生效，超限视频只会被压缩；
        - ``compress_max_mb >= cache_max_mb`` —— 压缩产物放不进缓存。
        """
        warnings: list[str] = []
        ceiling = self.compress_max_mb
        if ceiling >= self.max_size_mb:
            candidate = self.max_size_mb * COMPRESS_GATE_RATIO
            if candidate < MIN_COMPRESS_MB:
                warnings.append(
                    f"max_size_mb({self.max_size_mb:g}) 过小：compress_max_mb "
                    f"必须小于它，而下限是 {MIN_COMPRESS_MB:g}，插件不会自动修正，"
                    "请把体积闸门调到 1.0 以上"
                )
            else:
                ceiling = candidate
                warnings.append(
                    f"compress_max_mb({_mb_text(self.compress_max_mb)}) 必须小于 "
                    f"max_size_mb({_mb_text(self.max_size_mb)})，"
                    f"已自动下调为 {ceiling:.1f}"
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

        # 0.3.0：本地排队 + 压缩发生在视觉调用**之前**，而 openai_timeout_sec
        # 只管后者。两者相加吃掉一半以上预算时，最坏情况会顶穿视觉服务超时
        # （AUDIT-030 §E-2），必须显式告警而不是让用户去猜。
        if (
            self.effective_provider_mode == "openai_compat"
            and self.ffmpeg_queue_timeout_sec + self.max_compress_seconds
            >= self.openai_timeout_sec * 0.5
        ):
            warnings.append(
                f"ffmpeg_queue_timeout_sec({self.ffmpeg_queue_timeout_sec:g}s)"
                f"+max_compress_seconds({self.max_compress_seconds:g}s) 已吃掉"
                f"视觉服务超时预算(openai_timeout_sec={self.openai_timeout_sec:g}s)"
                " 的一半以上，建议调小其中一项（排队上限或压缩上限）"
            )

        crf = self.compress_crf
        if crf > 51:
            crf = 51
            warnings.append("compress_crf 超过 H.264 上限 51，已收敛为 51")

        if ceiling == self.compress_max_mb and crf == self.compress_crf:
            return self, warnings

        return replace(self, compress_max_mb=ceiling, compress_crf=crf), warnings
