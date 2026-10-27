"""「一个视频会发生什么」的**唯一真相源**。

0.2.1 把同一份判定写了三遍：``injector`` 执行时算一遍，``main._diagnose`` 为了打印
「结论」行再算一遍（注释写着"必须与 injector 完全一致"），``hooks`` 又按字符串猜
一遍失败原因。改一条规则要动三个文件，漏一处就产出"诊断说谎"——结论行说
「未超限，直接送」，下一行却是「❌ 分类: corrupt」（真机复现两次）。

这里只有一个**纯函数** :func:`classify`：吃事实（体积 / 时长 / 配置 / 工具能力），
吐 :class:`MediaPlan`。不读文件、不调 ffmpeg、不认识 AstrBot —— 于是 injector 执行、
main 诊断、hooks 解释原因吃的是同一份判决，结构上不可能再分叉。

刻意留在本模块之外：18MB data URI 上限的**执行期**检查仍在
``media.read_as_data_uri``（这里只预测，失败文案逐字不变）；ffmpeg 的真实失败
（超时 / 退出码 / 损坏）只有跑过才知道，归 ``MediaToolError.kind``。
"""

from __future__ import annotations

from dataclasses import dataclass

from .config import VideoConfig
from .media import MAX_DATA_URI_BYTES, will_truncate
from .placeholder import VIDEO_SUFFIXES

#: 判定为「Gemini 系」的 provider 类型 —— 只有它们才会把 audio_url 块按 MIME
#: 构造成媒体 Part（``process_audio_url`` 是 MIME 无关的）。其他 provider 会把
#: 视频当音频处理，因此默认拦住。仅 gemini_native 模式使用。
#:
#: 0.2.1 它住在 ``hooks/llm_request.py``，而 ``main._diagnose_gates`` 为了渲染
#: 闸门 ① 又抄了一遍「GEMINI | additional」表达式（ARCH-REVIEW R8）——
#: 于是"诊断说的"和"实际拦的"是两份代码。现在唯一实现在这里；``hooks`` 保留
#: 同名别名（re-export），公开名字不变。
GEMINI_PROVIDER_TYPES: frozenset[str] = frozenset(
    {
        "googlegenai_chat_completion",
        "gemini_video_chat_completion",  # 兼容早期设计/第三方实现
    }
)


def allowed_provider_types(config: VideoConfig) -> frozenset[str]:
    """``gemini_native`` 允许注入视频的 provider type 集合。

    「默认白名单 + ``additional_provider_types``」的唯一写法：钩子用它判定，
    ``/videotest`` 用它渲染闸门 ①，两边不可能再各写一遍（ARCH-REVIEW R8）。
    """
    return GEMINI_PROVIDER_TYPES | set(config.additional_provider_types)


#: 动作标签（唯一一份）：前三项是判决 / 配置词表，后四项是结果词表
#: （``VideoInjection.action``）。合成一张 7 键映射是 0.2.x 的报告格式，拆开要改
#: ``/videotest`` 的输出，属行为变化。
ACTION_LABELS: dict[str, str] = {
    "compress": "压缩", "reject": "拒绝", "truncate": "截断",
    "as-is": "直接送（未压缩）", "compressed": "已压缩",
    "truncated": "已截断", "rejected": "已拒绝",
}

#: 机器可读分类 -> 注入给模型的英文短语（**不含本机路径**）。这张表就是 0.2.1
#: ``hooks._SHORT_REASONS`` 的内容 —— 它对外可见（模型收到的提示语），重构不许改字。
#:
#: 刻意**不收录**三个键，理由各不相同：
#: - ``other``：``MediaToolError`` 的兜底桶，它的 reason 里往往还有更具体的信息
#:   （如「压缩异常: …」），交给调用方的文本回退比一句笼统话更准；
#: - ``internal``：同上，属"未预期异常"，不属判决；
#: - ``size``：0.2.0 的遗留码，全仓**没有任何发射点**（``MediaToolError.KINDS``、
#:   ``KIND_OF_CODE``、``injector``/``main`` 的 ``kind=`` 字面量里都没有它），
#:   0.2.1 曾把它原样搬过来并写"保留只为不动对外词表" —— 一个永远产生不出来的
#:   短语不算对外词表，本次已删（``explain("size")`` 现在返回空串）。
#:
#: 判决码（``no_ffmpeg`` / ``unknown_duration``）不在这里：``explain`` 先用
#: :data:`KIND_OF_CODE` 把它们折成对外 kind 再查表，因此同一个概念只有一句文案。
SHORT_REASONS: dict[str, str] = {
    "timeout": "ffmpeg timed out while transcoding the video",
    "exit_code": "ffmpeg failed to transcode the video",
    "not_found": "ffmpeg is not available",
    "corrupt": "the video file appears to be corrupt",
    "empty": "the video file is empty",
    "duration": "its duration could not be read",
    "too_large": "it exceeds the size limit",
    "missing": "the file could not be read",
    # 0.3.0：**等待槽位**超时，与 ffmpeg 自己超时是两件事。复用 ``timeout``
    # （"ffmpeg timed out while transcoding"）会把用户引向 ``max_compress_seconds``
    # —— 归因错误：真正该调的是 ``ffmpeg_max_concurrency`` /
    # ``ffmpeg_queue_timeout_sec``（AUDIT-030 §C.1-4）。
    "busy": "the server is busy processing other videos",
    # 「按配置拒绝」必须分成三条：0.2.1 把 oversize 与 overlong 都报成
    # "it exceeds the size limit"，于是"视频太长被拒"告诉模型"它太大了"，
    # 模型会去劝用户压小体积 —— 错误归因（ARCH-REVIEW §4.1）。
    "rejected_size": "it exceeds the size limit",
    "rejected_duration": "its duration exceeds the limit",
    "rejected_size_and_duration": "it exceeds the size and duration limits",
    # 没有码时（例如上游只给了一句 reason）不能猜是体积还是时长，只能说真话。
    "rejected": "it was rejected by the configured size or duration policy",
}

#: 「按配置拒绝」的三条判决码（体积 / 时长 / 两者）。``rejected*`` 前缀是契约：
#: ``injector._blocked_reason`` 用它渲染原因，``hooks._short_reason`` 用
#: :data:`REJECT_REASON_PREFIX` 在**没有码**时兜底。
REJECT_CODES: frozenset[str] = frozenset(
    {"rejected_size", "rejected_duration", "rejected_size_and_duration"}
)

#: ``injector._blocked_reason`` 渲染「按配置拒绝」时的前缀。生产方与消费方共用
#: 这一个常量，避免"文案一改、兜底分支静默失效"。
REJECT_REASON_PREFIX = "按配置拒绝"

#: 判决码 -> 对外失败分类（``PrepareOutcome.kind``）。取值集合**一个都没变**：
#: 测试、日志与 ``/videotest`` 的「分类:」行都依赖它。不会失败的判决不在表里。
#: ``rejected_*`` 三条是 0.2.1 之后新拆的（原来只有 ``rejected``），对外 kind 因此
#: 更精确：``/videotest`` 的「分类:」行会写 ``rejected_duration`` 而不是笼统的
#: ``rejected``。``MediaToolError.KINDS`` 与它们不相交（执行期 vs 判决期）。
KIND_OF_CODE: dict[str, str] = {
    "empty": "empty", "corrupt": "corrupt",
    "rejected_size": "rejected_size",
    "rejected_duration": "rejected_duration",
    "rejected_size_and_duration": "rejected_size_and_duration",
    "no_ffmpeg": "not_found", "unknown_duration": "duration",
    "too_large": "too_large",
}

#: data URI 硬上限（编码后）换算成人看的 MB；唯一来源是 ``media.MAX_DATA_URI_BYTES``
#: （0.2.1 曾在 main.py 里 ``*3/4`` / ``*4/3`` 各写一遍）。
URI_LIMIT_MB = MAX_DATA_URI_BYTES / 1024 / 1024
URI_LIMIT_RAW_MB = URI_LIMIT_MB * 3 / 4


def estimate_truncate_seconds(
    duration: float, size_mb: float, max_size_mb: float
) -> float:
    """按体积预算估算「截到多长才不超预算」（0.0 = 不截断）。

    体积近似随时长线性增长，再留 10% 余量（线性假设对 CRF 编码偏乐观），
    至少 1 秒、不超过源时长。
    """
    if duration <= 0 or size_mb <= 0 or max_size_mb <= 0:
        return 0.0
    return max(1.0, min(duration * (max_size_mb / size_mb) * 0.9, duration))


def encoding_limit_note(raw_mb: float) -> str:
    """原始体积编码成 data URI 后会不会撞硬上限；撞了就返回一句人话。"""
    encoded_mb = raw_mb * 4 / 3
    if encoded_mb <= URI_LIMIT_MB:
        return ""
    return (
        f"编码后约 {encoded_mb:.1f}MB > 编码上限 {URI_LIMIT_MB:.0f}MB"
        f"（≈ 原始 {URI_LIMIT_RAW_MB:.1f}MB），实际会以 too_large 失败"
    )


def _looks_like_video(mime_or_suffix: str | None) -> bool:
    """后缀或 MIME 是否属于常见视频容器（``.mp4`` / ``mp4`` / ``video/mp4``）。

    刻意**不用** :func:`core.media.suffix_of`：那个函数吃的是路径
    （``suffix_of("mp4")`` 与 ``suffix_of(".mp4")`` 都是空串，``splitext`` 把前导
    点当文件名），而这里的入参是「后缀或 MIME」这两种裸形态。两件事，两种实现。
    """
    text = (mime_or_suffix or "").strip().lower()
    if not text:
        return False
    if "/" in text:  # MIME
        return text.startswith("video/")
    return (text if text.startswith(".") else f".{text}") in VIDEO_SUFFIXES


@dataclass(frozen=True)
class MediaPlan:
    """「这个视频会怎样」的判决（不可变、可比较、无副作用）。

    ``action`` 与 ``reason_code`` 是两个维度：``action="compress", reason_code=
    "no_ffmpeg"`` 表示"本该压，但没有 ffmpeg，执行会以 not_found 失败"；折成
    ``action="reject"`` 会让结论行从「→ 压缩」变成「→ 拒绝」，那是行为变化。
    """

    action: str  # as-is | compress | truncate | reject（后三者同 oversize_action）
    reason_code: str  # ok | oversize | overlong | empty | corrupt | rejected_* | ...
    size_mb: float = 0.0
    duration_sec: float = 0.0
    truncate_to_sec: float = 0.0  # 0 = 不截断
    note: str = ""  # 触发条件（「体积 20.0MB > 闸门 13.0MB」），执行方拼 reason/note
    encoding_note: str = ""  # 源文件撞编码上限的预判；空串 = 无风险
    target_note: str = ""  # 压缩目标撞编码上限的预判尾句（含「；但…」）
    oversize: bool = False
    overlong: bool = False
    needs_ffmpeg: bool = False
    cache_eligible: bool = False  # 要压缩且 cache_enabled（缓存目录由调用方决定）
    applies_truncate: bool = False  # 真的会传 -t（= media.will_truncate）

    @property
    def kind(self) -> str:
        """对外失败分类（``PrepareOutcome.kind``）；不会失败时是空串。"""
        return KIND_OF_CODE.get(self.reason_code, "")

    @property
    def blocked(self) -> bool:
        """执行期会直接失败。``too_large`` 不算：那是 as-is 执行期的硬检查。"""
        blocked = {"empty", "corrupt", "no_ffmpeg", "unknown_duration"} | REJECT_CODES
        return self.reason_code in blocked

    @property
    def outcome_action(self) -> str:
        """``VideoInjection.action`` 的取值。"""
        if self.action == "as-is":
            return "as-is"
        if self.action == "reject":
            return "rejected"
        return "truncated" if self.applies_truncate else "compressed"

    @property
    def expected_duration_sec(self) -> float:
        """执行后预期的产物时长：截断时是截断目标，否则等于源时长。"""
        if self.truncate_to_sec > 0:
            return min(self.duration_sec, self.truncate_to_sec)
        return self.duration_sec


def classify(
    *,
    size_mb: float,
    duration_sec: float,
    config: VideoConfig,
    mime_or_suffix: str | None = None,
    has_ffmpeg: bool = True,
    has_ffprobe: bool = True,
) -> MediaPlan:
    """给定文件事实与配置，算出唯一判决。**纯函数**：同输入永远同输出。

    判定顺序即契约：空 -> 损坏 -> 闸门 -> 策略（``reject`` 优先于"没有 ffmpeg"，
    否则用户会收到一条他不需要的错误）-> 工具能力 -> 时长 -> 截断估算。
    """
    size_mb = float(size_mb or 0.0)
    duration_sec = float(duration_sec or 0.0)
    oversize = size_mb > config.max_size_mb
    overlong = (
        config.max_duration_sec > 0
        and duration_sec > 0
        and duration_sec > config.max_duration_sec
    )
    facts = {
        "size_mb": size_mb,
        "duration_sec": duration_sec,
        "oversize": oversize,
        "overlong": overlong,
    }

    if size_mb <= 0:  # 0 字节 / 读不到体积：当"未超限"等于喂给模型空内容
        # 这行会原样出现在 ``/videotest`` 的「原因:」行里，因此同样受
        # ``tests/test_interop.py`` 的跨插件兼容约束：不能出现
        # 「拒绝…发送」这种会被守卫当成复核评语的中文搭配（见
        # ``core/diagnostics.VERDICT_LABEL`` 的说明）。
        empty_note = "文件是 0 字节（或读不出体积），不会发送"
        return MediaPlan(action="reject", **facts, reason_code="empty", note=empty_note)
    # 有 ffprobe 却读不出时长 + 后缀是视频容器 -> 损坏；没有 ffprobe 时不误伤
    if has_ffprobe and duration_sec <= 0 and _looks_like_video(mime_or_suffix):
        corrupt_note = "无法解析视频（时长读取失败，文件可能已损坏或不是有效视频）"
        return MediaPlan(
            action="reject", **facts, reason_code="corrupt", note=corrupt_note
        )

    if not oversize and not overlong:
        # 闸门内也要算编码上限：否则闸门放到 200MB 时只写"未超限，直接送"，
        # 紧接着就是 too_large 失败 —— 诊断自相矛盾（0.2.1 的 P2 返工）。
        limit_note = encoding_limit_note(size_mb)
        return MediaPlan(
            action="as-is", **facts, encoding_note=limit_note,
            reason_code="too_large" if limit_note else "ok",
        )

    why = []
    if oversize:
        why.append(f"体积 {size_mb:.1f}MB > 闸门 {config.max_size_mb:.1f}MB")
    if overlong:
        why.append(f"时长 {duration_sec:.1f}s > 上限 {config.max_duration_sec:.1f}s")
    note = "；".join(why)
    code = "oversize" if oversize else "overlong"

    if config.oversize_action == "reject":
        # 拒绝的原因必须分得清"太大"和"太长"：模型据此给用户的建议完全不同
        # （压缩体积 vs 截短时长）。0.2.1 三条都用笼统的 ``rejected``，
        # 于是"仅超时长"被告知 "it exceeds the size limit"（ARCH-REVIEW §4.1）。
        if oversize and overlong:
            reject_code = "rejected_size_and_duration"
        elif oversize:
            reject_code = "rejected_size"
        else:
            reject_code = "rejected_duration"
        return MediaPlan(action="reject", **facts, reason_code=reject_code, note=note)

    # 压缩目标也可能撞编码上限（compress_max_mb 的上界远大于 13.5MB 原始体积）
    target_note = ""
    target_limit = encoding_limit_note(config.compress_max_mb)
    if target_limit:
        target_note = f"；但压缩目标 {config.compress_max_mb:.1f}MB {target_limit}"

    if not has_ffmpeg:
        return MediaPlan(
            action=config.oversize_action, **facts, reason_code="no_ffmpeg",
            note=note, target_note=target_note, needs_ffmpeg=True,
        )
    if duration_sec <= 0:
        return MediaPlan(
            action=config.oversize_action, **facts, reason_code="unknown_duration",
            note=note, target_note=target_note, needs_ffmpeg=True,
        )

    # 截断目标：用户显式设的 max_duration_sec 优先；truncate 策略再按体积预算估算
    # （0.2.0 的 truncate 与 compress 行为完全相同，选项形同虚设），两者取更小的。
    truncate_to = config.max_duration_sec if overlong else 0.0
    if config.oversize_action == "truncate" and oversize:
        estimated = estimate_truncate_seconds(duration_sec, size_mb, config.max_size_mb)
        if estimated > 0:
            truncate_to = min(estimated, truncate_to) if truncate_to else estimated
    return MediaPlan(
        action=config.oversize_action, **facts, reason_code=code,
        truncate_to_sec=truncate_to, note=note, target_note=target_note,
        needs_ffmpeg=True, cache_eligible=bool(config.cache_enabled),
        applies_truncate=will_truncate(duration_sec, truncate_to),
    )


def describe(plan: MediaPlan) -> str:
    """把判决渲染成 ``/videotest`` 的「结论」行正文（不含 ``结论   : `` 前缀）。

    只做渲染：文案是真机验证过的契约（README 与 ``tests/test_docs.py`` 逐字引用），
    任何"改一句更好懂"都算行为变化，必须单独提交。
    """
    if plan.reason_code == "empty":
        return f"体积 {plan.size_mb:.2f}MB（空文件）→ 会以 empty 失败"
    if plan.reason_code == "corrupt":
        return "无法解析（时长读取失败）→ 会以 corrupt 失败"
    if plan.action == "as-is":
        if plan.encoding_note:
            return f"体积/时长在闸门内 → 直接送；但{plan.encoding_note}"
        return "未超限，直接送"
    label = ACTION_LABELS.get(plan.action, plan.action)
    return f"{plan.note} → {label}{plan.target_note}"


def explain(reason_code: str) -> str:
    """机器可读分类 -> 注入给模型的英文短语；未知分类返回空串（调用方自决）。

    判决码（``no_ffmpeg`` / ``unknown_duration`` / …）先按 :data:`KIND_OF_CODE`
    折成对外 kind 再查表，因此 ``explain("no_ffmpeg")`` 与
    ``explain("not_found")`` 说的是同一句话 —— 同一个概念只有一份文案。

    空串是刻意设计：``hooks`` 还要处理**别的错误域**（视觉服务的 HTTP/SSL 错误，
    见 ``core/openai_vision._classify_error``），那些原因没有码；兜底桶
    （``other`` / ``internal``）也刻意留空，让调用方用更具体的 reason 文本。
    """
    code = (reason_code or "").strip()
    phrase = SHORT_REASONS.get(code, "")
    if phrase:
        return phrase
    return SHORT_REASONS.get(KIND_OF_CODE.get(code, ""), "")
