"""``/videotest`` 的全部**用户可见文本**：把事实与判决说成人话。

本模块与 ``core/`` 其余部分一样**不依赖 AstrBot**、不做任何 I/O、不判定任何
策略 —— 判决在 ``core/decide.py``，执行在 ``core/injector.py``，这里只负责
「已经发生 / 将要发生的事怎么说话」。

为什么单独一层：0.2.1 这些渲染散在 ``main.py``（758 行）里 —— ``_diagnose``
165 行、``_diagnose_gates`` 69 行、6 个 ``_xxx_summary/note/preview`` 小函数、
以及 54 行启动日志。于是「报告格式」与「插件生命周期」混在一起：改一行格式要
动主入口，格式本身无法单测，``core/`` 与 ``hooks/`` 的层次也挡住了复用。

**层次约束**（ARCH-REVIEW §2.4）：``core/`` 不得 import ``hooks/``，因此本模块
**不接收** ``event``、也不接收 provider 对象 —— provider 由 ``main`` 解析后以
:class:`ProviderFacts`（三个纯数据字段）交进来。

唯一入口是 :func:`render_report`；:func:`gate_lines` / :func:`startup_lines` /
:func:`videotest_usage` 是同一职责下的其它段落，``main`` 只负责采集事实。
"""

from __future__ import annotations

import os
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from .config import VideoConfig
from .decide import (
    ACTION_LABELS,
    URI_LIMIT_MB,
    URI_LIMIT_RAW_MB,
    MediaPlan,
    allowed_provider_types,
    describe,
)
from .injector import PrepareOutcome
from .limits import format_seconds
from .media import OUTPUT_MIME, guess_video_mime, suffix_of
from .openai_vision import build_vision_prompt
from .placeholder import VIDEO_SUFFIXES

#: 报告标题（``main`` 在"文件不存在 / 不是文件"这类前置失败时也要用）
TITLE = "🔍 视频链路诊断"

#: 「这个视频会发生什么」那一行的标签。
#:
#: **不要改成「判定」**：0.2.1 用的就是 ``判定   : ``，真机上这条报告被另一个
#: 插件（``astrbot_plugin_private_companion``）在发出去之前整行删掉了 ——
#: 它的 ``user_memory.py:9174-9210`` ``_response_review_meta_leak_reason()``
#: 用 ``(?:判定|审核|复核)(?:结果|结论|原因)?\s*[:：]`` 认「复核模型泄漏了判定
#: 字段」，``:9212-9235`` 再逐行丢弃命中的行（真机日志
#: ``astrbot.log`` 21:00:39 / 11:34:53 两条 ERRO ``reason=复核模型输出判定字段``，
#: 以及 21:00:39 的 ``respond.stage`` 实际发出的正文里没有这一行）。
#:
#: 同类风险不止一处：那份守卫还成对使用「中文审核语境词」与
#: 「无法…清洗/发送/改写」两条正则，两边同时命中同样整行丢弃。本模块所有
#: 用户可见文本因此都按「不命中那三条正则」来写，``tests/test_interop.py``
#: 逐行锁住（场景覆盖 + ``clean(text) == text``）。
VERDICT_LABEL = "结论   :"

#: ``prompt :`` 行最多显示多少字符
PROMPT_PREVIEW_CHARS = 200

#: 「— 视觉模型回复节选 —」最多显示多少字符
ANALYSIS_EXCERPT_CHARS = 400

#: 实测经中转站上传约 7-9 秒/MB（按**原始/产物**体积算）
UPLOAD_SEC_PER_MB = 8.0

#: gemini_native 模式下 provider 侧的总预算（不是配置项，是实测的经验值）
PROVIDER_BUDGET_SEC = 180.0


# ----------------------------------------------------------------- 纯数据入参


@dataclass(frozen=True)
class ProbeFacts:
    """``/videotest`` 采集到的文件事实（``main`` 采集，本模块只渲染）。

    ``ffmpeg`` / ``ffprobe`` 用空串表示"没找到"，避免 ``None`` 与本模块的
    "未提供该参数"混为一谈。
    """

    path: str
    size_mb: float
    duration_sec: float
    ffmpeg: str = ""
    ffprobe: str = ""

    @property
    def name(self) -> str:
        return os.path.basename(self.path)


@dataclass(frozen=True)
class ProviderFacts:
    """当前 provider 的**三项纯数据** + 解析失败原因。

    这是 ``core/`` 与 ``hooks/`` 之间唯一的缝：``main`` 调 ``hooks.resolve_provider``
    之后，把 ``provider_type`` / ``provider_id`` / ``modalities`` 三个值填进来，
    渲染层因此不需要 ``event``，也不需要 AstrBot 的 provider 对象。

    ``modalities`` 传的是**原始列表**（不是拼好的字符串）：闸门 ② 要判"是否含
    audio"，把它先拼成字符串再判子串就退回到字符串匹配了 —— 那正是"诊断说的和
    实际跑的不一样"的来源。``None`` 表示 provider 没声明 / 声明了非法类型。
    """

    provider_type: str = ""
    provider_id: str = ""
    modalities: Sequence[str] | None = None
    #: 非空 = provider 解析失败（取不到 provider）的原因码
    error: str = ""


@dataclass(frozen=True)
class VisionReport:
    """一次视觉服务调用的结果（``main`` 从 ``VisionResult`` 转换）。"""

    ok: bool
    elapsed_sec: float = 0.0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    analysis: str = ""
    reason: str = ""

    @classmethod
    def of(cls, result: Any) -> VisionReport:
        """从 ``core.openai_vision.VisionResult`` 转换。

        刻意用 ``getattr`` 而不是 ``import``：渲染层只需要"结果长什么样"，
        不需要认识产生它的那个模块（否则测试得先把视觉服务那一整套拖进来）。
        """
        return cls(
            ok=bool(getattr(result, "ok", False)),
            elapsed_sec=float(getattr(result, "elapsed_sec", 0.0) or 0.0),
            prompt_tokens=int(getattr(result, "prompt_tokens", 0) or 0),
            completion_tokens=int(getattr(result, "completion_tokens", 0) or 0),
            analysis=str(getattr(result, "analysis", "") or ""),
            reason=str(getattr(result, "reason", "") or ""),
        )


# --------------------------------------------------------------- 小段落渲染


def missing_credentials(cfg: VideoConfig) -> list[str]:
    """视觉服务三要素里缺哪几项（凭据齐全性诊断用）。"""
    missing: list[str] = []
    if not cfg.openai_base_url:
        missing.append("openai_base_url")
    if not cfg.openai_api_key:
        missing.append("openai_api_key")
    if not cfg.openai_model:
        missing.append("openai_model")
    return missing


def credentials_summary(cfg: VideoConfig) -> str:
    """一行"凭据是否齐全"，供 ``/videotest`` 使用。"""
    missing = missing_credentials(cfg)
    if not missing:
        return " ✅ base_url / api_key / model 齐全"
    return f" ❌ 缺 {'、'.join(missing)}"


def container_note(path: str) -> str:
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


def mime_line(path: str, action: str) -> str:
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


def prompt_preview(cfg: VideoConfig, mode: str) -> str:
    """将要发送（或注入）的 prompt 预览，压成一行并截断。"""
    if mode != "openai_compat":
        return "（gemini_native 不调用视觉服务，无 prompt）"
    flat = " ".join(build_vision_prompt(cfg, "").split())
    suffix = ""
    if "{question}" not in (cfg.vision_qa_prompt or ""):
        suffix = "（⚠️ vision_qa_prompt 缺 {question}，用户问题会被追加到末尾）"
    clipped = flat[:PROMPT_PREVIEW_CHARS]
    if len(flat) > PROMPT_PREVIEW_CHARS:
        clipped += "…"
    return clipped + suffix


def gate_lines(
    cfg: VideoConfig,
    *,
    provider_type: str = "",
    provider_id: str = "",
    modalities: Sequence[str] | None = None,
    error: str = "",
    mode: str = "gemini_native",
) -> list[str]:
    """输出 provider 信息与三道闸门的**渲染结果**。

    入参只有 provider 的三项纯数据（外加解析失败原因）—— 没有 ``event``、
    没有 provider 对象：``core/`` 不认识 AstrBot（ARCH-REVIEW §2.4）。

    判定用的是与 ``hooks/llm_request._handle`` 同一份真相源
    （``decide.allowed_provider_types``），因此"诊断说的"和"实际跑的"结构上
    不可能分叉。

    Args:
        cfg: 当前配置。
        provider_type: ``provider_config["type"]``（未声明为空串）。
        provider_id: ``provider_config["id"]``（未声明为空串）。
        modalities: 模型声明的模态**原始列表**；``None`` = 未声明/非法。
        error: 非空 = provider 解析失败（取不到 provider）的原因码。
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
            f"         ③ 视觉服务凭据{credentials_summary(cfg)}",
        ]

    declared = (
        ", ".join(str(item) for item in modalities) if modalities else "未声明"
    )
    lines = [
        (
            f"provider: type={provider_type or '未声明'} "
            f"id={provider_id or '未声明'} "
            f"modalities=[{declared}]"
        )
    ]
    allowed = allowed_provider_types(cfg)
    if error:
        gate1 = f"① provider 解析失败 ❌（{error}）"
    elif not cfg.require_gemini_provider:
        gate1 = (
            f"① require_gemini_provider=false → 放行"
            f"（类型 {provider_type or '未声明'}，不校验 MIME 兼容性）"
        )
    elif provider_type in allowed:
        gate1 = f"① 类型 {provider_type} ✅ 在允许列表"
    else:
        gate1 = (
            f"① 类型 {provider_type or '未声明'} ❌ 不在允许列表 {sorted(allowed)}"
        )
    if not modalities:
        # 「无法校验 + 清洗」曾同时命中那份守卫的两条中文正则（见 VERDICT_LABEL
        # 的说明），真机上这行会和结论行一起消失 —— 现在只说后果。
        gate2 = "② modalities 未声明 → ⚠️ 放行（视频块可能被替换成 [Audio] 文本）"
    elif "audio" in modalities:
        gate2 = "② modalities 含 audio ✅"
    else:
        gate2 = (
            "② modalities 不含 audio ❌"
            "（视频块会被替换成 [Audio] 文本，模型看不到画面）"
        )
    gate3 = f"③ 视觉服务凭据{credentials_summary(cfg)}（本模式不使用）"
    return [
        *lines,
        f"闸门   : {gate0}",
        f"         {gate1}",
        f"         {gate2}",
        f"         {gate3}",
    ]


# ------------------------------------------------------------------- 报告本体


def render_report(
    *,
    cfg: VideoConfig,
    mode: str,
    probe: ProbeFacts,
    plan: MediaPlan,
    provider: ProviderFacts | None = None,
    notices: Sequence[str] = (),
    warnings: Sequence[str] = (),
    outcome: PrepareOutcome | None = None,
    elapsed_sec: float = 0.0,
    vision: VisionReport | None = None,
    queue_summary: str = "",
    uri_limit_mb: float = URI_LIMIT_MB,
    uri_raw_limit_mb: float = URI_LIMIT_RAW_MB,
) -> str:
    """把一次 ``/videotest`` 的全部事实渲染成多行报告（**唯一入口**）。

    Args:
        cfg: 当前（已归一化的）配置。
        mode: ``effective_provider_mode``。
        probe: 文件事实（体积 / 时长 / 工具路径）。
        plan: ``decide.classify`` 的判决 —— 结论行**只渲染它**，绝不在这里重算。
        provider: provider 的三项纯数据；``None`` = 不适用（openai_compat）。
        notices: 启动时的"你的输入被收敛/被忽略"提示。
        warnings: 启动时的"已自动修正"告警。
        outcome: ``prepare_video`` 的真实结果；``None`` = 只渲染诊断前半段。
        elapsed_sec: ``prepare_video`` 的墙钟耗时。
        vision: 视觉服务调用结果（仅 openai_compat）。
        queue_summary: 限流器计数摘要（``LimiterStats.summary()``）；**空串 =
            不渲染** —— 只有真的排过队或被排队超时拒过时才多这一行，正常路径的
            报告格式保持不变（VERIFY-A P3-3）。
        uri_limit_mb: data URI 编码后的硬上限（MB）。
        uri_raw_limit_mb: 同一上限换算成原始体积（MB）。

    Returns:
        多行可读报告（直接回给管理员）。
    """
    lines: list[str] = [TITLE, f"文件: {probe.name}", f"体积: {probe.size_mb:.2f} MB"]
    lines.append(
        f"时长: {probe.duration_sec:.1f} s" if probe.duration_sec > 0
        else "时长: 读取失败"
    )
    lines.append("")
    lines.append(
        f"ffmpeg : {'✅ ' + probe.ffmpeg if probe.ffmpeg else '❌ 未找到'}"
    )
    lines.append(
        f"ffprobe: {'✅ ' + probe.ffprobe if probe.ffprobe else '❌ 未找到'}"
    )
    lines.append(
        f"配置   : 闸门={cfg.max_size_mb:.1f}MB 上限={cfg.compress_max_mb:.1f}MB "
        f"CRF={cfg.compress_crf} "
        f"策略={ACTION_LABELS.get(cfg.oversize_action, cfg.oversize_action)}"
    )
    # 配置被自动修正过时，把"你写的原值 → 生效值"一起打出来。
    for notice in notices:
        lines.append(f"配置提示: {notice}")
    for warning in warnings:
        lines.append(f"配置修正: {warning}")
    # 与体积闸门无关的第二道体积限制：data URI 编码上限。
    lines.append(
        f"编码上限: data URI {uri_limit_mb:.0f}MB ≈ 原始 {uri_raw_limit_mb:.1f}MB"
        "（编码后超过必失败，与体积闸门无关）"
    )

    # ---- 模式 / 凭据（0.2.0 只报模式）----
    degraded = cfg.provider_mode != mode
    lines.append(
        f"模式   : {mode}"
        + (
            f"（provider_mode={cfg.provider_mode} 三项未配齐，运行时降级）"
            if degraded
            else ""
        )
    )
    lines.append(f"凭据   : 视觉服务{credentials_summary(cfg)}")

    # ---- provider 与三道闸门（0.2.0 完全不诊断，而这正是最常见的失败点）----
    facts = provider or ProviderFacts()
    lines.extend(
        gate_lines(
            cfg,
            provider_type=facts.provider_type,
            provider_id=facts.provider_id,
            modalities=facts.modalities,
            error=facts.error,
            mode=mode,
        )
    )

    # ---- 显式路径的扩展名风险（闸门 ⓪ 在本指令里不适用）----
    note = container_note(probe.path)
    if note:
        lines.append(f"提醒   : {note}")

    # ---- 体积 / 时长判定：只渲染 decide 的判决，绝不在这里重算 ----
    lines.append(f"{VERDICT_LABEL} {describe(plan)}")

    # ---- 限流排队摘要（VERIFY-A P3-3）----
    # 只在**真的排过队 / 超时过**时出现（``LimiterStats.summary()`` 为空串时
    # 一行都不加），正常路径的报告格式保持原样。
    if queue_summary:
        # 标签用「队列」而不是「排队」：summary() 自己已经以「排队 N 次」开头，
        # 用「排队   : 」会渲染成「排队   : 排队 3 次…」（VERIFY-VV-R7 P2-3）。
        lines.append(f"队列   : {queue_summary}")

    # ---- 将要发给视觉服务的 prompt 预览（截断）----
    lines.append(f"prompt : {prompt_preview(cfg, mode)}")

    if outcome is None:  # 只要"诊断前半段"（格式单测与降级场景用）
        return "\n".join(lines)

    lines.append("")
    if not outcome.ok or outcome.injection is None:
        lines.append(f"❌ 准备失败（{elapsed_sec:.2f}s）")
        if outcome.kind:
            lines.append(f"分类: {outcome.kind}")
        lines.append(f"原因: {outcome.reason}")
        return "\n".join(lines)

    inj = outcome.injection
    is_openai = mode == "openai_compat"
    budget = cfg.openai_timeout_sec if is_openai else PROVIDER_BUDGET_SEC
    budget_name = "视觉服务预算" if is_openai else "provider 预算"
    uri_mb = len(inj.data_uri) / 1024 / 1024
    est_upload = inj.size_mb * UPLOAD_SEC_PER_MB

    lines.append(f"✅ 准备成功（{elapsed_sec:.2f}s）")
    lines.append(f"动作: {ACTION_LABELS.get(inj.action, inj.action)}")
    lines.append(f"产物: {inj.size_mb:.2f} MB / {inj.duration_sec:.1f} s")
    lines.append(f"编码: data URI {uri_mb:.2f} MB")
    # 声明给 provider 的 MIME：未知后缀回落 video/mp4 时标 (推断)，
    # 免得"✅ 准备成功 + video/mp4"被读成"这个 gif 真的会以视频发出去"。
    lines.append(mime_line(inj.used_path, inj.action))
    if inj.note:
        lines.append(f"说明: {inj.note}")
    lines.append("")
    # 用 format_seconds 而不是 ``%.0f``：0.05MB 的视频算出来是 0.4s，``%.0f``
    # 会渲染成「约 0s」——"不许把 0.4 秒写成 0 秒"这条规则在排队/TTL 链路早就
    # 生效了，这里曾经遗漏（ARCH-VV-C2 P2-5）。
    lines.append(
        f"预计上传耗时: 约 {format_seconds(est_upload)}"
        f"（{budget_name} {budget:.0f}s）"
    )
    if est_upload > budget:
        hint = "openai_timeout_sec" if is_openai else "provider timeout"
        lines.append(f"⚠️ 可能超时，建议调低体积闸门或提高 {hint}")
    else:
        lines.append("✅ 在预算内")

    if is_openai and vision is not None:
        lines.append("")
        lines.extend(_vision_lines(vision))
    return "\n".join(lines)


def _vision_lines(vision: VisionReport) -> list[str]:
    """openai_compat 模式的端到端验证段（真实调用一次视觉服务的结果）。"""
    if not vision.ok:
        return [f"❌ 视觉调用失败: {vision.reason}"]
    excerpt = vision.analysis[:ANALYSIS_EXCERPT_CHARS]
    if len(vision.analysis) > ANALYSIS_EXCERPT_CHARS:
        excerpt += "…"
    return [
        (
            f"✅ 视觉调用成功（{vision.elapsed_sec:.1f}s | "
            f"token {vision.prompt_tokens}+{vision.completion_tokens}）"
        ),
        "— 视觉模型回复节选 —",
        excerpt,
    ]


# ------------------------------------------------------------------ 启动日志


def startup_lines(
    cfg: VideoConfig,
    *,
    version: str,
    ffmpeg: str = "",
    ffprobe: str = "",
    cache_root: str = "",
) -> list[tuple[str, str]]:
    """启动时那 6 段日志的文本（``main`` 只负责按级别打出去）。

    0.2.1 这 54 行直接写在 ``VideoVisionPlugin.__init__`` 里：文案即契约
    （"只告警一次"的承诺、密钥只露后 4 位），却无法单测 —— 只有把插件整个装
    起来、跑起来才能看到。这里把**文本**与**副作用**分开：本函数纯计算，
    ``mark_degradation_warned()`` 之类的状态仍留在 ``main``。

    Returns:
        ``[(level, message), ...]``，``level`` 是 ``"info"`` / ``"warning"``。
    """
    strategy = ACTION_LABELS.get(cfg.oversize_action, cfg.oversize_action)
    lines: list[tuple[str, str]] = [
        (
            "info",
            (
                f"[VideoVision] v{version} 已加载 | 启用={cfg.enable} | "
                f"体积闸门={cfg.max_size_mb:.1f}MB | "
                f"压缩上限={cfg.compress_max_mb:.1f}MB | "
                f"CRF={cfg.compress_crf} | "
                f"超限策略={strategy} | "
                f"仅 Gemini={cfg.require_gemini_provider}"
            ),
        ),
        (
            "info",
            (
                f"[VideoVision] ffmpeg={ffmpeg or '未找到'} | "
                f"ffprobe={ffprobe or '未找到'} | "
                f"缓存={cache_root if cfg.cache_enabled else '已关闭'}"
            ),
        ),
    ]
    if cfg.effective_provider_mode == "openai_compat":
        lines.append(
            (
                "info",
                (
                    f"[VideoVision] 模式=openai_compat | "
                    f"视觉服务={cfg.openai_base_url or '未配置'} | "
                    f"模型={cfg.openai_model} | key={_mask_key(cfg.openai_api_key)} | "
                    f"超时={cfg.openai_timeout_sec:.0f}s"
                ),
            )
        )
    else:
        lines.append(
            (
                "info",
                (
                    f"[VideoVision] 模式=gemini_native | "
                    f"仅 Gemini provider={cfg.require_gemini_provider}"
                ),
            )
        )
    if cfg.provider_mode != cfg.effective_provider_mode:
        lines.append(
            (
                "warning",
                (
                    "[VideoVision] provider_mode=openai_compat 但 openai_base_url / "
                    "openai_api_key / openai_model 未配齐，已降级为 gemini_native "
                    "（老配置零改动仍可继续用）。要使用 OpenAI 兼容视觉服务，"
                    "请把这三项补齐。"
                ),
            )
        )
    if not ffmpeg:
        lines.append(
            (
                "warning",
                (
                    "[VideoVision] 未找到 ffmpeg：超限视频将无法压缩，会被直接跳过。"
                    "请安装 ffmpeg 或在插件配置里指定 ffmpeg_path。"
                ),
            )
        )
    return lines


def _mask_key(key: str) -> str:
    """密钥只露后 4 位。

    日志经常被贴到 issue 里，前 6 位足以还原出「sk-xxx」这类前缀，没必要泄露。
    """
    if len(key) >= 8:
        return f"…{key[-4:]}"
    return "已配置" if key else "未配置"


# ------------------------------------------------------- ``/videotest`` 用法

#: ``/videotest`` 的用法。三条入口只有前两条在任何平台都成立，第三条看平台：
#:
#: * ① 显式路径 —— 永远是最后的手段；
#: * ② 引用（回复）含视频的消息 —— **唯一**让「视频」和「命令」落在同一条
#:   事件里的自然方式。被引用消息的消息段在 ``Reply.chain`` 里
#:   （``aiocqhttp_platform_adapter.py:328-337``），``Reply.message_str`` 只是文本；
#: * ③ 视频和命令写在同一条消息里（平台支持图文混排时）。
#:
#: 0.2.1 的旧文案写「或先发一个视频/视频文件，再发送 /videotest」，这句话是
#: **做不到的**：那是两条独立消息、两个独立事件。真机证据见 LIVE-TEST。
USAGE_TEXT = (
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
VIDEO_REASON_TEXT = {
    "no-media": "本条消息和引用的消息里都没有视频",
    "quote-content-missing": (
        "引用了消息，但平台没有把被引用消息的内容交给插件"
        "（aiocqhttp 的 get_msg 失败时就是这样），看不到里面的视频"
    ),
    "quote-no-video": "引用的那条消息里没有视频",
    "video-unreadable": "找到了视频，但没能取到本地文件（链接可能已过期）",
}


def videotest_usage(reason: str = "") -> str:
    """用法提示；``reason`` 非空时把「本次为什么没找到视频」一起说出来。"""
    note = VIDEO_REASON_TEXT.get(reason, "")
    if not note:
        return USAGE_TEXT
    head, _, tail = USAGE_TEXT.partition("\n注意：")
    return f"{head}\n本次没找到视频：{note}\n注意：{tail}"
