"""跨插件互操作：我们的外发文本必须能穿过**别人**的「回复复核评语泄漏」守卫。

真机故障（不是猜测）
====================

私聊里引用视频发 ``/videotest``，命令执行成功、报告也发出去了，但报告里最关键
的那一行 **在用户看到之前被另一个插件删掉了**。

守卫在 ``astrbot_plugin_private_companion``（已安装的三方插件）里：

* ``user_memory.py:9174-9210`` ``_response_review_meta_leak_reason(text)`` ——
  认「这段文本是复核模型泄漏出来的内部判断」，其中 ``:9208`` 是::

      if re.search(r"(?:判定|审核|复核)(?:结果|结论|原因)?\\s*[:：]", compact):
          return "复核模型输出判定字段"

* ``user_memory.py:9212-9235`` ``_strip_response_review_meta_leak(text)`` ——
  **逐行**丢弃 ``reason`` 非空的行，再 ``"\\n".join(kept)``。
* ``main.py:10677-10700`` ``_restore_response_review_meta_leak_before_send()`` ——
  外发前把整条回复换成 ``Plain(cleaned)``，并打一条
  ``logger.error("发送前拦截到回复复核内部判断: ...")``。

真机证据（``C:\\Users\\UwU\\.astrbot\\data\\logs\\astrbot.log``，两次）::

    21:00:39 ERRO 发送前拦截到回复复核内部判断: reason=复核模型输出判定字段
    11:34:53 ERRO 发送前拦截到回复复核内部判断: reason=复核模型输出判定字段

而且 ``21:00:39`` 那次 ``[Core] [INFO] [respond.stage:233]`` 打出的**实际发出的
正文**里，``判定   : 未超限，直接送`` 整行不见了，闸门行前面的缩进也一起被
``line.strip()`` 吃掉（见下面 :data:`REAL_REPORT_2100`）。

为什么在我们这一侧修，而不是改它
================================

1. 那份守卫是**已安装的第三方插件**，本仓库既没有所有权也不该去改（它的行为对
   其它插件同样生效；改它会波及整个陪伴体系）。
2. 它作用于**所有**出站消息、只认文本、不看来源 —— 换句话说，"被判成复核评语"
   是任何插件都可能踩到的兼容性问题，能改的只有我们自己发出去的文本。
3. 于是本文件把它的判定逻辑**逐字抄过来**当契约，锁住两件事：

   * 逐行 ``reason(line) == ""``（不会被删）；
   * ``clean(text) == text``（最终目标：守卫对整条报告不做任何改动）。

``test_guard_replica_reproduces_the_real_failures`` 是**反向对照**：同一份复刻
实现必须能删掉旧文案（``判定   : …`` 与 ``无法校验，视频块可能被清洗``），否则
本文件全绿也说明不了任何问题。

本文件只写"文本契约"，不改任何行为；行为回归仍在各自的测试文件里。
"""

from __future__ import annotations

import ast
import re
from collections.abc import Iterator
from pathlib import Path

import pytest

from vv_plugin.core import diagnostics
from vv_plugin.core.config import VideoConfig
from vv_plugin.core.decide import SHORT_REASONS, classify
from vv_plugin.core.diagnostics import (
    ProbeFacts,
    ProviderFacts,
    VisionReport,
    gate_lines,
    render_report,
    startup_lines,
    videotest_usage,
)
from vv_plugin.core.injector import (
    PrepareOutcome,
    VideoInjection,
    _blocked_reason,
)
from vv_plugin.core.openai_vision import _classify_error

REPO = Path(__file__).resolve().parents[1]

# ===========================================================================
# 守卫的正则：**逐字抄自** astrbot_plugin_private_companion 的
# ``user_memory.py:9174-9210``。不要在这里"顺手优化"——它们就是对方的行为，
# 优化过的副本会让本文件失去意义（改一个字就可能放过真实的回归）。
# ===========================================================================

#: ``user_memory.py:9181``
R_MAYBE = re.compile(r"\bmaybe\s+\d+(?:\.\d+)?%\s+of\s+the\s+time\b")
#: ``user_memory.py:9183``
R_AT_END = re.compile(r"\bat\s+the\s+(?:very\s+)?end\s+of\s+(?:a|the)\s+run\b")
#: ``user_memory.py:9185-9188`` —— 任务书里的 ``R_EN``
R_EN = re.compile(
    r"\b(?:decision|verdict|review result|review reason|reason)\s*[:：]", re.I
)
#: ``user_memory.py:9190-9195``
R_EN_TEXT = re.compile(
    r"\b(?:response|output|message)\b.{0,80}\b(?:needs?\s+(?:to\s+be\s+)?rewritten|"
    r"cannot\s+be\s+saniti[sz]ed|formatting\s+(?:issue|problem)|should\s+not\s+be\s+sent)\b"
)
#: ``user_memory.py:9196-9200`` —— 任务书里的 ``R_CTX``
R_CTX = re.compile(
    r"(?:原(?:回复|文本|输出)|这条(?:回复|消息|输出)|回复内容|输出内容|后处理|清洗|复核|审核|"
    r"格式化表达|重复标点|一字废话|最终回复|正常人无法容忍)"
)
#: ``user_memory.py:9201-9205`` —— 任务书里的 ``R_VERDICT``（与 ``R_CTX`` 成对才删行）
R_VERDICT = re.compile(
    r"(?:无法|不能|不应|不宜|不适合|未通过|拒绝|需要|应当|建议).{0,24}"
    r"(?:清洗|规整|发送|通过|重写|改写|修正)"
)
#: ``user_memory.py:9208`` —— 任务书里的 ``R_META``
R_META = re.compile(r"(?:判定|审核|复核)(?:结果|结论|原因)?\s*[:：]")

#: 任务书点名的三条（``R_CTX`` 与 ``R_VERDICT`` 必须**同时**命中才删行）
NAMED_PATTERNS = {
    "R_EN": R_EN,
    "R_CTX": R_CTX,
    "R_VERDICT": R_VERDICT,
    "R_META": R_META,
}

#: 只出现一处就会删行的分支（``R_CTX``/``R_VERDICT`` 要成对，故不入此表）
SINGLE_SHOT = {
    "R_MAYBE": R_MAYBE,
    "R_AT_END": R_AT_END,
    "R_EN": R_EN,
    "R_EN_TEXT": R_EN_TEXT,
    "R_META": R_META,
}


def guard_hits(text: str) -> list[str]:
    """这一行命中了守卫的哪几条正则（空列表 = 一条都不命中）。"""
    compact = re.sub(r"\s+", " ", str(text or "")).strip()
    if not compact:
        return []
    lower = compact.lower()
    hits: list[str] = []
    for name, pattern in SINGLE_SHOT.items():
        haystack = lower if name.startswith("R_EN") or name == "R_AT_END" else compact
        if pattern.search(haystack):
            hits.append(name)
    if R_CTX.search(compact):
        hits.append("R_CTX")
    if R_VERDICT.search(compact):
        hits.append("R_VERDICT")
    return hits


def guard_reason(text: str) -> str:
    """``user_memory.py:9174-9210`` 的逐字等价实现（返回空串 = 不会被删）。"""
    raw = str(text or "").strip()
    if not raw:
        return ""
    compact = re.sub(r"\s+", " ", raw).strip()
    lower = compact.lower()
    if R_MAYBE.search(lower):
        return "复核模型输出概率说明"
    if R_AT_END.search(lower):
        return "复核模型输出运行说明"
    if R_EN.search(lower):
        return "复核模型输出判定字段"
    if R_EN_TEXT.search(lower):
        return "复核模型输出英文审核评语"
    if R_CTX.search(compact) and R_VERDICT.search(compact):
        return "复核模型输出中文审核评语"
    if R_META.search(compact):
        return "复核模型输出判定字段"
    return ""


def guard_clean(text: str) -> tuple[str, str]:
    """``user_memory.py:9212-9235`` 的逐字等价实现（返回 ``(改后文本, 原因)``）。"""
    raw = str(text or "").strip()
    if not raw:
        return "", ""
    kept: list[str] = []
    reasons: list[str] = []
    for line in raw.splitlines():
        stripped = line.strip()
        if not stripped:
            if kept and kept[-1] != "":
                kept.append("")
            continue
        reason = guard_reason(stripped)
        if reason:
            reasons.append(reason)
            continue
        kept.append(stripped)
    if not reasons:
        whole_reason = guard_reason(raw)
        if whole_reason:
            return "", whole_reason
        return raw, ""
    return "\n".join(kept).strip(), "、".join(dict.fromkeys(reasons))


# ===========================================================================
# 真机样张：21:00:39 那次 ``/videotest`` **实际发出**的正文（守卫已经动过手：
# ``判定   : 未超限，直接送`` 整行消失，闸门行的缩进被 strip 掉）。
# 来源：astrbot.log 的 ``[Core] [INFO] [respond.stage:233]: Prepare to send - …``
# ===========================================================================

REAL_REPORT_2100 = """\
🔍 视频链路诊断
文件: media_video_20261026210038230_9755.mp4
体积: 5.71 MB
时长: 11.5 s

ffmpeg : ✅ D:\\ffmpeg\\bin\\ffmpeg.EXE
ffprobe: ✅ D:\\ffmpeg\\bin\\ffprobe.EXE
配置   : 闸门=13.0MB 上限=11.7MB CRF=30 策略=压缩
配置提示: compress_max_mb=15.0 超过体积闸门 max_size_mb=13.0，已按闸门的 90% 收敛为 11.7
编码上限: data URI 18MB ≈ 原始 13.5MB（编码后超过必失败，与体积闸门无关）
模式   : gemini_native（provider_mode=openai_compat 三项未配齐，运行时降级）
凭据   : 视觉服务 ❌ 缺 openai_base_url、openai_api_key
provider: type=googlegenai_chat_completion id=AlmoonsGemini-1/［抗截断］gemini-3.8flash modalities=[text, image, audio, tool_use]
闸门   : ⓪ File 组件扩展名筛选 → 本指令用显式路径，不适用
① 类型 googlegenai_chat_completion ✅ 在允许列表
② modalities 含 audio ✅
③ 视觉服务凭据 ❌ 缺 openai_base_url、openai_api_key（本模式不使用）
prompt : （gemini_native 不调用视觉服务，无 prompt）

✅ 准备成功（0.22s）
动作: 直接送（未压缩）
产物: 5.71 MB / 11.5 s
编码: data URI 7.61 MB
MIME   : video/mp4

预计上传耗时: 约 46s（provider 预算 180s）
✅ 在预算内"""  # noqa: E501 - 样张必须逐字对齐真机日志，不许换行


# ===========================================================================
# 场景：把每一种用户能看到的文本都造出来（全部走**生产代码**，不手抄文案）
# ===========================================================================


def _injection(**overrides) -> VideoInjection:
    data_uri = "data:video/mp4;base64," + "A" * 2_500_000
    options = {
        "block": {"type": "audio_url", "audio_url": {"url": data_uri}},
        "source_path": r"C:\videos\demo.mp4",
        "used_path": r"C:\cache\compressed.mp4",
        "size_mb": 1.79,
        "duration_sec": 20.0,
        "action": "compressed",
        "note": "体积 21.8MB > 闸门 13.0MB -> 1.8MB",
    }
    options.update(overrides)
    return VideoInjection(**options)


def _probe(path: str = r"C:\videos\demo.mp4", **overrides) -> ProbeFacts:
    options = {
        "size_mb": 21.79,
        "duration_sec": 20.0,
        "ffmpeg": r"D:\ffmpeg\bin\ffmpeg.exe",
        "ffprobe": r"D:\ffmpeg\bin\ffprobe.exe",
    }
    options.update(overrides)
    return ProbeFacts(path=path, **options)


def _report(
    cfg: VideoConfig,
    *,
    mode: str = "gemini_native",
    probe: ProbeFacts | None = None,
    size_mb: float = 21.79,
    duration_sec: float = 20.0,
    suffix: str = ".mp4",
    provider: ProviderFacts | None = None,
    notices: tuple[str, ...] = (),
    warnings: tuple[str, ...] = (),
    outcome: PrepareOutcome | None = None,
    elapsed_sec: float = 0.12,
    vision: VisionReport | None = None,
    has_ffmpeg: bool = True,
    has_ffprobe: bool = True,
    plan=None,
) -> str:
    facts = probe or _probe(size_mb=size_mb, duration_sec=duration_sec)
    plan = plan or classify(
        size_mb=size_mb,
        duration_sec=duration_sec,
        config=cfg,
        mime_or_suffix=suffix,
        has_ffmpeg=has_ffmpeg,
        has_ffprobe=has_ffprobe,
    )
    return render_report(
        cfg=cfg,
        mode=mode,
        probe=facts,
        plan=plan,
        provider=provider,
        notices=notices,
        warnings=warnings,
        outcome=outcome,
        elapsed_sec=elapsed_sec,
        vision=vision,
    )


def _failed(plan, cfg: VideoConfig, source: str = r"C:\videos\x.mp4") -> PrepareOutcome:
    """用 ``injector._blocked_reason`` 造一条真实的「准备失败」结果。

    刻意不手抄失败文案：``原因:`` 行是用户能看到的，必须来自生产代码。
    """
    return PrepareOutcome(
        False, reason=_blocked_reason(plan, cfg, source), kind=plan.kind
    )


NATIVE = VideoConfig(max_size_mb=13.0, compress_max_mb=8.0, max_duration_sec=0.0)
OPENAI = VideoConfig(
    max_size_mb=13.0,
    compress_max_mb=8.0,
    max_duration_sec=0.0,
    provider_mode="openai_compat",
    openai_base_url="https://vision.invalid/v1/chat/completions",
    openai_api_key="sk-unit-test-value",
    openai_model="vision-model",
)
GEMINI_PROVIDER = ProviderFacts(
    "googlegenai_chat_completion", "gemini-2.5-flash", ["text", "image", "audio"]
)


def _scenes() -> Iterator[tuple[str, str]]:
    # ---- 四种素材场景（任务书点名）----
    yield "未超限/直接送", _report(
        NATIVE,
        size_mb=5.71,
        duration_sec=11.5,
        probe=_probe(size_mb=5.71, duration_sec=11.5),
        provider=GEMINI_PROVIDER,
        outcome=PrepareOutcome(
            True,
            _injection(
                used_path=r"C:\videos\demo.mp4",
                size_mb=5.71,
                duration_sec=11.5,
                action="as-is",
                note="",
            ),
        ),
    )
    yield "超限/压缩", _report(
        NATIVE, provider=GEMINI_PROVIDER, outcome=PrepareOutcome(True, _injection())
    )
    yield "corrupt", _report(
        NATIVE,
        size_mb=2.0,
        duration_sec=0.0,
        probe=_probe(size_mb=2.0, duration_sec=0.0),
        plan=classify(
            size_mb=2.0, duration_sec=0.0, config=NATIVE, mime_or_suffix=".mp4"
        ),
        outcome=PrepareOutcome(
            False,
            reason=(
                "无法解析视频（时长读取失败，文件可能已损坏或不是有效视频）: "
                r"C:\videos\broken.mp4"
            ),
            kind="corrupt",
        ),
    )
    yield "gif 提醒（未知后缀）", _report(
        NATIVE,
        size_mb=0.17,
        duration_sec=2.0,
        suffix=".gif",
        probe=_probe(path="anim.gif", size_mb=0.17, duration_sec=2.0),
        outcome=PrepareOutcome(
            True,
            _injection(
                used_path="anim.gif",
                size_mb=0.17,
                duration_sec=2.0,
                action="as-is",
                note="",
            ),
        ),
    )

    # ---- 四种策略 ----
    for label, action in (("reject 体积", "reject"), ("truncate", "truncate")):
        cfg = VideoConfig(
            max_size_mb=13.0,
            compress_max_mb=8.0,
            max_duration_sec=0.0,
            oversize_action=action,
        )
        plan = classify(
            size_mb=21.79, duration_sec=20.0, config=cfg, mime_or_suffix=".mp4"
        )
        if action == "reject":
            outcome = _failed(plan, cfg)
        else:
            note = f"{plan.note} -> 8.4MB（按体积预算截断到 7.9s）"
            outcome = PrepareOutcome(
                True, _injection(action="truncated", note=note)
            )
        yield label, _report(cfg, plan=plan, outcome=outcome, elapsed_sec=0.9)

    overlong = VideoConfig(
        max_size_mb=13.0,
        compress_max_mb=8.0,
        max_duration_sec=10.0,
        oversize_action="reject",
    )
    plan = classify(
        size_mb=5.0, duration_sec=20.0, config=overlong, mime_or_suffix=".mp4"
    )
    yield "reject 时长", _report(
        overlong,
        size_mb=5.0,
        plan=plan,
        outcome=_failed(plan, overlong),
    )
    both = VideoConfig(
        max_size_mb=13.0,
        compress_max_mb=8.0,
        max_duration_sec=10.0,
        oversize_action="reject",
    )
    plan = classify(
        size_mb=21.79, duration_sec=20.0, config=both, mime_or_suffix=".mp4"
    )
    yield "reject 体积+时长", _report(
        both,
        plan=plan,
        outcome=_failed(plan, both),
    )

    # ---- 边界：没有 ffmpeg / 读不出时长 / 空文件 ----
    no_tools = classify(
        size_mb=21.79,
        duration_sec=20.0,
        config=NATIVE,
        mime_or_suffix=".mp4",
        has_ffmpeg=False,
        has_ffprobe=False,
    )
    yield "无 ffmpeg", _report(
        NATIVE,
        probe=_probe(ffmpeg="", ffprobe=""),
        has_ffmpeg=False,
        has_ffprobe=False,
        plan=no_tools,
        outcome=_failed(no_tools, NATIVE),
    )
    unknown = classify(
        size_mb=21.79,
        duration_sec=0.0,
        config=NATIVE,
        mime_or_suffix=".mov",
        has_ffprobe=False,
    )
    yield "空时长（无 ffprobe）", _report(
        NATIVE,
        duration_sec=0.0,
        suffix=".mov",
        probe=_probe(duration_sec=0.0, ffprobe=""),
        has_ffprobe=False,
        plan=unknown,
        outcome=_failed(unknown, NATIVE),
    )
    empty_cfg = NATIVE
    empty_plan = classify(
        size_mb=0.0, duration_sec=0.0, config=empty_cfg, mime_or_suffix=".mp4"
    )
    yield "空文件（empty）", _report(
        empty_cfg,
        size_mb=0.0,
        duration_sec=0.0,
        probe=_probe(size_mb=0.0, duration_sec=0.0),
        plan=empty_plan,
        outcome=_failed(empty_plan, empty_cfg, r"C:\videos\zero.mp4"),
    )

    # ---- provider 与三道闸门的每一条分支 ----
    yield "无 provider（解析失败）", _report(
        NATIVE,
        size_mb=5.71,
        duration_sec=11.5,
        provider=ProviderFacts(error="no-context"),
    )
    yield "modalities 未声明", _report(
        NATIVE,
        size_mb=5.71,
        duration_sec=11.5,
        provider=ProviderFacts(
            "googlegenai_chat_completion", "gemini-2.5-flash", None
        ),
    )
    yield "modalities 不含 audio", _report(
        NATIVE,
        size_mb=5.71,
        duration_sec=11.5,
        provider=ProviderFacts("openai_chat_completion", "gpt-4o", ["text"]),
    )
    escape = VideoConfig(require_gemini_provider=False)
    yield "require_gemini_provider=false 逃生门", _report(
        escape,
        size_mb=5.71,
        duration_sec=11.5,
        provider=ProviderFacts("openai_chat_completion", "gpt-4o", ["audio"]),
    )
    yield "闸门 ① 不在允许列表", "\n".join(
        gate_lines(
            VideoConfig(require_gemini_provider=True),
            provider_type="openai_chat_completion",
            modalities=["audio"],
        )
    )
    yield "闸门 ① 解析失败", "\n".join(
        gate_lines(VideoConfig(require_gemini_provider=True), error="no-context")
    )

    # ---- 体积闸门之外的两道限制 ----
    wide = VideoConfig(max_size_mb=200.0, compress_max_mb=8.0, max_duration_sec=0.0)
    yield "编码上限预判（未超限但撞 18MB）", _report(
        wide,
        size_mb=15.0,
        probe=_probe(size_mb=15.0),
        plan=classify(
            size_mb=15.0, duration_sec=20.0, config=wide, mime_or_suffix=".mp4"
        ),
    )
    yield "预计上传超预算", _report(
        wide,
        size_mb=25.0,
        probe=_probe(size_mb=25.0),
        plan=classify(
            size_mb=25.0, duration_sec=20.0, config=wide, mime_or_suffix=".mp4"
        ),
        outcome=PrepareOutcome(
            True, _injection(size_mb=25.0, action="as-is", note="")
        ),
        elapsed_sec=0.31,
    )
    yield "压缩目标撞编码上限", _report(
        VideoConfig(max_size_mb=200.0, compress_max_mb=15.0, max_duration_sec=0.0),
        size_mb=21.79,
        probe=_probe(),
        plan=classify(
            size_mb=21.79,
            duration_sec=20.0,
            config=VideoConfig(
                max_size_mb=200.0, compress_max_mb=15.0, max_duration_sec=0.0
            ),
            mime_or_suffix=".mp4",
        ),
        outcome=PrepareOutcome(True, _injection()),
    )

    # ---- 配置提示 / 修正（含 0.5MB 闸门那条原本带「无法…修正」的文案）----
    tiny, tiny_warnings = VideoConfig(
        max_size_mb=0.5, compress_max_mb=0.5
    ).normalized()
    yield "配置修正（0.5MB 闸门）", _report(
        tiny,
        size_mb=0.2,
        duration_sec=1.0,
        probe=_probe(size_mb=0.2, duration_sec=1.0),
        warnings=tuple(tiny_warnings),
        plan=classify(
            size_mb=0.2, duration_sec=1.0, config=tiny, mime_or_suffix=".mp4"
        ),
    )
    yield "配置提示 + 修正", _report(
        NATIVE,
        size_mb=5.71,
        duration_sec=11.5,
        notices=("未知配置键已忽略: compress_max_mb2",),
        warnings=(
            "compress_max_mb(15.0) 必须小于 max_size_mb(13.0)，已自动下调为 11.7",
        ),
    )

    # ---- openai_compat：端到端那一段 ----
    oa_outcome = PrepareOutcome(
        True,
        _injection(
            source_path=r"C:\videos\demo.mp4",
            used_path=r"C:\videos\demo.mp4",
            size_mb=0.93,
            duration_sec=3.0,
            action="as-is",
            note="",
        ),
    )
    yield "openai_compat/视觉成功", _report(
        OPENAI,
        mode="openai_compat",
        size_mb=0.93,
        duration_sec=3.0,
        probe=_probe(size_mb=0.93, duration_sec=3.0),
        plan=classify(
            size_mb=0.93, duration_sec=3.0, config=OPENAI, mime_or_suffix=".mp4"
        ),
        outcome=oa_outcome,
        vision=VisionReport(
            True,
            elapsed_sec=1.25,
            prompt_tokens=262,
            completion_tokens=17,
            analysis="画面里有一只猫在跑（第三方正文，不可信数据）",
        ),
    )
    vision_reasons = [
        ("超时", _classify_error(TimeoutError("timed out"))),
        ("SSL", _classify_error(RuntimeError("CERTIFICATE_VERIFY_FAILED"))),
        ("连不上", _classify_error(OSError("getaddrinfo failed"))),
        ("其他", _classify_error(ValueError("boom"))),
        # 下面四条是 core/openai_vision.py 里的固定文案（:247 / :284 / :293 / :298）
        ("无 httpx", "视觉服务不可用：缺少 httpx（AstrBot 官方环境自带，请检查安装）"),
        ("HTTP 非 200", "视觉服务返回 HTTP 500: internal error"),
        ("响应解析失败", "视觉服务响应解析失败: JSONDecodeError: x"),
        ("空内容", "视觉服务返回了空内容"),
    ]
    for label, reason in vision_reasons:
        yield f"openai_compat/视觉失败：{label}", _report(
            OPENAI,
            mode="openai_compat",
            size_mb=0.93,
            duration_sec=3.0,
            probe=_probe(size_mb=0.93, duration_sec=3.0),
            plan=classify(
                size_mb=0.93, duration_sec=3.0, config=OPENAI, mime_or_suffix=".mp4"
            ),
            outcome=oa_outcome,
            vision=VisionReport(False, reason=reason),
        )
    no_placeholder = VideoConfig(vision_qa_prompt="没有占位符的模板")
    yield "prompt 预览缺占位符", _report(
        no_placeholder,
        mode="openai_compat",
        size_mb=5.71,
        duration_sec=11.5,
        plan=classify(
            size_mb=5.71,
            duration_sec=11.5,
            config=no_placeholder,
            mime_or_suffix=".mp4",
        ),
    )
    yield "prompt 预览被截断", "\n".join(
        [diagnostics.prompt_preview(VideoConfig(vision_describe_prompt="词" * 1000),
                                    "openai_compat")]
    )

    # ---- 用法提示与错误回复（`/videotest` 的其它出口）----
    yield "用法提示/无原因", videotest_usage("")
    for code in (
        "no-media",
        "quote-content-missing",
        "quote-no-video",
        "video-unreadable",
    ):
        yield f"用法提示/{code}", videotest_usage(code)
    yield "错误回复/非管理员", "该指令仅管理员可用。"
    yield "错误回复/文件不存在", (
        f"{diagnostics.TITLE}\n❌ 文件不存在: C:\\videos\\nope.mp4"
    )
    yield "错误回复/不是文件", f"{diagnostics.TITLE}\n❌ 不是文件: C:\\videos"
    yield "错误回复/诊断异常", "❌ 诊断过程异常: RuntimeError: boom"

    # ---- 启动日志与注入给模型的英文短语（后者可能被模型复读）----
    yield "启动日志", "\n".join(
        message
        for _level, message in startup_lines(
            VideoConfig(openai_api_key="sk-super-secret-value"),
            version="0.2.1",
            ffmpeg="",
            ffprobe="",
            cache_root="/tmp/cache",
        )
    )
    yield "启动日志/openai_compat", "\n".join(
        message
        for _level, message in startup_lines(
            OPENAI, version="0.2.1", ffmpeg="f", ffprobe="p", cache_root="/tmp/c"
        )
    )
    yield "注入给模型的英文短语", "\n".join(SHORT_REASONS.values())

    # ---- 真机样张 ----
    yield "真机 21:00:39 实际发出的正文", REAL_REPORT_2100
    yield "真机报告的修复后还原版", _report(
        VideoConfig(
            max_size_mb=13.0, compress_max_mb=11.7, max_duration_sec=0.0
        ),
        size_mb=5.71,
        duration_sec=11.5,
        probe=_probe(
            path="media_video_20261026210038230_9755.mp4",
            size_mb=5.71,
            duration_sec=11.5,
            ffmpeg=r"D:\ffmpeg\bin\ffmpeg.EXE",
            ffprobe=r"D:\ffmpeg\bin\ffprobe.EXE",
        ),
        plan=classify(
            size_mb=5.71,
            duration_sec=11.5,
            config=VideoConfig(
                max_size_mb=13.0, compress_max_mb=11.7, max_duration_sec=0.0
            ),
            mime_or_suffix=".mp4",
        ),
        provider=ProviderFacts(
            "googlegenai_chat_completion",
            "AlmoonsGemini-1/［抗截断］gemini-3.8flash",
            ["text", "image", "audio", "tool_use"],
        ),
        notices=(
            "compress_max_mb=15.0 超过体积闸门 max_size_mb=13.0，"
            "已按闸门的 90% 收敛为 11.7",
        ),
        outcome=PrepareOutcome(
            True,
            _injection(
                used_path="media_video_20261026210038230_9755.mp4",
                size_mb=5.71,
                duration_sec=11.5,
                action="as-is",
                note="",
            ),
        ),
        elapsed_sec=0.22,
    )


SCENES = list(_scenes())
SCENE_IDS = [name for name, _text in SCENES]


# ===========================================================================
# 反向对照：复刻的守卫必须还能复现真机故障，否则本文件全绿也没有意义
# ===========================================================================


def test_guard_replica_reproduces_the_real_failures():
    """同一份复刻实现必须删掉旧文案 —— 否则下面的绿是假绿。"""
    old_label_line = "判定   : 未超限，直接送"
    assert guard_reason(old_label_line) == "复核模型输出判定字段"

    old_gate_line = (
        "         ② modalities 未声明 → ⚠️ 放行"
        "（无法校验，视频块可能被清洗）"
    )
    assert guard_reason(old_gate_line) == "复核模型输出中文审核评语"

    old_text = "\n".join(
        [
            "🔍 视频链路诊断",
            old_gate_line,
            old_label_line,
            "✅ 在预算内",
        ]
    )
    cleaned, reason = guard_clean(old_text)
    assert reason == "复核模型输出中文审核评语、复核模型输出判定字段"
    assert "判定   :" not in cleaned
    assert "无法校验" not in cleaned
    # 守卫连缩进都会吃掉（真机 21:00:39 的正文里闸门行就是顶格的）
    assert cleaned != old_text


def test_the_old_label_is_what_the_guard_deletes_and_the_new_one_is_not():
    """锁住标签本身：``结论`` 不在守卫的词表里，``判定`` 在。"""
    assert diagnostics.VERDICT_LABEL == "结论   :"
    assert guard_reason(f"{diagnostics.VERDICT_LABEL} 未超限，直接送") == ""
    assert guard_hits("判定   : 未超限，直接送") == ["R_META"]


# ===========================================================================
# 正面断言：逐行不被删 + 整条文本一个字节都不变
# ===========================================================================


@pytest.mark.parametrize(("name", "text"), SCENES, ids=SCENE_IDS)
def test_outbound_text_passes_the_guard(name, text):
    """每一行都不会被判成复核评语；整条文本过一遍守卫后逐字不变。"""
    assert text, name
    dropped = [
        (line, guard_reason(line))
        for line in text.splitlines()
        if line.strip() and guard_reason(line)
    ]
    assert dropped == [], f"{name} 有行会被删: {dropped}"

    assert text == text.strip(), f"{name} 首尾有空白，守卫的 .strip() 会改到它"

    cleaned, reason = guard_clean(text)
    assert reason == "", f"{name} 整条被判成: {reason}"
    assert cleaned == text, f"{name} 守卫改动了正文"


def test_no_outbound_line_matches_even_one_of_the_named_regexes():
    """更严的一条：逐条正则都不许命中（哪怕只命中一半也不留隐患）。"""
    violations: list[str] = []
    for name, text in SCENES:
        for line in text.splitlines():
            if not line.strip():
                continue
            for label, pattern in NAMED_PATTERNS.items():
                if pattern.search(line) or pattern.search(re.sub(r"\s+", " ", line)):
                    violations.append(f"{name} :: {label} :: {line}")
    assert violations == []


#: 这些行型必须真的出现在场景里，否则上面两条断言可能"绿得空洞"
REQUIRED_LINE_KINDS = (
    "🔍 视频链路诊断",
    "结论   :",
    "闸门   : ⓪",
    "① 类型",
    "② modalities 含 audio",
    "② modalities 未声明",
    "② modalities 不含 audio",
    "提醒   :",
    "MIME   :",
    "(推断)",
    "配置提示:",
    "配置修正:",
    "说明:",
    "❌ 准备失败",
    "原因:",
    "分类:",
    "⚠️ 可能超时",
    "✅ 在预算内",
    "✅ 视觉调用成功",
    "— 视觉模型回复节选 —",
    "❌ 视觉调用失败:",
    "用法（任选一种）：",
    "本次没找到视频",
    "该指令仅管理员可用",
    "❌ 文件不存在",
    "❌ 不是文件",
    "❌ 诊断过程异常",
    "[VideoVision]",
)


def test_the_scenes_cover_every_line_kind_we_can_emit():
    """自检：场景必须覆盖到每一种行型，否则守卫断言等于没跑。"""
    joined = "\n".join(text for _name, text in SCENES)
    missing = [needle for needle in REQUIRED_LINE_KINDS if needle not in joined]
    assert missing == [], f"这些行型没被任何场景覆盖: {missing}"


def test_the_real_machine_report_is_now_stable():
    """真机那份正文（守卫删行 + 去缩进之后的产物）现在必须原样通过。"""
    cleaned, reason = guard_clean(REAL_REPORT_2100)
    assert reason == ""
    assert cleaned == REAL_REPORT_2100
    # 真机上消失的正是这一行：修复后的渲染器必须产出它，且它不再被判成复核字段
    rebuilt = dict(SCENES)["真机报告的修复后还原版"]
    assert "结论   : 未超限，直接送" in rebuilt
    assert guard_reason("结论   : 未超限，直接送") == ""
    assert guard_clean(rebuilt) == (rebuilt, "")


# ===========================================================================
# 静态防线：源码里的**字符串字面量**也不许长得像复核字段
# ===========================================================================


def _docstring_ids(tree: ast.AST) -> set[int]:
    ids: set[int] = set()
    for node in ast.walk(tree):
        body = getattr(node, "body", None)
        if not isinstance(body, list) or not body:
            continue
        first = body[0]
        if (
            isinstance(first, ast.Expr)
            and isinstance(first.value, ast.Constant)
            and isinstance(first.value.value, str)
        ):
            ids.add(id(first.value))
    return ids


def _string_literals(path: Path) -> Iterator[tuple[int, str]]:
    """源码里所有字符串字面量（跳过 docstring：它们永远不会被发出去）。"""
    tree = ast.parse(path.read_text(encoding="utf-8"))
    skip = _docstring_ids(tree)
    for node in ast.walk(tree):
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            if id(node) not in skip:
                yield node.lineno, node.value


def test_no_source_string_literal_can_be_mistaken_for_a_review_field():
    """连"将来才拼进文本"的字面量一起防：标签一定是字面量，不会运行时才长出来。

    覆盖 ``core/`` / ``hooks/`` / ``main.py``（也就是所有会产出外发文本的地方，
    含 logger 模板 —— 日志会被贴出去，措辞同样不要长成审核评语）。
    ``_conf_schema.json`` 的 hint 只出现在 WebUI 面板里，不经消息链，故不在范围。
    """
    violations: list[str] = []
    paths = sorted(
        list((REPO / "core").glob("*.py"))
        + list((REPO / "hooks").glob("*.py"))
        + [REPO / "main.py"]
    )
    for path in paths:
        for lineno, value in _string_literals(path):
            for label, pattern in NAMED_PATTERNS.items():
                if pattern.search(value):
                    violations.append(
                        f"{path.relative_to(REPO)}:{lineno} :: {label} :: "
                        f"{value[:80]!r}"
                    )
    assert violations == []
