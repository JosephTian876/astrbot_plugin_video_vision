"""本地 I/O 故障的短原因必须**说真话**：权限 / 磁盘不许被说成"体积超限"（P2-2）。

AUDIT-VV-R9 §4.1 的实测（中文 locale 特有）：

    zh_perm  ("压缩异常: [Errno 13] 拒绝访问。: 'C:\\tmp\\x.mp4'")  -> "it exceeds the size limit"  ← 撒谎
    en_perm  ("压缩异常: [Errno 13] Permission denied: 'C:/tmp/x.mp4'") -> "compression failed"      ← 对

根因是文本回退把中文「拒绝访问」里的「拒绝」当成了「按配置拒绝」（体积/时长闸门）。
后果不是文案难看：模型会去劝用户"把视频压小一点"，而真实原因是本机写不了文件 ——
用户怎么压都没用。

``kind`` 是 ``internal`` / ``other`` 时 ``decide.explain`` 给的是空串（``core/decide.py``
刻意不收这两个键），所以**一定**会掉进这段文本回退。这里把三类故障钉死：

* 权限类 -> 一句"本地文件访问被拒绝"，绝不提体积/时长；
* 磁盘满 -> 一句"本地磁盘满了"，绝不提体积/时长；
* 分类不了 -> 中性文案（"compression failed" / "it could not be processed"）。

**同时**保留真正该说"超限"的那条路（``按配置拒绝`` 前缀 / 「闸门」），
否则就变成了"为了不撒谎而不敢说话"。
"""

from __future__ import annotations

import asyncio

import pytest

from vv_plugin.core import decide
from vv_plugin.core.config import VideoConfig
from vv_plugin.core.injector import PrepareOutcome
from vv_plugin.hooks import llm_request as hooks

#: 权限类：本机读/写被拒（中文 Windows 的 PermissionError 文案是「拒绝访问。」）
PERMISSION_PHRASE = "local file access was denied"
#: 磁盘满
DISK_PHRASE = "the local disk is full"
#: 分类不了时的中性文案（都不许提"大小/时长"）
NEUTRAL_PHRASES = frozenset({"compression failed", "it could not be processed"})

#: 谁都不许用来形容权限/磁盘故障的说法
LIES = ("size limit", "duration", "too large", "太大", "太长")


def _no_lie(text: str) -> None:
    """断言这句话没有把本地 I/O 故障说成"体积/时长超限"。"""
    low = text.lower()
    for lie in LIES:
        assert lie not in low, f"文案在撒谎：{text!r} 含 {lie!r}"


@pytest.mark.parametrize(
    "reason",
    [
        # 中文 locale：PermissionError 的 strerror 是「拒绝访问。」
        r"压缩异常: [Errno 13] 拒绝访问。: 'C:\tmp\x.mp4'",
        "未预期异常: PermissionError: [Errno 13] 拒绝访问。",
        # 英文 locale 的同一件事（旧代码在这条上是"对的"，不许改坏）
        r"压缩异常: [Errno 13] Permission denied: 'C:/tmp/x.mp4'",
        "未预期异常: PermissionError: [WinError 5] Access is denied",
        "压缩失败: [Errno 1] Operation not permitted",
        r"压缩异常: PermissionError: 没有权限写入: 'C:\tmp\out.mp4'",
    ],
)
@pytest.mark.parametrize("kind", ["internal", "other", ""])
def test_permission_failures_are_never_reported_as_a_size_problem(reason, kind):
    """权限失败 -> 一句"本地文件访问被拒绝"，绝不说成体积/时长超限。"""
    text = hooks._short_reason(reason, kind)
    _no_lie(text)
    assert text == PERMISSION_PHRASE, text


@pytest.mark.parametrize(
    "reason",
    [
        "压缩异常: OSError: [Errno 28] 设备上没有空间",
        "未预期异常: OSError: [Errno 28] No space left on device",
        "压缩异常: OSError: [Errno 28] 磁盘空间不足",
        "未预期异常: OSError: [WinError 112] 磁盘空间不足",
    ],
)
@pytest.mark.parametrize("kind", ["internal", "other", ""])
def test_disk_full_is_reported_as_a_disk_problem(reason, kind):
    """磁盘满 -> 说磁盘满（不是"视频太大"，也不是笼统的 "it could not be processed"）。"""
    text = hooks._short_reason(reason, kind)
    _no_lie(text)
    assert text == DISK_PHRASE, text


@pytest.mark.parametrize(
    "reason",
    [
        "压缩异常: RuntimeError: boom",
        "未预期异常: RuntimeError: boom",
        "ffmpeg 内部错误（未知）",
        "压缩异常: ",
    ],
)
@pytest.mark.parametrize("kind", ["internal", "other", ""])
def test_unknown_local_failures_use_a_neutral_phrase(reason, kind):
    """分类不了就用中性文案 —— 猜"太大"是最坏的猜法。"""
    text = hooks._short_reason(reason, kind)
    _no_lie(text)
    assert text in NEUTRAL_PHRASES, text


@pytest.mark.parametrize(
    ("reason", "kind", "expected"),
    [
        # 生产方（injector._blocked_reason）用的前缀：**没有码**时只能说"被策略拒绝"
        # （不猜是体积还是时长 —— 猜错就是`_short_reason` 存在的那个老 bug）
        (
            "按配置拒绝（体积 20.0MB > 闸门 13.0MB）",
            "internal",
            decide.explain("rejected"),
        ),
        (
            "按配置拒绝（时长 7200.0s > 闸门 60.0s）",
            "internal",
            decide.explain("rejected"),
        ),
        # 有码时走码表：体积 / 时长各说各的
        (
            "按配置拒绝（体积 20.0MB > 闸门 13.0MB）",
            "rejected_size",
            decide.explain("too_large"),
        ),
        (
            "按配置拒绝（时长 7200.0s > 闸门 60.0s）",
            "rejected_duration",
            decide.explain("rejected_duration"),
        ),
        # 没有前缀、但「闸门」本身就是体积的证据（不许被"压缩"二字抢走）
        (
            "需要压缩但体积 20.0MB 超过闸门 13.0MB",
            "internal",
            decide.explain("too_large"),
        ),
        (
            "文件超过闸门（13.0MB），需要压缩",
            "internal",
            decide.explain("too_large"),
        ),
    ],
)
def test_real_size_rejections_still_say_they_are_too_large(reason, kind, expected):
    """真正该说"超限"的那条路不许被这次修复带歪。"""
    assert hooks._short_reason(reason, kind) == expected


def test_refusing_to_compress_without_a_duration_stays_a_duration_problem():
    """「拒绝盲压缩」里的「拒绝」不是"按配置拒绝"，归因必须是时长（旧行为不变）。"""
    reason = "无法读取时长，拒绝盲压缩（体积超闸门）"
    assert hooks._short_reason(reason, "internal") == decide.explain("unknown_duration")
    # 生产路径带码（``core/media.py`` 的 ``kind="duration"``），两条都必须落到同一句
    assert hooks._short_reason(reason, "duration") == decide.explain("unknown_duration")


def test_the_new_phrases_pass_the_companion_guard():
    """新文案同样要穿过陪伴插件的「复核评语」守卫（逐字复用它的正则，不另抄一份）。"""
    from test_interop import guard_clean, guard_reason

    for phrase in (PERMISSION_PHRASE, DISK_PHRASE, *sorted(NEUTRAL_PHRASES)):
        assert guard_reason(phrase) == "", (phrase, guard_reason(phrase))
        assert guard_clean(phrase) == (phrase, ""), phrase


def test_the_hook_injects_the_honest_phrase_for_a_permission_failure(monkeypatch):
    """端到端（AUDIT-VV-R9 §4.1 的复现路径）：钩子注入给模型的就是这句话。

    同时锁住"两种读者"的分工：**模型**只看到不含本机路径的短原因；**日志**留着
    原始原因（运维要能看到 `[Errno 13] 拒绝访问。: 'C:\\tmp\\x.mp4'`）。
    """
    from astrbot.api import logger as recording_logger

    raw = r"压缩异常: [Errno 13] 拒绝访问。: 'C:\tmp\x.mp4'"

    def fake_prepare(placeholder, **_kwargs):
        return PrepareOutcome(False, reason=raw, kind="internal")

    class _Part:
        def __init__(self, text: str) -> None:
            self.text = text

    monkeypatch.setattr(hooks, "prepare_video", fake_prepare)
    recording_logger.clear()
    try:
        prepared, consumed, failures = asyncio.run(
            hooks.build_video_blocks(
                [_Part(r"[Video Attachment: name a.mp4, path C:\videos\a.mp4]")],
                config=VideoConfig(provider_mode="gemini_native"),
                cache_root="",
                limiter=None,
            )
        )
        warnings = [
            item
            for item in recording_logger.messages("warning")
            if "视频未注入" in item
        ]
    finally:
        recording_logger.clear()

    assert prepared == [] and consumed == set()
    assert failures == [PERMISSION_PHRASE], failures
    assert "size limit" not in " ".join(failures)
    # 日志里必须留着真实原因（模型看不到路径，运维必须看得到 errno）
    assert len(warnings) == 1, warnings
    assert "拒绝访问" in warnings[0] and "Errno 13" in warnings[0], warnings
