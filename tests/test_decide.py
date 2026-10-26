"""``core/decide.py``：判决唯一真相源的测试。

两层：

1. **纯单元**（本文件绝大多数用例）：不读文件、不跑 ffmpeg，覆盖闸门边界、
   ``oversize_action`` 三值、截断估算、corrupt / empty、data URI 上限、无 ffmpeg、
   幂等与不可变；
2. **判决 vs 真实执行**（最后两条）：拿真 ffmpeg 跑一遍 ``prepare_video``，断言
   「结论」行说的就是真实发生的（``action`` / ``kind`` 完全一致）。ARCH-REVIEW
   §6.2-1 把这条列为最该补的测试 —— 本轮重构之后它不再是"两份实现刚好一致"，
   而是"同一份判决"。
"""

from __future__ import annotations

import os
from dataclasses import FrozenInstanceError
from pathlib import Path

import pytest

from vv_plugin.core import decide
from vv_plugin.core.config import VideoConfig
from vv_plugin.core.decide import (
    URI_LIMIT_MB,
    URI_LIMIT_RAW_MB,
    MediaPlan,
    classify,
    describe,
    encoding_limit_note,
    estimate_truncate_seconds,
    explain,
)
from vv_plugin.core.injector import prepare_video
from vv_plugin.core.media import MediaToolError, probe_duration, resolve_ffmpeg_tools

#: 一个"什么闸门都不触发"的基础配置，各用例只改自己关心的那一项。
BASE = VideoConfig(max_size_mb=13.0, max_duration_sec=60.0)


# ------------------------------------------------------------------ 闸门边界


def test_within_both_gates_is_as_is():
    plan = classify(size_mb=1.0, duration_sec=2.0, config=BASE)

    assert plan.action == "as-is"
    assert plan.reason_code == "ok"
    assert plan.kind == "" and plan.blocked is False
    assert plan.needs_ffmpeg is False and plan.cache_eligible is False
    assert describe(plan) == "未超限，直接送"


@pytest.mark.parametrize(
    ("size_mb", "expected"),
    [
        (13.0, "ok"),  # 边界：等于闸门不算超（严格大于）
        (13.01, "oversize"),
        (0.01, "ok"),
    ],
)
def test_size_gate_is_strictly_greater(size_mb, expected):
    plan = classify(size_mb=size_mb, duration_sec=1.0, config=BASE)
    assert plan.reason_code == expected


@pytest.mark.parametrize(
    ("duration_sec", "expected"),
    [
        (60.0, "ok"),  # 边界：等于上限不算超
        (60.1, "overlong"),
        (0.1, "ok"),
    ],
)
def test_duration_gate_is_strictly_greater(duration_sec, expected):
    plan = classify(size_mb=1.0, duration_sec=duration_sec, config=BASE)
    assert plan.reason_code == expected
    assert plan.overlong is (expected == "overlong")


def test_zero_duration_gate_disables_the_check():
    """``max_duration_sec=0`` 是"不限时长"，不是"零秒上限"。"""
    cfg = VideoConfig(max_size_mb=13.0, max_duration_sec=0.0)
    plan = classify(size_mb=1.0, duration_sec=9999.0, config=cfg)
    assert plan.reason_code == "ok"
    assert plan.overlong is False


def test_both_gates_exceeded_reports_both_triggers():
    plan = classify(size_mb=20.0, duration_sec=70.0, config=BASE)
    assert plan.oversize is True and plan.overlong is True
    assert plan.note == "体积 20.0MB > 闸门 13.0MB；时长 70.0s > 上限 60.0s"
    assert describe(plan) == (
        "体积 20.0MB > 闸门 13.0MB；时长 70.0s > 上限 60.0s → 压缩"
    )


# --------------------------------------------------------- oversize_action 三值


def test_compress_action_plans_a_compression():
    plan = classify(size_mb=20.0, duration_sec=2.0, config=BASE)

    assert plan.action == "compress"
    assert plan.reason_code == "oversize"
    assert plan.applies_truncate is False and plan.truncate_to_sec == 0.0
    assert plan.outcome_action == "compressed"
    assert plan.needs_ffmpeg is True and plan.cache_eligible is True
    assert describe(plan) == "体积 20.0MB > 闸门 13.0MB → 压缩"


def test_truncate_action_estimates_a_target_from_the_size_budget():
    # 26MB 压到 13MB 的预算 ≈ 时长减半，留 10% 余量：20s * 0.5 * 0.9 = 9.0s
    cfg = VideoConfig(
        max_size_mb=13.0, max_duration_sec=60.0, oversize_action="truncate"
    )
    plan = classify(size_mb=26.0, duration_sec=20.0, config=cfg)

    assert plan.action == "truncate"
    assert plan.truncate_to_sec == pytest.approx(9.0)
    assert plan.applies_truncate is True
    assert plan.outcome_action == "truncated"
    assert plan.expected_duration_sec == pytest.approx(9.0)


def test_truncate_takes_the_smaller_of_user_limit_and_size_estimate():
    # 用户上限 3s 比体积估算 9s 更狠 -> 取 3s
    cfg = VideoConfig(
        max_size_mb=13.0, max_duration_sec=3.0, oversize_action="truncate"
    )
    plan = classify(size_mb=26.0, duration_sec=20.0, config=cfg)
    assert plan.truncate_to_sec == pytest.approx(3.0)


def test_reject_action_never_needs_ffmpeg_or_cache():
    cfg = VideoConfig(max_size_mb=13.0, oversize_action="reject")
    plan = classify(size_mb=20.0, duration_sec=2.0, config=cfg)

    assert plan.action == "reject" and plan.reason_code == "rejected_size"
    assert plan.kind == "rejected_size" and plan.blocked is True
    assert plan.needs_ffmpeg is False and plan.cache_eligible is False
    assert describe(plan) == "体积 20.0MB > 闸门 13.0MB → 拒绝"


def test_overlong_only_still_truncates_even_with_compress_policy():
    """只超时长时，compress 策略也必须真的传 ``-t``（标签由 applies_truncate 定）。"""
    cfg = VideoConfig(max_size_mb=50.0, max_duration_sec=1.5)
    plan = classify(size_mb=1.0, duration_sec=4.0, config=cfg)

    assert plan.reason_code == "overlong"
    assert plan.truncate_to_sec == pytest.approx(1.5)
    assert plan.applies_truncate is True
    assert plan.outcome_action == "truncated"


# ------------------------------------------------------------------ 截断估算


@pytest.mark.parametrize(
    ("duration", "size_mb", "max_size_mb", "expected"),
    [
        (10.0, 20.0, 10.0, 4.5),  # 线性 + 10% 余量
        (10.0, 1000.0, 10.0, 1.0),  # 下限 1 秒
        (3.0, 1.0, 10.0, 3.0),  # 不超过源时长（不放大）
        (0.0, 10.0, 5.0, 0.0),  # 无时长 -> 不截断
        (5.0, 0.0, 5.0, 0.0),  # 无体积 -> 不截断
        (5.0, 10.0, 0.0, 0.0),  # 无闸门 -> 不截断
    ],
)
def test_estimate_truncate_seconds(duration, size_mb, max_size_mb, expected):
    assert estimate_truncate_seconds(duration, size_mb, max_size_mb) == pytest.approx(
        expected
    )


def test_estimate_is_never_longer_than_the_source():
    for size_mb in (0.1, 1.0, 12.9):
        estimated = estimate_truncate_seconds(3.0, size_mb, 13.0)
        assert 0.0 < estimated <= 3.0


# --------------------------------------------------------------- 空 / 损坏


def test_zero_byte_file_is_an_empty_plan():
    plan = classify(size_mb=0.0, duration_sec=0.0, config=BASE)

    assert plan.action == "reject" and plan.reason_code == "empty"
    assert plan.kind == "empty" and plan.blocked is True
    assert describe(plan) == "体积 0.00MB（空文件）→ 会以 empty 失败"


def test_unreadable_video_is_a_corrupt_plan():
    plan = classify(
        size_mb=2.0, duration_sec=0.0, config=BASE, mime_or_suffix=".mp4"
    )

    assert plan.action == "reject" and plan.reason_code == "corrupt"
    assert plan.kind == "corrupt" and plan.blocked is True
    assert describe(plan) == "无法解析（时长读取失败）→ 会以 corrupt 失败"


def test_corrupt_needs_ffprobe():
    """没有 ffprobe 时无法判断时长，不能误伤（旧行为：照样送）。"""
    plan = classify(
        size_mb=2.0,
        duration_sec=0.0,
        config=BASE,
        mime_or_suffix=".mp4",
        has_ffprobe=False,
    )
    assert plan.reason_code == "ok"
    assert describe(plan) == "未超限，直接送"


def test_corrupt_needs_a_video_container_suffix():
    """显式路径可以是任意文件；读不出时长但后缀不是视频 -> 不判损坏。"""
    for suffix in (".txt", ".gif", "", None):
        plan = classify(
            size_mb=0.01, duration_sec=0.0, config=BASE, mime_or_suffix=suffix
        )
        assert plan.reason_code == "ok", suffix


def test_corrupt_accepts_a_mime_as_well_as_a_suffix():
    as_suffix = classify(
        size_mb=2.0, duration_sec=0.0, config=BASE, mime_or_suffix="mp4"
    )
    as_mime = classify(
        size_mb=2.0, duration_sec=0.0, config=BASE, mime_or_suffix="video/mp4"
    )
    not_video = classify(
        size_mb=2.0, duration_sec=0.0, config=BASE, mime_or_suffix="image/gif"
    )
    assert as_suffix.reason_code == as_mime.reason_code == "corrupt"
    assert not_video.reason_code == "ok"


def test_corrupt_is_reported_before_the_size_gate():
    """损坏优先于超限：告诉用户"文件坏了"比"文件太大"更有用。"""
    plan = classify(
        size_mb=200.0, duration_sec=0.0, config=BASE, mime_or_suffix=".mp4"
    )
    assert plan.reason_code == "corrupt"


# ------------------------------------------------------------- data URI 上限


def test_data_uri_constants_come_from_media():
    assert URI_LIMIT_MB == 18.0
    assert URI_LIMIT_RAW_MB == 13.5


@pytest.mark.parametrize(
    ("raw_mb", "flagged"),
    [(13.4, False), (13.5, False), (13.6, True), (20.0, True)],
)
def test_encoding_limit_note_boundary(raw_mb, flagged):
    note = encoding_limit_note(raw_mb)
    assert bool(note) is flagged
    if flagged:
        assert "too_large" in note and "编码上限 18MB" in note


def test_as_is_still_predicts_too_large_without_changing_the_action():
    """as-is 的编码上限是执行期硬检查：判决只预测（reason_code/encoding_note），
    动作仍是 as-is —— 否则失败文案与 kind 都会变。"""
    cfg = VideoConfig(max_size_mb=200.0, max_duration_sec=60.0)
    plan = classify(size_mb=14.0, duration_sec=2.0, config=cfg)

    assert plan.action == "as-is"
    assert plan.reason_code == "too_large"
    assert plan.blocked is False
    assert "编码后约 18.7MB > 编码上限 18MB" in plan.encoding_note
    assert "实际会以 too_large 失败" in plan.encoding_note
    assert describe(plan).startswith("体积/时长在闸门内 → 直接送；但编码后约")


def test_compressed_target_that_hits_the_limit_is_flagged_in_the_line():
    cfg = VideoConfig(max_size_mb=13.0, compress_max_mb=20.0)
    plan = classify(size_mb=14.0, duration_sec=2.0, config=cfg)

    assert plan.target_note.startswith("；但压缩目标 20.0MB 编码后约 26.7MB")
    assert describe(plan).endswith(plan.target_note)
    assert "→ 压缩" in describe(plan)


def test_reject_does_not_print_a_compress_target_note():
    cfg = VideoConfig(max_size_mb=13.0, compress_max_mb=20.0, oversize_action="reject")
    plan = classify(size_mb=14.0, duration_sec=2.0, config=cfg)
    assert plan.target_note == ""
    assert describe(plan) == "体积 14.0MB > 闸门 13.0MB → 拒绝"


# ------------------------------------------------------------------ 无 ffmpeg


def test_missing_ffmpeg_keeps_the_intended_action_but_blocks():
    """判决仍是"压缩"（结论行因此不变），失败由 reason_code 说明。"""
    plan = classify(size_mb=20.0, duration_sec=2.0, config=BASE, has_ffmpeg=False)

    assert plan.action == "compress"
    assert plan.reason_code == "no_ffmpeg"
    assert plan.kind == "not_found" and plan.blocked is True
    assert plan.needs_ffmpeg is True
    # 结论行与 0.2.1 一致：超限就报策略标签，不在诊断里预告"没装 ffmpeg"
    assert describe(plan) == "体积 20.0MB > 闸门 13.0MB → 压缩"


def test_reject_wins_over_missing_ffmpeg():
    """顺序契约：配置拒绝优先于"没有 ffmpeg"，否则用户会收到一条他不需要的错误。"""
    cfg = VideoConfig(max_size_mb=13.0, oversize_action="reject")
    plan = classify(size_mb=20.0, duration_sec=2.0, config=cfg, has_ffmpeg=False)
    assert plan.reason_code == "rejected_size"


def test_oversize_without_duration_blocks_a_blind_compression():
    cfg = VideoConfig(max_size_mb=13.0)
    plan = classify(
        size_mb=20.0, duration_sec=0.0, config=cfg, mime_or_suffix=".bin"
    )
    assert plan.reason_code == "unknown_duration"
    assert plan.kind == "duration" and plan.blocked is True


# --------------------------------------------------------- 幂等 / 不可变 / 词表


@pytest.mark.parametrize(
    "options",
    [
        {},
        {"max_size_mb": 13.0, "oversize_action": "truncate"},
        {"max_size_mb": 0.5, "oversize_action": "reject"},
        {"max_duration_sec": 1.0, "cache_enabled": False},
    ],
)
def test_classify_is_idempotent(options):
    cfg = VideoConfig(**options)
    first = classify(size_mb=20.0, duration_sec=4.0, config=cfg, mime_or_suffix=".mp4")
    second = classify(size_mb=20.0, duration_sec=4.0, config=cfg, mime_or_suffix=".mp4")
    assert first == second
    assert describe(first) == describe(second)


def test_media_plan_is_frozen_and_self_describing():
    plan = classify(size_mb=1.0, duration_sec=2.0, config=BASE)
    assert isinstance(plan, MediaPlan)
    with pytest.raises(FrozenInstanceError):
        plan.action = "reject"  # type: ignore[misc]
    # 判决自带事实（诊断只吃它，不再另算）
    assert plan.size_mb == 1.0 and plan.duration_sec == 2.0


def test_decide_module_is_pure_zero_io_and_astrbot_free():
    """结构：``decide.py`` 是纯函数模块 —— 无 I/O、无 ffmpeg、不认识 AstrBot。

    「结构上不可能再分叉」的前提就是这个模块**只能**吃事实、吐判决：一旦有人
    在这里 ``open()`` 或 ``import astrbot``，它就会重新依赖运行时状态，测试
    也就无法在纯单元层面锁定判决。用 AST 断言，而不是靠人读。
    """
    import ast

    source = (Path(__file__).resolve().parents[1] / "core" / "decide.py").read_text(
        encoding="utf-8"
    )
    tree = ast.parse(source)

    imported: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            imported.add(node.module or "" if node.level == 0 else f".{node.module}")
    expected_imports = {
        "__future__", "dataclasses", ".config", ".media", ".placeholder",
    }
    assert imported == expected_imports, f"decide.py 的依赖变了: {sorted(imported)}"

    # 没有文件 / 进程 / 网络 / 路径拼接
    attrs = {node.attr for node in ast.walk(tree) if isinstance(node, ast.Attribute)}
    names = {node.id for node in ast.walk(tree) if isinstance(node, ast.Name)}
    forbidden = {
        "open", "system", "popen", "run", "urlopen", "socket", "exists", "stat",
    }
    assert not (attrs & forbidden), f"出现了 I/O 调用: {sorted(attrs & forbidden)}"
    assert not (names & forbidden), f"出现了 I/O 调用: {sorted(names & forbidden)}"
    assert "ffprobe" not in names, "判决不许调外部工具"


def test_judgement_carries_the_facts_it_was_given():
    """判决自带 ``size_mb`` / ``duration_sec``：诊断只渲染它，绝不再探测一遍。"""
    plan = classify(size_mb=21.79, duration_sec=30.0, config=BASE)
    assert (plan.size_mb, plan.duration_sec) == (21.79, 30.0)
    assert plan.oversize is True and plan.overlong is False


def test_kind_vocabulary_is_unchanged():
    """要求：``MediaToolError.KINDS`` 的对外取值集合一个都不许变。

    ``rejected_*`` 三条是**判决期**（配置拒绝）的分类，与执行期的
    ``MediaToolError.KINDS`` 不相交。0.2.1 只有一个笼统的 ``rejected``，
    现在按「体积 / 时长 / 两者」拆开 —— 动机见
    :func:`test_reject_distinguishes_size_from_duration`。
    """
    extra = set(decide.KIND_OF_CODE.values()) - MediaToolError.KINDS
    assert extra == set(decide.REJECT_CODES)
    assert all(code.startswith("rejected") for code in extra)
    assert not (MediaToolError.KINDS & set(decide.REJECT_CODES))


def test_reject_distinguishes_size_from_duration():
    """回归（ARCH-REVIEW §4.1）：仅超时长被拒不能说"超过体积上限"。

    0.2.1 把 oversize 与 overlong 都映射成 ``rejected`` → "it exceeds the size
    limit"。于是"视频太长被拒"告诉模型"它太大了"，模型会去劝用户压小体积 ——
    和 ``ae222e4`` 修掉的那类错误归因一模一样。
    """
    overlong = classify(
        size_mb=1.0,
        duration_sec=10.0,
        config=VideoConfig(
            max_size_mb=1000.0, max_duration_sec=1.0, oversize_action="reject"
        ),
    )
    assert overlong.reason_code == "rejected_duration"
    assert overlong.kind == "rejected_duration"
    assert explain(overlong.kind) == "its duration exceeds the limit"
    assert "size" not in explain(overlong.kind)

    oversize = classify(
        size_mb=20.0,
        duration_sec=1.0,
        config=VideoConfig(
            max_size_mb=13.0, max_duration_sec=0.0, oversize_action="reject"
        ),
    )
    assert oversize.reason_code == "rejected_size"
    assert explain(oversize.kind) == "it exceeds the size limit"

    both = classify(
        size_mb=20.0,
        duration_sec=10.0,
        config=VideoConfig(
            max_size_mb=13.0, max_duration_sec=1.0, oversize_action="reject"
        ),
    )
    assert both.reason_code == "rejected_size_and_duration"
    assert explain(both.kind) == "it exceeds the size and duration limits"

    # 三条短语互不相同 —— 否则"拆开"没有意义
    assert len({explain(code) for code in decide.REJECT_CODES}) == 3
    # 没有码时的兜底文案必须诚实（不猜体积/时长）
    assert explain("rejected") == (
        "it was rejected by the configured size or duration policy"
    )


def test_every_code_has_a_phrase():
    """ARCH-REVIEW §6.2-2：kind 全覆盖，不许静默回落到字符串匹配。

    ``other`` / ``internal`` 是**兜底桶**（不是判决码）：它们的 reason 里常带比
    「无法处理」更具体的信息（如「压缩异常: …」），``hooks`` 的文本回退在那里比
    查表更准，因此 :func:`explain` 对它们返回空串是刻意的（见 ``SHORT_REASONS``）。
    """
    codes = (MediaToolError.KINDS - {"other"}) | {
        "missing",
        "rejected",
        "no_ffmpeg",
        "unknown_duration",
    } | set(decide.REJECT_CODES)
    for code in codes:
        assert explain(code), code


def test_explain_returns_empty_string_for_foreign_reasons():
    assert explain("") == ""
    assert explain("internal") == ""
    assert explain("other") == ""  # 兜底桶：交给文本回退
    assert explain("视觉服务返回 HTTP 500") == ""
    # 0.2.0 的遗留码 ``size`` 已删（全仓无发射点，见 SHORT_REASONS 注释）：
    # 它现在跟任何陌生码一样返回空串，不许悄悄长回来。
    assert "size" not in decide.SHORT_REASONS
    assert explain("size") == ""


def test_plan_codes_resolve_to_the_same_phrase_as_their_kind():
    """判决码与对外 kind 说的是同一句话 —— 同一个概念只有一份文案。"""
    assert explain("no_ffmpeg") == explain("not_found") == "ffmpeg is not available"
    assert (
        explain("unknown_duration")
        == explain("duration")
        == "its duration could not be read"
    )


def test_all_phrases_are_path_free_and_english():
    for phrase in decide.SHORT_REASONS.values():
        assert "\\" not in phrase and ":" not in phrase
        assert phrase.isascii()


# ===================================================== 判决 vs 真实执行（矩阵）


def _decide_and_run(cfg, path, cache_root=None):
    """用同一份事实分别喂 classify 与 prepare_video，返回 (plan, outcome)。"""
    ffmpeg, ffprobe = resolve_ffmpeg_tools(cfg.ffmpeg_path)
    size_mb = os.path.getsize(path) / 1024 / 1024
    duration = probe_duration(path, ffprobe)
    plan = classify(
        size_mb=size_mb,
        duration_sec=duration,
        config=cfg,
        mime_or_suffix=os.path.splitext(path)[1],
        has_ffmpeg=bool(ffmpeg),
        has_ffprobe=bool(ffprobe),
    )
    from vv_plugin.core.placeholder import VideoPlaceholder

    outcome = prepare_video(
        VideoPlaceholder(name=os.path.basename(path), path=path),
        config=cfg,
        cache_root=cache_root,
        debug=False,
    )
    return plan, outcome


@pytest.mark.parametrize(
    ("label", "options"),
    [
        ("as-is", {"max_size_mb": 50.0, "max_duration_sec": 60.0}),
        ("compress", {"max_size_mb": 0.10, "compress_max_mb": 0.08}),
        (
            "truncate-overlong",
            {"max_size_mb": 50.0, "compress_max_mb": 20.0, "max_duration_sec": 1.5},
        ),
        (
            "truncate-by-size",
            {
                "max_size_mb": 0.10,
                "compress_max_mb": 0.08,
                "oversize_action": "truncate",
                "max_duration_sec": 60.0,
            },
        ),
        (
            "reject",
            {
                "max_size_mb": 0.10,
                "compress_max_mb": 0.08,
                "oversize_action": "reject",
            },
        ),
    ],
)
def test_gate_matrix_matches_the_real_execution(
    label, options, media_tools, ffmpeg_dir, noisy_video, stub_logger
):
    """结论行说什么，``prepare_video`` 就必须做什么（ARCH-REVIEW §6.2-1）。"""
    cfg = VideoConfig(
        ffmpeg_path=ffmpeg_dir, cache_enabled=False, **options
    )
    plan, outcome = _decide_and_run(cfg, noisy_video)
    line = describe(plan)

    if plan.blocked:  # 只有 reject 这一档会走到这里
        assert outcome.ok is False, f"{label}: {outcome.reason}"
        assert outcome.kind == plan.kind, f"{label}: {outcome.kind} != {plan.kind}"
        assert "→ 拒绝" in line, line
        return

    assert outcome.ok is True, f"{label}: {outcome.reason}"
    injection = outcome.injection
    assert injection is not None
    assert injection.action == plan.outcome_action, f"{label}: {injection.action}"
    if plan.note:  # 压缩/截断产物的说明必须包含结论行的触发条件
        assert injection.note.startswith(plan.note), f"{label}: {injection.note}"


def test_failure_matrix_matches_the_real_execution(
    media_tools, ffmpeg_dir, tmp_path, stub_logger
):
    """空 / 损坏 / 没装 ffmpeg：结论行的分类必须等于真实 ``kind``。"""
    empty = tmp_path / "empty.mp4"
    empty.write_bytes(b"")
    broken = tmp_path / "broken.mp4"
    broken.write_bytes(b"\x00\x01\x02not a video" * 500)
    # 超过 1KB 闸门的"视频"：走到"需要 ffmpeg"这一档（内容无所谓，反正没 ffmpeg）
    oversize = tmp_path / "oversize.mp4"
    oversize.write_bytes(b"x" * 4096)

    cases = [
        ("empty", VideoConfig(ffmpeg_path=ffmpeg_dir), empty, "empty"),
        ("corrupt", VideoConfig(ffmpeg_path=ffmpeg_dir), broken, "corrupt"),
        (
            "no-ffmpeg",
            VideoConfig(
                ffmpeg_path=str(tmp_path / "missing-dir"),
                max_size_mb=0.001,
                compress_max_mb=0.0005,
            ),
            oversize,
            "not_found",
        ),
    ]

    for label, cfg, path, expected_kind in cases:
        plan, outcome = _decide_and_run(cfg, str(path))
        assert plan.kind == expected_kind, label
        assert plan.blocked is True, label
        assert outcome.ok is False, f"{label}: {outcome.reason}"
        assert outcome.kind == expected_kind, f"{label}: {outcome.kind}"
