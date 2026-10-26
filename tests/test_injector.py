"""注入决策层：0 字节/损坏拒绝、truncate 真截断、data URI 硬上限、契约不抛异常。

对应修复项：A1（缓存联动）、B6、B11、B13、B14、B16。
"""

from __future__ import annotations

import shutil
import tempfile

import pytest

from vv_plugin.core import injector, media
from vv_plugin.core.config import VideoConfig
from vv_plugin.core.injector import PrepareOutcome, prepare_video
from vv_plugin.core.media import MediaToolError, compress_video, probe_duration

# 0.2.1 新增的符号用模块属性取，保证本文件在 0.2.0 上也能被收集
# （从而拿到逐用例的基线对比，而不是收集期 ImportError）。
_estimate_truncate_seconds = getattr(injector, "_estimate_truncate_seconds", None)
MAX_DATA_URI_BYTES = getattr(media, "MAX_DATA_URI_BYTES", None)


def config_for(ffmpeg_dir: str, **kwargs) -> VideoConfig:
    """测试用配置：显式指定 ffmpeg 目录，避免依赖 PATH。"""
    return VideoConfig(ffmpeg_path=ffmpeg_dir, **kwargs)


# ------------------------------------------------------------------ 空 / 损坏


def test_zero_byte_file_is_rejected(tmp_path, make_placeholder, stub_logger):
    """回归：0.2.0 会把 0 字节文件当"未超限"编码成 data URI 并注入。"""
    empty = tmp_path / "empty.mp4"
    empty.write_bytes(b"")

    outcome = prepare_video(
        make_placeholder(str(empty)), config=VideoConfig(), logger=stub_logger
    )

    assert outcome.ok is False
    assert outcome.kind == "empty"
    assert "0 字节" in outcome.reason


def test_corrupt_file_is_rejected(
    media_tools, ffmpeg_dir, tmp_path, make_placeholder, stub_logger
):
    """回归：随机字节命名成 .mp4 时 0.2.0 仍然 ok=True 并把它发给模型。"""
    corrupt = tmp_path / "corrupt.mp4"
    corrupt.write_bytes(b"\x00\x01\x02not a video" * 500)

    outcome = prepare_video(
        make_placeholder(str(corrupt)),
        config=config_for(ffmpeg_dir),
        logger=stub_logger,
    )

    assert outcome.ok is False
    assert outcome.kind == "corrupt"
    assert "损坏" in outcome.reason or "无法解析视频" in outcome.reason


def test_corrupt_file_is_still_allowed_without_ffprobe(
    tmp_path, make_placeholder, stub_logger
):
    """没有 ffprobe 时无法判断时长，不能误伤（保持旧行为：照样送）。"""
    media_file = tmp_path / "x.mp4"
    media_file.write_bytes(b"some bytes")
    cfg = VideoConfig(ffmpeg_path=str(tmp_path / "missing-dir"))
    outcome = prepare_video(
        make_placeholder(str(media_file)),
        config=cfg,
        cache_root=None,
        logger=stub_logger,
    )
    assert outcome.ok is True
    assert outcome.injection.action == "as-is"


# ------------------------------------------------------------------- truncate


def test_estimate_truncate_seconds_formula():
    assert _estimate_truncate_seconds is not None, "0.2.1 才引入按体积估算截断"
    assert _estimate_truncate_seconds(10.0, 20.0, 10.0) == pytest.approx(4.5)
    assert _estimate_truncate_seconds(10.0, 1000.0, 10.0) == 1.0  # 下限 1 秒
    assert _estimate_truncate_seconds(3.0, 1.0, 10.0) == 3.0  # 不超过源时长
    assert _estimate_truncate_seconds(0.0, 10.0, 5.0) == 0.0
    assert _estimate_truncate_seconds(5.0, 0.0, 5.0) == 0.0


def test_truncate_action_really_truncates(
    media_tools, ffmpeg_dir, noisy_video, make_placeholder, stub_logger
):
    """回归：0.2.0 的 truncate 与 compress 行为完全相同（从不截断）。"""
    _, ffprobe = media_tools
    source_duration = probe_duration(noisy_video, ffprobe)
    assert source_duration > 2

    cfg = config_for(
        ffmpeg_dir,
        max_size_mb=0.25,
        compress_max_mb=0.2,
        oversize_action="truncate",
        max_duration_sec=60.0,
        cache_enabled=False,
    )
    outcome = prepare_video(
        make_placeholder(noisy_video),
        config=cfg,
        cache_root=None,
        logger=stub_logger,
    )

    assert outcome.ok is True, outcome.reason
    injection = outcome.injection
    assert injection.action == "truncated"
    assert injection.duration_sec < source_duration * 0.9
    assert "截断到" in injection.note


def test_compress_action_label_follows_real_dash_t(
    media_tools, ffmpeg_dir, noisy_video, make_placeholder, stub_logger
):
    """只超时长（不超体积）时也必须报告"已截断"（0.2.0 报 compressed）。"""
    cfg = config_for(
        ffmpeg_dir,
        max_size_mb=50.0,
        compress_max_mb=20.0,
        oversize_action="compress",
        max_duration_sec=1.5,
        cache_enabled=False,
    )
    outcome = prepare_video(
        make_placeholder(noisy_video),
        config=cfg,
        cache_root=None,
        logger=stub_logger,
    )
    assert outcome.ok is True, outcome.reason
    assert outcome.injection.action == "truncated"
    assert outcome.injection.duration_sec < 2.5


def test_reject_action_does_not_run_ffmpeg(
    media_tools, ffmpeg_dir, noisy_video, make_placeholder, stub_logger
):
    cfg = config_for(
        ffmpeg_dir,
        max_size_mb=0.1,
        compress_max_mb=0.05,
        oversize_action="reject",
    )
    outcome = prepare_video(
        make_placeholder(noisy_video), config=cfg, logger=stub_logger
    )
    assert outcome.ok is False
    assert outcome.kind == "rejected"
    assert "拒绝" in outcome.reason


# --------------------------------------------------------------- ffmpeg 参数


def _capture_run(monkeypatch, placeholder_copy: str):
    captured: dict[str, object] = {}

    def fake_run(args, timeout):
        captured["args"] = list(args)
        captured["timeout"] = timeout
        shutil.copyfile(placeholder_copy, args[-1])

    monkeypatch.setattr(media, "_run_ffmpeg", fake_run)
    return captured


def test_compress_passes_hard_size_cap_and_t(
    media_tools, noisy_video, tmp_path, monkeypatch
):
    """回归：0.2.0 没有 -fs，bufsize 是 2×maxrate。"""
    ffmpeg, ffprobe = media_tools
    captured = _capture_run(monkeypatch, noisy_video)
    output = tmp_path / "out.mp4"

    compress_video(
        noisy_video,
        output,
        max_mb=2.0,
        max_duration_sec=1.5,
        ffmpeg=ffmpeg,
        ffprobe=ffprobe,
        timeout=30,
    )

    args = captured["args"]
    assert isinstance(args, list)
    assert "-fs" in args
    assert int(args[args.index("-fs") + 1]) == int(2.0 * 1024 * 1024)
    assert "-t" in args
    assert args[args.index("-t") + 1] == "1.500"
    assert args[args.index("-bufsize") + 1] == args[args.index("-maxrate") + 1]


def test_compress_without_duration_limit_has_no_dash_t(
    media_tools, noisy_video, tmp_path, monkeypatch
):
    ffmpeg, ffprobe = media_tools
    captured = _capture_run(monkeypatch, noisy_video)
    compress_video(
        noisy_video,
        tmp_path / "out.mp4",
        max_mb=2.0,
        max_duration_sec=0.0,
        ffmpeg=ffmpeg,
        ffprobe=ffprobe,
        timeout=30,
    )
    assert "-t" not in captured["args"]


def test_fs_really_caps_output_size(media_tools, long_video, tmp_path):
    """极小预算 + 30s 源：0.2.0 的 maxrate 会超 1.49 倍，-fs 必须把它压住。"""
    ffmpeg, ffprobe = media_tools
    output = tmp_path / "capped.mp4"
    budget = 0.5

    compress_video(
        long_video,
        output,
        max_mb=budget,
        crf=30,
        preset="ultrafast",
        max_duration_sec=0.0,
        ffmpeg=ffmpeg,
        ffprobe=ffprobe,
        timeout=120,
    )

    size_mb = output.stat().st_size / 1024 / 1024
    assert size_mb <= budget * 1.15, f"体积硬上限失效: {size_mb:.2f}MB"
    # -fs 触顶时 ffmpeg 会提前封口：产物时长明显短于 30s 的源
    assert probe_duration(output, ffprobe) < 25.0


# ------------------------------------------------------------------ data URI


def test_data_uri_has_a_hard_limit(sparse_file):
    """回归：0.2.0 没有任何上限。"""
    big = sparse_file("big.mp4", 14 * 1024 * 1024)
    with pytest.raises(MediaToolError) as excinfo:
        media.read_as_data_uri(big)
    assert excinfo.value.kind == "too_large"
    assert "硬上限" in str(excinfo.value)
    assert MAX_DATA_URI_BYTES == 18 * 1024 * 1024


def test_data_uri_limit_boundary(sparse_file):
    assert MAX_DATA_URI_BYTES == 18 * 1024 * 1024, "0.2.1 才有 18MB 硬上限"
    exact = sparse_file("exact.bin", MAX_DATA_URI_BYTES // 4 * 3)
    uri = media.read_as_data_uri(exact)
    assert uri.startswith("data:")
    assert len(uri.split(",", 1)[1]) == MAX_DATA_URI_BYTES

    one_more = sparse_file("over.bin", MAX_DATA_URI_BYTES // 4 * 3 + 1)
    with pytest.raises(MediaToolError):
        media.read_as_data_uri(one_more)


def test_data_uri_rejects_empty_file(tmp_path):
    empty = tmp_path / "empty.bin"
    empty.write_bytes(b"")
    with pytest.raises(MediaToolError) as excinfo:
        media.read_as_data_uri(empty)
    assert excinfo.value.kind == "empty"


def test_prepare_video_converts_data_uri_limit(
    media_tools, ffmpeg_dir, tiny_video, make_placeholder, monkeypatch, stub_logger
):
    """限流不能变成抛异常：必须转成 ok=False 的可读原因。"""
    monkeypatch.setattr(media, "MAX_DATA_URI_BYTES", 1024)
    outcome = prepare_video(
        make_placeholder(tiny_video),
        config=config_for(ffmpeg_dir),
        cache_root=None,
        logger=stub_logger,
    )
    assert outcome.ok is False
    assert outcome.kind == "too_large"
    assert "硬上限" in outcome.reason


# ------------------------------------------------------------------- 契约


def test_mkdtemp_failure_does_not_escape(
    media_tools, ffmpeg_dir, tiny_video, make_placeholder, monkeypatch, stub_logger
):
    """回归：0.2.0 里 mkdtemp 在 try 之外，OSError 直接穿出 prepare_video。"""
    cfg = config_for(
        ffmpeg_dir,
        max_size_mb=0.001,
        compress_max_mb=0.0005,
        cache_enabled=False,
    )

    def boom(*args, **kwargs):
        raise OSError(28, "No space left on device")

    monkeypatch.setattr(tempfile, "mkdtemp", boom)

    outcome = prepare_video(
        make_placeholder(tiny_video), config=cfg, logger=stub_logger
    )

    assert isinstance(outcome, PrepareOutcome)
    assert outcome.ok is False
    assert outcome.reason
    assert outcome.kind in {"internal", "other"}


def test_prepare_video_never_raises_on_weird_paths(make_placeholder, stub_logger):
    for path in ("", "/no/such/file.mp4", "."):
        outcome = prepare_video(
            make_placeholder(path), config=VideoConfig(), logger=stub_logger
        )
        assert outcome.ok is False
        assert outcome.reason


def test_unknown_ffmpeg_binary_is_reported(
    media_tools, noisy_video, tmp_path, make_placeholder, stub_logger
):
    cfg = VideoConfig(
        ffmpeg_path=str(tmp_path / "no-such-dir"),
        max_size_mb=0.1,
        compress_max_mb=0.05,
    )
    outcome = prepare_video(
        make_placeholder(noisy_video), config=cfg, logger=stub_logger
    )
    assert outcome.ok is False
    assert outcome.kind == "not_found"


# -------------------------------------------------------- ffmpeg 失败分类


def test_timeout_is_killed_and_classified(tmp_path):
    """回归：0.2.0 的 subprocess.run 会阻塞到孙进程结束（实测 11.23s）。"""
    if __import__("os").name != "nt":
        pytest.skip("该回归只在 Windows 的 .cmd 包装器上成立")
    wrapper = tmp_path / "slow.cmd"
    wrapper.write_text("@echo off\r\nping -n 12 127.0.0.1 >nul\r\nexit /b 0\r\n")

    import time

    started = time.monotonic()
    with pytest.raises(MediaToolError) as excinfo:
        media._run_ffmpeg([str(wrapper)], timeout=2.0)
    elapsed = time.monotonic() - started

    assert excinfo.value.kind == "timeout"
    assert elapsed < 8.0, f"超时没有真实生效: {elapsed:.2f}s"


def test_corrupt_input_is_classified_from_stderr(media_tools, tmp_path):
    ffmpeg, _ = media_tools
    corrupt = tmp_path / "bad.mp4"
    corrupt.write_bytes(b"\x00\xff" * 2048)

    with pytest.raises(MediaToolError) as excinfo:
        media._run_ffmpeg(
            [ffmpeg, "-v", "error", "-i", str(corrupt), "-f", "null", "-"],
            timeout=30,
        )
    assert excinfo.value.kind == "corrupt"


def test_missing_binary_is_classified_as_not_found(tmp_path):
    with pytest.raises(MediaToolError) as excinfo:
        media._run_ffmpeg([str(tmp_path / "nope-not-a-binary")], timeout=5)
    assert excinfo.value.kind == "not_found"


def test_will_truncate_matches_ffmpeg_semantics():
    assert media.will_truncate(10.0, 5.0) is True
    assert media.will_truncate(10.0, 10.0) is False
    assert media.will_truncate(10.0, 0.0) is False
    assert media.will_truncate(10.0, -1.0) is False


def test_compress_video_without_ffmpeg_raises_not_found(tmp_path):
    with pytest.raises(MediaToolError) as excinfo:
        compress_video(
            tmp_path / "x.mp4",
            tmp_path / "y.mp4",
            max_mb=1,
            ffmpeg=None,
            ffprobe=None,
        )
    assert excinfo.value.kind == "not_found"


def test_probe_duration_returns_zero_on_garbage(tmp_path):
    garbage = tmp_path / "garbage.mp4"
    garbage.write_bytes(b"nope")
    assert media.probe_duration(garbage, None) == 0.0


def test_compress_video_rejects_unreadable_duration(tmp_path, media_tools):
    ffmpeg, ffprobe = media_tools
    garbage = tmp_path / "garbage.mp4"
    garbage.write_bytes(b"nope")
    with pytest.raises(MediaToolError) as excinfo:
        compress_video(
            garbage,
            tmp_path / "out.mp4",
            max_mb=1,
            ffmpeg=ffmpeg,
            ffprobe=ffprobe,
        )
    assert excinfo.value.kind == "duration"


def test_run_ffmpeg_uses_popen_instead_of_run():
    """守卫：_run_ffmpeg 必须用 Popen + 有界二次回收，否则超时会被孙进程拖住。"""
    import ast
    import inspect
    import textwrap

    source = textwrap.dedent(inspect.getsource(media._run_ffmpeg))
    tree = ast.parse(source)
    function = tree.body[0]
    body = function.body[1:] if ast.get_docstring(function) else function.body
    code = "\n".join(ast.unparse(node) for node in body)

    assert "subprocess.Popen" in code
    assert ".communicate(timeout" in code
    assert "subprocess.run" not in code
    assert "_kill_process_tree" in code

    killer = textwrap.dedent(inspect.getsource(media._kill_process_tree))
    assert "taskkill" in killer
    assert "/T" in killer
