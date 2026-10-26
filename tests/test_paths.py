"""「小写后缀」只有一份实现（ARCH-REVIEW R5）。

0.2.1 有四处各自算一遍后缀：

* ``main._suffix_of``（``splitext``）
* ``placeholder.VideoPlaceholder.looks_like_video``（``splitext`` + 回退到 name）
* ``media.guess_video_mime``（``Path(path).suffix``）
* ``main`` 找 File 组件时内联的 ``os.path.splitext``

现在统一走 ``core.media.suffix_of()``。这里锁三件事：

1. ``suffix_of`` 本身的行为（含空/无后缀/大写/``PathLike`` 等边界）；
2. 各调用点算出来的结果与原实现**逐例相等**（含 ``Path().suffix`` 的那点差异，
   已逐例验证不影响任何下游判断 —— 见 ``..._matches_pathlib_except_a_bare_dot``）；
3. 结构上不再有第二份实现（源码里不许再出现 ``splitext``，``main`` 不许再有
   ``_suffix_of``）。

``core/decide._looks_like_video`` 是**另一件事**（吃「后缀或 MIME」两种裸形态，
不是路径），因此它保留自己的实现 —— 它的行为由 ``test_decide.py`` 覆盖。
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

from vv_plugin import core, main
from vv_plugin.core import diagnostics, media, placeholder
from vv_plugin.core.decide import classify
from vv_plugin.core.config import VideoConfig
from vv_plugin.core.media import (
    OUTPUT_MIME,
    VIDEO_MIME_BY_SUFFIX,
    CompressCache,
    guess_video_mime,
    suffix_of,
)
from vv_plugin.core.placeholder import VIDEO_SUFFIXES, VideoPlaceholder

REPO = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize(
    ("path", "expected"),
    [
        ("clip.mp4", ".mp4"),
        ("clip.MP4", ".mp4"),
        ("clip.MoV", ".mov"),
        ("/tmp/a.b/clip.webm", ".webm"),
        (r"D:\UwU\videos\a b c.mkv", ".mkv"),
        ("archive.tar.gz", ".gz"),
        ("no-extension", ""),
        ("", ""),
        (None, ""),
        (".mp4", ""),  # splitext 把前导点当文件名 —— 与 Path(".mp4").suffix 一致
        ("dir.mp4/clip", ""),  # 点出现在目录里不算后缀
        ("clip.中文", ".中文"),
    ],
)
def test_suffix_of(path, expected):
    assert suffix_of(path) == expected


def test_suffix_of_accepts_pathlike():
    assert suffix_of(Path(r"D:\videos\clip.MP4")) == ".mp4"


@pytest.mark.parametrize(
    "name",
    ["clip.mp4", "clip.MOV", "a.b.webm", "no-extension", "clip.", "", ".mp4"],
)
def test_suffix_of_matches_pathlib_except_a_bare_dot(name):
    """``Path(p).suffix.lower()`` 是重构前 ``guess_video_mime`` 的写法。

    唯一差异是 ``"clip."``：``splitext`` 给出 ``"."``、``Path.suffix`` 给出 ``""``。
    两者都不在 ``VIDEO_SUFFIXES`` / ``VIDEO_MIME_BY_SUFFIX`` 里，因此**下游结果
    完全相同** —— 这个用例把这句话钉死，而不是"看起来差不多"。
    """
    splitext_value = suffix_of(name)
    pathlib_value = Path(name).suffix.lower()
    if name == "clip.":
        assert (splitext_value, pathlib_value) == (".", "")
        assert splitext_value not in VIDEO_MIME_BY_SUFFIX
        assert pathlib_value not in VIDEO_MIME_BY_SUFFIX
        assert splitext_value not in VIDEO_SUFFIXES
        assert pathlib_value not in VIDEO_SUFFIXES
        return
    assert splitext_value == pathlib_value


@pytest.mark.parametrize("suffix", sorted(VIDEO_SUFFIXES))
def test_guess_video_mime_maps_every_known_suffix(suffix):
    assert guess_video_mime(f"clip{suffix.upper()}") == VIDEO_MIME_BY_SUFFIX[suffix]


def test_guess_video_mime_falls_back_for_unknown_suffixes():
    for name in ("anim.gif", "notes.txt", "no-extension", ""):
        assert guess_video_mime(name) == OUTPUT_MIME


@pytest.mark.parametrize(
    ("path", "name", "expected"),
    [
        ("clip.mp4", "ignored", True),
        ("", "clip.MKV", True),  # 路径为空时回落到 name（原来的行为）
        ("", "notes.txt", False),
        ("", "", False),
        ("no-extension", "", False),
        ("clip.gif", "", False),  # gif 不是视频容器
    ],
)
def test_placeholder_looks_like_video_agrees_with_suffix_of(path, name, expected):
    placeholder_value = VideoPlaceholder(name=name, path=path).looks_like_video()
    suffix_value = suffix_of(path or name) in VIDEO_SUFFIXES
    assert placeholder_value is expected
    assert placeholder_value is suffix_value


@pytest.mark.parametrize("ext", [".mp4", ".MP4", ".gif", ".txt", ""])
def test_diagnose_suffix_helpers_agree_with_suffix_of(ext):
    """两处后缀判断（扩展名警示 / MIME 行）与 ``suffix_of`` 同源。

    两个 helper 现在住在 ``core/diagnostics.py``（本轮从 ``main`` 搬出），
    断言不变：后缀已知时没有警示、MIME 行用真实探测值；未知时警示 + ``(推断)``。
    """
    path = f"clip{ext}"
    expected_added = suffix_of(path)
    known = expected_added in VIDEO_SUFFIXES

    note = diagnostics.container_note(path)
    mime_line = diagnostics.mime_line(path, "as-is")

    expected = (
        f"MIME   : {guess_video_mime(path)}"
        if known
        else f"MIME   : {OUTPUT_MIME} (推断)"
    )
    assert (note == "") is known
    assert mime_line == expected


def test_decide_still_rejects_a_corrupt_video_container_after_the_dedup():
    """回归：后缀判断搬家后，corrupt 判决不受影响（大小写/裸扩展名都认）。"""
    base = VideoConfig(max_size_mb=13.0, max_duration_sec=60.0)
    for value in (".mp4", ".MP4", "mp4", "video/mp4", "VIDEO/MP4"):
        plan = classify(
            size_mb=2.0, duration_sec=0.0, config=base, mime_or_suffix=value
        )
        assert plan.reason_code == "corrupt", value


def test_suffix_logic_has_exactly_one_source_file():
    """结构断言：``splitext`` 只允许出现在 ``core/media.py::suffix_of`` 里。

    用 AST 而不是字符串计数 —— 文档里提到 ``splitext`` 不该算实现（本轮就踩过）。
    ``core/diagnostics.py`` 也是新增的调用点，一并纳入检查。
    """
    offenders = []
    for module in (main, placeholder, diagnostics):
        source = Path(module.__file__).read_text(encoding="utf-8")
        if "splitext" in source:
            offenders.append(Path(module.__file__).name)
    assert offenders == [], f"这些文件还在自己算后缀: {offenders}"
    assert not hasattr(main, "_suffix_of"), "main 的本地后缀实现应已删除"

    tree = ast.parse((REPO / "core" / "media.py").read_text(encoding="utf-8"))
    calls = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "splitext"
    ]
    assert len(calls) == 1, f"splitext 出现 {len(calls)} 次"
    owners = [
        node.name
        for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef)
        and any(child is calls[0] for child in ast.walk(node))
    ]
    assert owners == ["suffix_of"], f"splitext 出现在 {owners}"


def test_dead_code_from_arch_review_p1_7_stays_removed():
    """ARCH-REVIEW P1-7 的死代码：删掉的**不许长回来**。

    同族的 ``main._suffix_of`` 上一轮已删（守卫在
    ``test_suffix_logic_has_exactly_one_source_file``）；其余几项拖到 0.2.1
    收尾才做，正是 VERIFY2 第 10 项判 FAIL 的原因。这里用 ``hasattr`` /
    ``inspect.signature`` 把它们钉住，防止下一次重构"顺手"再导出一遍。
    """
    import inspect

    # a/b: probe_video + VideoInfo（函数与 dataclass，含 core 包的 re-export）
    assert not hasattr(media, "probe_video"), "media.probe_video 应已删除"
    assert not hasattr(media, "VideoInfo"), "media.VideoInfo 应已删除"
    assert not hasattr(core, "probe_video"), "core.probe_video 不该再导出"
    assert not hasattr(core, "VideoInfo"), "core.VideoInfo 不该再导出"
    assert "probe_video" not in core.__all__ and "VideoInfo" not in core.__all__
    # c: CompressCache.stamp（全仓零调用的时间戳工具）
    assert not hasattr(CompressCache, "stamp"), "CompressCache.stamp 应已删除"
    # d: iter_placeholders 的 include_file_kind 形参（没有调用方传 False）
    assert "include_file_kind" not in inspect.signature(
        placeholder.iter_placeholders
    ).parameters
