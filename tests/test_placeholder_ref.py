"""ref 形态占位符：引用视频在本地取不到时，核心只留一个地址。

形态（``astrbot/core/astr_main_agent.py`` 的 ``_append_video_attachment``）：:

    [Video Attachment in quoted message: name a1b2.mp4, ref https://.../a1b2.mp4]

0.3.0 的 ``_VIDEO_RE`` 硬性要求 ``, path ``，所以这种行 100% 静默不注入。
本文件锁两件事：

1. **解析**：ref 行能还原出 name / 地址 / quoted，且存在 ``]``、逗号的边界；
2. **行为不变**：ref 不进 ``iter_placeholders`` 的候选 —— 本 PR 不做物化
   （下载），把 URL 交给 ``prepare_video`` 只会得到「文件不存在」的用户可见
   失败，那是下一个 PR 才该引入的行为变更。
"""

from __future__ import annotations

from types import SimpleNamespace

from vv_plugin.core import placeholder
from vv_plugin.hooks import llm_request as hooks

REF_URL = "https://cdn.example/v/a1b2.mp4?sign=abc&t=1"
REF_LINE = f"[Video Attachment in quoted message: name a1b2.mp4, ref {REF_URL}]"


def test_quoted_ref_placeholder_is_parsed():
    parsed = placeholder.parse_placeholder_text(REF_LINE)

    assert parsed is not None
    assert parsed.name == "a1b2.mp4"
    # 地址复用 path 字段：不新增 dataclass 字段（空字段就是死字段）
    assert parsed.path == REF_URL
    assert parsed.quoted is True
    assert parsed.kind == "ref"
    assert parsed.is_ref_kind is True
    assert parsed.is_file_kind is False


def test_a_bracket_inside_the_ref_url_survives():
    """URL 里的 ``]`` 必须原样保留：贪婪匹配到**行尾**的 ``]``。

    未编码的方括号在 CDN 签名链接里并不罕见。非贪婪的 ``.*?`` 会在第一个
    ``]`` 处截断，得到一个永远打不开的地址（J2b 下载时才会暴露，归因极难）。
    """
    url = "https://cdn.example/v/a[1].mp4?sig=x]y"
    parsed = placeholder.parse_placeholder_text(
        f"[Video Attachment in quoted message: name a[1].mp4, ref {url}]"
    )

    assert parsed is not None
    assert parsed.path == url
    assert parsed.name == "a[1].mp4"


def test_a_comma_inside_the_ref_name_is_kept():
    """文件名里的逗号不能被当成 name 与地址的分隔符。"""
    parsed = placeholder.parse_placeholder_text(
        "[Video Attachment in quoted message: "
        "name 2026-01-01, 会议录屏.mp4, ref https://cdn.example/v/1.mp4]"
    )

    assert parsed is not None
    assert parsed.name == "2026-01-01, 会议录屏.mp4"
    assert parsed.path == "https://cdn.example/v/1.mp4"


def test_path_and_ref_forms_are_distinguished():
    """同一批内容块里两种形态都解析出来，且 ``kind`` 不同。

    同时钉住行为边界：ref **不进注入候选**（本 PR 不物化）。这条让"静默跳过"
    与 0.3.0 完全一致 —— 不会多出一条「文件不存在」的失败提示。
    """
    path_line = (
        r"[Video Attachment in quoted message: name a.mp4, path C:\v\a.mp4]"
    )
    path_parsed = placeholder.parse_placeholder_text(path_line)
    ref_parsed = placeholder.parse_placeholder_text(REF_LINE)

    assert path_parsed is not None and ref_parsed is not None
    assert path_parsed.kind == "video"
    assert ref_parsed.kind == "ref"
    assert path_parsed.is_ref_kind is False
    assert ref_parsed.is_ref_kind is True

    parts = [{"text": path_line}, {"text": REF_LINE}]
    assert [(i, p.kind) for i, p in placeholder.iter_placeholders(parts)] == [
        (0, "video")
    ]


def test_file_attachment_has_no_ref_variant():
    """显式否定：``File Attachment`` **没有** ref 变体。

    核心的 ref 分支只写在 Video 组件上；``_FILE_RE`` 要求 ``, path ``，所以这
    两行解析失败（不会被 ``_VIDEO_REF_RE`` 接走 —— 前缀就不同）。但
    ``is_attachment_line`` 仍认它们：前缀相同，于是这类行不会被当"用户问题"。
    """
    plain = "[File Attachment: name a.mp4, ref https://cdn.example/v/a.mp4]"
    quoted = (
        "[File Attachment in quoted message: name a.mp4, "
        "ref https://cdn.example/v/a.mp4]"
    )

    for line in (plain, quoted):
        assert placeholder.parse_placeholder_text(line) is None
        assert placeholder.is_attachment_line(line) is True


def test_is_attachment_line_accepts_the_ref_form():
    """``is_attachment_line`` 不用改就认 ref 行：前缀与 path 行逐字节相同。

    它的唯一实现是 ``line.strip().startswith(_PREFIXES)``，因此前缀集合里
    **不需要**出现 ``, ref `` 之类的条目（多一份前缀就多一处会漂移的定义）。
    """
    assert placeholder.is_attachment_line(REF_LINE)
    assert placeholder.is_attachment_line(f"   {REF_LINE}\t")
    assert REF_LINE.startswith("[Video Attachment")
    assert placeholder._PREFIXES == ("[Video Attachment", "[File Attachment")


def test_extract_question_drops_the_ref_url():
    """ref URL 不能进"用户问题"：它是第三方 CDN 的签名地址，不是用户说的话。"""
    event = SimpleNamespace(get_message_str=lambda: f"{REF_LINE}\n这视频讲了什么")

    question = hooks._extract_question(event, SimpleNamespace())

    assert question == "这视频讲了什么"
    assert "cdn.example" not in question
