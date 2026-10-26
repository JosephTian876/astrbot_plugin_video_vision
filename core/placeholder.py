"""解析 AstrBot 核心注入的附件占位文本。

核心在 ``astr_main_agent.py`` 里把附件写成一行文本塞进
``ProviderRequest.extra_user_content_parts``。实测发现**同一个视频可能以两种
组件形态到达**，占位文本因此有两种：

    [Video Attachment: name a1b2.mp4, path C:\\...\\a1b2.mp4]
    [Video Attachment in quoted message: name a1b2.mp4, path C:\\...\\a1b2.mp4]

    [File Attachment: name fileseg_xxx.mp4, path C:\\...\\fileseg_xxx.mp4]

第二种是实测踩到的坑：**QQ / NapCat 把用户发的视频当作 ``File`` 组件投递**，
核心于是走 ``isinstance(comp, File)`` 分支产出 ``[File Attachment: ...]``，
而插件只认 ``[Video Attachment: ...]`` 就永远不会触发。

因此这里把两种都解析出来，并用 ``kind`` 区分来源；由上层决定
「File 类型的附件是否按视频处理」（依据扩展名判断）。

**不依赖 AstrBot**，纯字符串处理。
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import Any

from .media import suffix_of

#: 核心生成的四种形态。path 用贪婪匹配到行尾的 `]`，以容忍路径里出现 `]`。
_VIDEO_RE = re.compile(
    r"^\[Video Attachment(?P<quoted> in quoted message)?: "
    r"name (?P<name>.*?), path (?P<path>.*)\]$"
)
_FILE_RE = re.compile(
    r"^\[File Attachment(?P<quoted> in quoted message)?: "
    r"name (?P<name>.*?), path (?P<path>.*)\]$"
)

_PREFIXES = ("[Video Attachment", "[File Attachment")


def is_attachment_line(line: str) -> bool:
    """这一行是否是附件占位符（``_PREFIXES`` 的**唯一**对外入口）。

    给"从用户文本里剔除占位符"这类调用方用（``hooks.llm_request._extract_question``）。
    0.2.1 那里另存了一份逐字节相同的 ``_ATTACHMENT_PREFIXES`` 元组，改一处就会
    漂移（ARCH-REVIEW R6）；现在两份合成一份，本函数就是那个出口。

    Args:
        line: 任意一行文本（前后空白会被忽略）。

    Returns:
        True 表示这一行以附件占位符前缀开头。
    """
    return line.strip().startswith(_PREFIXES)


#: 判定「这个 File 附件其实是视频」用的扩展名集合
VIDEO_SUFFIXES: frozenset[str] = frozenset(
    {
        ".mp4", ".m4v", ".mov", ".webm", ".mkv", ".avi", ".flv",
        ".wmv", ".mpeg", ".mpg", ".3gp", ".ts", ".ogv",
    }
)


@dataclass(frozen=True)
class VideoPlaceholder:
    """从占位文本中还原出的附件引用。"""

    name: str
    path: str
    quoted: bool = False
    kind: str = "video"  # "video" = 来自 Video 组件；"file" = 来自 File 组件

    @property
    def is_file_kind(self) -> bool:
        return self.kind == "file"

    def looks_like_video(self) -> bool:
        """File 类型附件是否长得像视频（按扩展名判断）。"""
        return suffix_of(self.path or self.name) in VIDEO_SUFFIXES


def parse_placeholder_text(text: Any) -> VideoPlaceholder | None:
    """解析单段文本；不是附件占位符则返回 None。"""
    if not isinstance(text, str):
        return None
    stripped = text.strip()
    # 快速排除，避免对每段文本都跑正则
    if not stripped.startswith(_PREFIXES):
        return None

    match = _VIDEO_RE.match(stripped)
    kind = "video"
    if not match:
        match = _FILE_RE.match(stripped)
        kind = "file"
    if not match:
        return None

    path = match.group("path").strip()
    if not path:
        return None
    return VideoPlaceholder(
        name=(match.group("name") or "").strip(),
        path=path,
        quoted=bool(match.group("quoted")),
        kind=kind,
    )


def _part_text(part: Any) -> str | None:
    """从 TextPart 对象或 dict 中取出文本，鸭子类型，避免导入 AstrBot。"""
    text = getattr(part, "text", None)
    if isinstance(text, str):
        return text
    if isinstance(part, Mapping):
        value = part.get("text")
        if isinstance(value, str):
            return value
    return None


def iter_placeholders(
    parts: Iterable[Any] | None,
) -> list[tuple[int, VideoPlaceholder]]:
    """扫描内容块列表，返回 [(下标, 占位符), ...]。

    ``[Video Attachment: ...]`` 与 ``[File Attachment: ...]`` 都算候选，由上层
    再按扩展名 / ``treat_file_as_video`` 过滤（``hooks.llm_request`` 的闸门 0）。
    0.2.1 这里还有个 ``include_file_kind`` 形参，但三个调用点从不传 False，
    那个分支等于死代码（ARCH-REVIEW P1-7），已删除。
    """
    found: list[tuple[int, VideoPlaceholder]] = []
    if not parts:
        return found
    for index, part in enumerate(parts):
        placeholder = parse_placeholder_text(_part_text(part))
        if placeholder is None:
            continue
        found.append((index, placeholder))
    return found
