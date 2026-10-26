"""与 AstrBot 解耦的核心逻辑。

本包内所有模块**不导入 AstrBot 框架内部**，唯一例外是 ``injector.py`` 里的
``from astrbot.api import logger`` —— 插件市场（LLM Guard）的日志规范要求记录器
必须且只能从 ``astrbot.api`` 取得，不允许直接用内置 ``logging``（见
``CHANGELOG.md`` 0.1.1 条）。也就是说：除 ``astrbot.api.logger`` 这一个合规例外，
本包不 ``import astrbot.*``（``tests/test_docs.py`` 用 AST 锁住这条，docstring
写错会红），以便日后把视频支持上移到 AstrBot 核心（PR）时可以直接搬迁，无需改写。
"""

from .config import VideoConfig
from .injector import PrepareOutcome, VideoInjection, prepare_video
from .media import (
    CompressCache,
    MediaToolError,
    compress_video,
    file_size_mb,
    guess_video_mime,
    probe_duration,
    read_as_data_uri,
    resolve_ffmpeg_tools,
    suffix_of,
)
from .openai_vision import (
    VisionResult,
    analyze_video,
    analyze_videos,
    build_vision_prompt,
)
from .placeholder import (
    VIDEO_SUFFIXES,
    VideoPlaceholder,
    iter_placeholders,
    parse_placeholder_text,
)

__all__ = [
    "CompressCache",
    "MediaToolError",
    "PrepareOutcome",
    "VIDEO_SUFFIXES",
    "VideoConfig",
    "VideoInjection",
    "VideoPlaceholder",
    "VisionResult",
    "analyze_video",
    "analyze_videos",
    "build_vision_prompt",
    "compress_video",
    "file_size_mb",
    "guess_video_mime",
    "iter_placeholders",
    "parse_placeholder_text",
    "prepare_video",
    "probe_duration",
    "read_as_data_uri",
    "resolve_ffmpeg_tools",
    "suffix_of",
]
