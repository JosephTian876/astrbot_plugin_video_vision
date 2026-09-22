"""与 AstrBot 解耦的核心逻辑。

本包内所有模块**不得**导入 astrbot.*，以便日后把视频支持上移到 AstrBot 核心
（PR）时可以直接搬迁，无需改写。
"""

from .config import VideoConfig
from .injector import PrepareOutcome, VideoInjection, prepare_video
from .media import (
    CompressCache,
    MediaToolError,
    VideoInfo,
    compress_video,
    file_size_mb,
    guess_video_mime,
    probe_duration,
    probe_video,
    read_as_data_uri,
    resolve_ffmpeg_tools,
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
    "VideoInfo",
    "VideoInjection",
    "VideoPlaceholder",
    "compress_video",
    "file_size_mb",
    "guess_video_mime",
    "iter_placeholders",
    "parse_placeholder_text",
    "prepare_video",
    "probe_duration",
    "probe_video",
    "read_as_data_uri",
    "resolve_ffmpeg_tools",
]
