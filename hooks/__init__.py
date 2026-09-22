"""AstrBot 钩子层。

纯逻辑在 ``core/``；这里只放与 AstrBot 事件系统的粘合代码。
"""

from .llm_request import (
    GEMINI_PROVIDER_TYPES,
    build_video_blocks,
    handle_llm_request,
)

__all__ = [
    "GEMINI_PROVIDER_TYPES",
    "build_video_blocks",
    "handle_llm_request",
]
