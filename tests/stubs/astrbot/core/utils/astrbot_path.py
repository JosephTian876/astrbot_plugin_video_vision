"""``astrbot.core.utils.astrbot_path`` 桩。"""

from __future__ import annotations

import tempfile


def get_astrbot_data_path() -> str:
    """返回一个临时目录（避免测试写到用户主目录）。"""
    return tempfile.gettempdir()


__all__ = ["get_astrbot_data_path"]
