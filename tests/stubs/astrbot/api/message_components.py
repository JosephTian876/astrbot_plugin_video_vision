"""``astrbot.api.message_components`` 桩：``Video`` / ``File``。"""

from __future__ import annotations


class Video:
    """视频组件。"""

    def __init__(self, path: str = "") -> None:
        self.path = path

    async def convert_to_file_path(self) -> str:
        return self.path


class File:
    """文件组件（QQ/NapCat 把视频当文件投递时走这条）。"""

    def __init__(self, name: str = "", path: str = "") -> None:
        self.name = name
        self.path = path

    async def get_file(self) -> str:
        return self.path


__all__ = ["File", "Video"]
