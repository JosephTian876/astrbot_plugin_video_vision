"""``astrbot.api.message_components`` 桩：``Video`` / ``File`` / ``Reply``。

字段照抄 AstrBot 4.29.1 的 ``core/message/components.py``：
``Video(file, url, cover, path)``、``File(name, file, url)``、
``Reply(id, chain, sender_id, sender_nickname, message_str)``。

``convert_to_file_path`` / ``get_file`` 只回路径、不做下载（测试不联网）。引用视频
的数据来源是 ``Reply.chain`` —— 桩里必须保留这个字段，否则"引用视频"相关用例会
用假结构验出假通过。
"""

from __future__ import annotations


class Video:
    """视频组件。"""

    def __init__(
        self, file: str = "", url: str = "", path: str = "", **_: object
    ) -> None:
        self.file = file
        self.url = url
        self.path = path

    async def convert_to_file_path(self) -> str:
        return self.path or self.file


class File:
    """文件组件（QQ/NapCat 把视频当文件投递时走这条）。"""

    def __init__(self, name: str = "", file: str = "", url: str = "", **_: object):
        self.name = name
        self.file = file
        self.url = url

    async def get_file(self) -> str:
        return self.file


class Reply:
    """引用（回复）组件。``chain`` 才是被引用消息的消息段。"""

    def __init__(
        self,
        id: str | int = 0,
        chain: list | None = None,
        sender_id: str | int = 0,
        sender_nickname: str = "",
        message_str: str = "",
        **_: object,
    ) -> None:
        self.id = id
        self.chain = chain if chain is not None else []
        self.sender_id = sender_id
        self.sender_nickname = sender_nickname
        self.message_str = message_str


__all__ = ["File", "Reply", "Video"]
