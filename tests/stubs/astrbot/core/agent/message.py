"""``astrbot.core.agent.message`` 桩：``TextPart`` / ``AudioURLPart``。

只保留插件用到的语义：``mark_as_temp()`` 返回自身并打上 ``_no_save``，
``model_dump_for_context()`` 输出带 ``_no_save`` 的字典。
"""

from __future__ import annotations

from typing import Any


class _Part:
    _no_save: bool = False

    def mark_as_temp(self):
        object.__setattr__(self, "_no_save", True)
        return self


class TextPart(_Part):
    def __init__(self, text: str = "") -> None:
        self.text = text
        self._no_save = False

    def model_dump_for_context(self) -> dict[str, Any]:
        data: dict[str, Any] = {"type": "text", "text": self.text}
        if self._no_save:
            data["_no_save"] = True
        return data


class AudioURLPart(_Part):
    class AudioURL:
        def __init__(self, url: str = "") -> None:
            self.url = url

        def model_dump(self) -> dict[str, Any]:
            return {"url": self.url}

    def __init__(self, audio_url: AudioURL | None = None) -> None:
        self.audio_url = audio_url or self.AudioURL()
        self._no_save = False

    @property
    def url(self) -> str:
        return self.audio_url.url

    def model_dump_for_context(self) -> dict[str, Any]:
        data: dict[str, Any] = {
            "type": "audio_url",
            "audio_url": {"url": self.audio_url.url},
        }
        if self._no_save:
            data["_no_save"] = True
        return data


__all__ = ["AudioURLPart", "TextPart"]
