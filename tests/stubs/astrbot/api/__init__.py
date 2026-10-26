"""``astrbot.api`` 桩：只提供插件用的 ``logger``。

logger 是一个**记录器**，把每条日志存进 ``records``，测试据此断言告警文案
（例如"未知配置键"、"取不到 provider"、"响应过长已截断"）。插件在模块级
``from astrbot.api import logger``，拿到的是同一个对象，因此测试直接读
``astrbot.api.logger.records`` 即可。
"""

from __future__ import annotations


class _RecordingLogger:
    """最小 logger：记录 (level, 渲染后的文本)。"""

    def __init__(self) -> None:
        self.records: list[tuple[str, str]] = []

    # -- 内部 --
    def _record(self, level: str, message: object, *args: object) -> None:
        try:
            text = message % args if args else str(message)
        except Exception:  # noqa: BLE001 - 渲染失败也不能影响被测代码
            text = str(message)
        self.records.append((level, text))

    # -- 插件用到的级别 --
    def debug(self, message: object, *args: object, **kwargs: object) -> None:
        self._record("debug", message, *args)

    def info(self, message: object, *args: object, **kwargs: object) -> None:
        self._record("info", message, *args)

    def warning(self, message: object, *args: object, **kwargs: object) -> None:
        self._record("warning", message, *args)

    def error(self, message: object, *args: object, **kwargs: object) -> None:
        self._record("error", message, *args)

    # -- 测试辅助 --
    def clear(self) -> None:
        self.records.clear()

    def messages(self, level: str | None = None) -> list[str]:
        return [
            text
            for lvl, text in self.records
            if level is None or lvl == level
        ]


logger = _RecordingLogger()

__all__ = ["logger"]
