"""``astrbot.api.event`` 桩：事件类型与 ``filter`` 装饰器。

装饰器**原样返回函数**，不注册任何东西；测试直接调用
``VideoVisionPlugin.cmd_videotest(...)`` 或复刻框架的参数提取逻辑。
"""

from __future__ import annotations

from typing import Any


class AstrMessageEvent:  # pragma: no cover - 仅用于类型注解
    """事件基类的占位实现。"""


class _Filter:
    def on_llm_request(self, priority: int = 0):  # noqa: D401 - 装饰器工厂
        def decorator(func):
            return func

        return decorator

    def command(self, name: str, alias: set[str] | None = None):
        def decorator(func):
            func._vv_command = name  # 供测试确认注册名
            return func

        return decorator


filter = _Filter()

__all__ = ["AstrMessageEvent", "filter", "Any"]
