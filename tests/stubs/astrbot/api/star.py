"""``astrbot.api.star`` 桩：``Star`` / ``Context`` / ``register``。"""

from __future__ import annotations


class Context:  # pragma: no cover - 仅用于类型注解
    """Star 上下文占位。"""

    async def get_using_provider_async(self, umo: object = None) -> object:
        return None


class Star:
    """插件基类：只保存 context。"""

    def __init__(self, context: Context | None = None) -> None:
        self.context = context


def register(*args: object, **kwargs: object):
    """``@register(...)`` 在测试里只做透传。"""

    def decorator(cls):
        return cls

    return decorator


__all__ = ["Context", "Star", "register"]
