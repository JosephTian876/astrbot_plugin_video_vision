"""``astrbot.core.star.filter.command`` 桩：``GreedyStr``。

只提供类型本身。**贪婪语义不在桩里实现** —— 测试 ``test_main.py`` 直接复刻
框架 ``CommandFilter.init_handler_md`` / ``validate_and_convert_params`` 的
那几行源码来验证参数提取，这样"实现改了但桩没改"不会造成假通过。
"""

from __future__ import annotations


class GreedyStr(str):
    """把剩余所有参数合并成一个字符串（框架的标记类型）。"""


__all__ = ["GreedyStr"]
