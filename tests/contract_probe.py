"""真 AstrBot 运行时契约探针（不是 pytest 用例，由 ``test_contract.py`` 起子进程跑）。

为什么要单独一个文件 + 子进程：``tests/conftest.py`` 把 ``tests/stubs`` 插在
``sys.path[0]``，于是**即使装了真框架**，本进程里的 ``astrbot`` 也永远是桩。
桩里没有 ``astrbot.core.star.register`` 子包，``star_map`` / ``star_handlers_registry``
也不存在 —— 在同一个进程里做"真框架"验收是不可能的。所以：

* 本文件只依赖**真框架**，用装了框架的解释器直接跑；
* ``tests/test_contract.py`` 用 ``sys.executable`` 起子进程跑它：venv 里没装框架
  → 退出码 2 → 该用例 **skip**；装了框架 → 退出码 0 → 该用例通过。

手动跑法::

    <AstrBot python> tests/contract_probe.py

退出码：0 = 全部通过；2 = 没有真框架（调用方据此 skip）；1 = 断言失败。
"""

from __future__ import annotations

import importlib
import os
import sys
import types
import warnings
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
#: 与 ``StarManager.load()`` 的导入名一致（``data.plugins.<目录名>.main``）
MODULE = f"data.plugins.{REPO.name}.main"


def _has_real_framework() -> bool:
    """真框架的标志：``astrbot.core.star.register`` 子包（桩里没有）。

    框架目录不一定在 ``sys.path`` 上（例如 AstrBot 的后端把 ``astrbot`` 放在
    ``<backend>/app`` 下，只有它自己的启动脚本会把它加进去）。所以先用环境变量
    ``VV_ASTRBOT_APP`` 指定那个 ``app`` 目录，再尝试直接 import —— 仓库里**不**
    写任何开发机私有路径（``tests/conftest.py`` 的既有约定）。
    """
    extra = os.environ.get("VV_ASTRBOT_APP", "").strip()
    if extra and extra not in sys.path:
        sys.path.insert(0, extra)
    try:
        import astrbot  # noqa: F401
    except Exception:  # noqa: BLE001
        return False
    return (Path(astrbot.__file__).parent / "core" / "star" / "register").is_dir()


def _import_plugin():
    """以 ``data.plugins.<目录名>`` 的包名导入仓库里的插件（父包 __path__ 指过来）。"""
    for name in ("data", "data.plugins"):
        if name not in sys.modules:
            package = types.ModuleType(name)
            package.__path__ = []  # type: ignore[attr-defined]
            sys.modules[name] = package
    package_name = f"data.plugins.{REPO.name}"
    if package_name not in sys.modules:
        package = types.ModuleType(package_name)
        package.__path__ = [str(REPO)]  # type: ignore[attr-defined]
        sys.modules[package_name] = package
    return importlib.import_module(MODULE)


def main() -> int:
    if not _has_real_framework():
        print("NO_REAL_ASTRBOT")
        return 2

    from astrbot.core.star.star import star_map
    from astrbot.core.star.star_handler import (
        EventType,
        star_handlers_registry,
    )

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        plugin_main = _import_plugin()
    deprecations = [
        f"{item.category.__name__}: {item.message}"
        for item in caught
        if issubclass(item.category, DeprecationWarning)
    ]
    assert deprecations == [], f"导入期出现废弃告警（@register 没删干净？）: {deprecations}"

    entry = star_map.get(MODULE)
    assert entry is not None, "没有装饰器时 Star.__init_subclass__ 应自动登记本模块"
    assert entry.star_cls_type is plugin_main.VideoVisionPlugin, entry.star_cls_type

    handlers = star_handlers_registry.get_handlers_by_module_name(MODULE)
    llm = [item for item in handlers if item.event_type == EventType.OnLLMRequestEvent]
    commands = [
        item for item in handlers if item.event_type == EventType.AdapterMessageEvent
    ]
    assert len(llm) == 1, [(item.event_type, item.handler_name) for item in handlers]
    assert llm[0].extras_configs["priority"] == plugin_main.HOOK_PRIORITY
    assert [item.handler_name for item in commands] == ["cmd_videotest"]

    print(
        "OK star_cls_type=VideoVisionPlugin "
        f"hook_priority={llm[0].extras_configs['priority']} "
        f"commands={[item.handler_name for item in commands]} "
        f"deprecations={deprecations}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
