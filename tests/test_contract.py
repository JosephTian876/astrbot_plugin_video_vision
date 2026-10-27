"""契约：没有 ``@register`` 时，真 AstrBot 仍能识别这个插件。

为什么要起子进程：``tests/conftest.py`` 把 ``tests/stubs`` 插在 ``sys.path[0]``，
**本进程里的 ``astrbot`` 永远是桩**（桩里没有 ``core/star/register`` 子包）。
所以"真框架验收"只能在**另一个进程**里做 —— 见 ``tests/contract_probe.py``。

默认 **skip**：venv 里没装 AstrBot 时，探针以退出码 2 结束，本用例 skip（CI 因此
不会因为缺框架变红）。本地装了框架就跑得到::

    <AstrBot python> -m pytest tests/test_contract.py -q

它锁三件事（AUDIT-030 §B.3）：

1. 导入期**没有** ``DeprecationWarning``（``register_star`` 已废弃）；
2. ``Star.__init_subclass__`` 把类登记进了 ``star_map``（不依赖装饰器）；
3. ``on_llm_request`` 钩子恰好 1 条、priority 仍是 ``HOOK_PRIORITY``，
   ``/videotest`` 仍登记为 ``cmd_videotest``。
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
PROBE = REPO / "tests" / "contract_probe.py"


def test_plugin_is_recognised_without_the_register_decorator(tmp_path):
    assert PROBE.is_file(), PROBE
    # 真框架在 import 期就会按 ASTRBOT_ROOT 建 ``data/`` 目录（t2i 模板等）。
    # 指到临时目录，免得在仓库里留下未跟踪文件。
    env = dict(os.environ)
    env.setdefault("ASTRBOT_ROOT", str(tmp_path / "astrbot_root"))
    completed = subprocess.run(
        [sys.executable, str(PROBE)],
        cwd=str(REPO),
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=180,
        env=env,
    )
    if completed.returncode == 2 and "NO_REAL_ASTRBOT" in completed.stdout:
        pytest.skip(
            "需要真 AstrBot 运行时（tests/stubs 里的桩没有 core/star/register）；"
            "用装了框架的解释器跑才有效"
        )
    assert completed.returncode == 0, (
        f"探针失败（{completed.returncode}）\n"
        f"stdout:\n{completed.stdout}\nstderr:\n{completed.stderr}"
    )
    assert "OK star_cls_type=VideoVisionPlugin" in completed.stdout
