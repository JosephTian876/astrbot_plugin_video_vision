"""CI 配置本身也是被测对象：yml 语法、pin 的版本、矩阵、不稳点处理。

仓库此前**零 CI**（ARCH-REVIEW §7.1）。门禁最怕两种情况：写错了没人发现
（yml 语法错误只会让 workflow 静默不跑），以及悄悄退化成"绿但没测"。
这里把两件事都钉住：

1. ``.github/workflows/ci.yml`` 能被 YAML 解析，且关键字段齐全
   （pin 的 ruff 版本、2×4 矩阵、concurrency、15 分钟超时、junit artifact、
   ``VV_REQUIRE_FFMPEG=1``）；
2. 本地不稳点的三处修复不许回退（``VV_REQUIRE_FFMPEG`` 语义、conftest 里没有
   开发机私有路径、``_drip`` 不再滴 32 秒）。
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

REPO = Path(__file__).resolve().parents[1]
WORKFLOW_PATH = REPO / ".github" / "workflows" / "ci.yml"
CONFTEST = (REPO / "tests" / "conftest.py").read_text(encoding="utf-8")
PYTEST_INI = (REPO / "pytest.ini").read_text(encoding="utf-8")
INJECTOR_TEST = (REPO / "tests" / "test_injector.py").read_text(encoding="utf-8")
VISION_TEST = (REPO / "tests" / "test_vision.py").read_text(encoding="utf-8")


def _normalize_keys(value):
    """YAML 1.1 会把裸 ``on:`` 解析成布尔 ``True``（PyYAML 的行为）。

    这里把它还原成字符串键 ``"on"``，其余键保持原样 —— 免得整个测试用
    ``workflow[True]`` 这种读不懂的写法。
    """
    assert isinstance(value, dict)
    normalized = {}
    for key, item in value.items():
        normalized["on" if key is True else "off" if key is False else str(key)] = item
    return normalized


@pytest.fixture(scope="module")
def workflow() -> dict:
    assert WORKFLOW_PATH.is_file(), "缺少 .github/workflows/ci.yml"
    return _normalize_keys(yaml.safe_load(WORKFLOW_PATH.read_text(encoding="utf-8")))


def test_workflow_is_valid_yaml_with_the_expected_shape(workflow):
    assert workflow["name"] == "CI"
    assert set(workflow["on"]) == {"push", "pull_request", "workflow_dispatch"}
    assert "jobs" in workflow and set(workflow["jobs"]) == {"lint", "test"}


def test_workflow_cancels_superseded_runs(workflow):
    concurrency = workflow["concurrency"]
    assert "github.ref" in concurrency["group"]
    assert concurrency["cancel-in-progress"] is True


def test_every_job_has_a_fifteen_minute_timeout(workflow):
    for name, job in workflow["jobs"].items():
        assert job["timeout-minutes"] == 15, name


def test_ruff_is_pinned_to_the_astrbot_version(workflow):
    """0.17.x 在本仓库有 30 个 error：不 pin 的门禁等于没有门禁。"""
    steps = workflow["jobs"]["lint"]["steps"]
    install = [step for step in steps if "ruff==" in str(step.get("run", ""))]
    assert len(install) == 1, steps
    assert "ruff==0.15.22" in install[0]["run"]
    checks = [
        step
        for step in steps
        if str(step.get("run", "")).startswith("ruff check")
    ]
    assert len(checks) == 1, steps


def test_test_matrix_covers_two_platforms_and_four_pythons(workflow):
    matrix = workflow["jobs"]["test"]["strategy"]["matrix"]
    assert matrix["os"] == ["ubuntu-latest", "windows-latest"]
    assert matrix["python"] == ["3.10", "3.11", "3.12", "3.13"]
    assert workflow["jobs"]["test"]["strategy"]["fail-fast"] is False


def test_junit_reports_are_uploaded(workflow):
    steps = workflow["jobs"]["test"]["steps"]
    pytest_step = [step for step in steps if "pytest" in str(step.get("run", ""))]
    assert len(pytest_step) == 1, steps
    assert "--junitxml=" in pytest_step[0]["run"]
    upload = [step for step in steps if "upload-artifact" in str(step.get("uses", ""))]
    assert len(upload) == 1, steps
    assert upload[0]["if"] == "always()"
    assert "junit-*.xml" == upload[0]["with"]["path"]


def test_ci_requires_ffmpeg_instead_of_silently_skipping(workflow):
    """ARCH-REVIEW §7.3-6：缺 ffmpeg 时约 40 例会 skip，CI 绿但覆盖腰斩。"""
    assert workflow["env"]["VV_REQUIRE_FFMPEG"] == "1"
    # Linux 用 apt，Windows 用 choco，两端都装
    installs = [
        step
        for step in workflow["jobs"]["test"]["steps"]
        if str(step.get("name", "")).startswith("Install ffmpeg")
    ]
    assert len(installs) == 2, installs
    assert {step["if"] for step in installs} == {
        "runner.os == 'Linux'",
        "runner.os == 'Windows'",
    }


# --------------------------------------------------------------- 本地不稳点


def test_conftest_does_not_hardcode_a_private_ffmpeg_path():
    """回归：``D:\\ffmpeg\\bin`` 曾以 raw 字面量写进 conftest，且**优先于 PATH**。"""
    assert 'r"D:\\ffmpeg' not in CONFTEST
    assert "VV_FFMPEG_DIR" in CONFTEST
    assert "shutil.which" in CONFTEST


def test_conftest_fails_when_ffmpeg_is_required(monkeypatch):
    """``VV_REQUIRE_FFMPEG=1`` 时缺 ffmpeg 必须 fail，不是 skip。"""
    import conftest as conftest_module  # 与 tests/conftest.py 同一个模块

    monkeypatch.setattr(conftest_module, "MEDIA_TOOLS", None)

    monkeypatch.setenv("VV_REQUIRE_FFMPEG", "1")
    assert conftest_module._ffmpeg_is_required() is True
    with pytest.raises(pytest.fail.Exception):
        conftest_module.media_tools.__wrapped__()

    for value in ("0", "", "no"):
        monkeypatch.setenv("VV_REQUIRE_FFMPEG", value)
        assert conftest_module._ffmpeg_is_required() is False, value
    monkeypatch.delenv("VV_REQUIRE_FFMPEG")
    assert conftest_module._ffmpeg_is_required() is False
    # 没设这个开关时保持原来的行为：跳过相关用例（本机没装 ffmpeg 也能跑其余 300+）
    with pytest.raises(pytest.skip.Exception):
        conftest_module.media_tools.__wrapped__()


def test_pytest_ini_is_strict():
    assert "--strict-config" in PYTEST_INI
    assert "--strict-markers" in PYTEST_INI


def test_fs_cap_assertion_uses_the_documented_worst_case():
    """``-fs`` 溢出量与 libavformat 版本有关：断言按文档最坏 1.28× 取 1.30。"""
    assert "budget * 1.30" in INJECTOR_TEST
    assert "budget * 1.15" not in INJECTOR_TEST


def test_drip_endpoint_no_longer_drips_for_32_seconds():
    assert "for _ in range(8):" in VISION_TEST
    assert "for _ in range(64):" not in VISION_TEST


def test_dev_requirements_cover_what_the_suite_needs():
    text = (REPO / "tests" / "requirements-dev.txt").read_text(encoding="utf-8")
    pinned = [
        line.strip()
        for line in text.splitlines()
        if line.strip() and not line.strip().startswith("#")
    ]
    assert pinned, "tests/requirements-dev.txt 不能是空的"
    assert any(line.startswith("pytest") for line in pinned), pinned
    assert any(line.startswith("httpx") for line in pinned), pinned
    assert any(line.lower().startswith("pyyaml") for line in pinned), pinned
    # 运行时依赖仍然是"只有注释"（插件市场硬约束）
    runtime = (REPO / "requirements.txt").read_text(encoding="utf-8")
    for line in runtime.splitlines():
        assert not line.strip() or line.strip().startswith("#")


def test_workflow_file_is_listed_in_the_repo_tree_docs():
    """README 的目录结构要提到 CI（否则下一个维护者不知道门禁在哪）。"""
    readme = (REPO / "README.md").read_text(encoding="utf-8")
    assert ".github/workflows/ci.yml" in readme
    assert "VV_REQUIRE_FFMPEG" in readme
