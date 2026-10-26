"""文档与实现一致性（R2B 维度 9 的 9 处）。

对应修复项：B19。断言写得很具体 —— 文档写错一次，这里就会红。
"""

from __future__ import annotations

import json
from pathlib import Path

from vv_plugin.hooks.llm_request import _NEUTRALIZE_TEXT

REPO = Path(__file__).resolve().parents[1]
README = (REPO / "README.md").read_text(encoding="utf-8")
CHANGELOG = (REPO / "CHANGELOG.md").read_text(encoding="utf-8")
METADATA = (REPO / "metadata.yaml").read_text(encoding="utf-8")
SCHEMA = json.loads((REPO / "_conf_schema.json").read_text(encoding="utf-8"))
MEDIA_SOURCE = (REPO / "core" / "media.py").read_text(encoding="utf-8")


def squash(text: str) -> str:
    return "".join(text.split())


def test_version_is_bumped():
    assert "version: 0.2.1" in METADATA
    assert CHANGELOG.startswith("# 更新日志")
    assert "## 0.2.1" in CHANGELOG


def test_changelog_021_lists_the_fixed_items():
    section = CHANGELOG.split("## 0.2.1", 1)[1].split("## 0.2.0", 1)[0]
    for needle in ("缓存", "truncate", "-fs", "GreedyStr", "1e999", "tests/", "18MB"):
        assert needle in section, needle


def test_astrbot_lower_bound_matches_what_the_code_needs():
    """mark_as_temp 自 4.24.1 起，占位符与 modalities 自 4.23.5 起。"""
    assert 'astrbot_version: ">=4.24.1,<5"' in METADATA
    assert "需要 **AstrBot >= 4.26**" not in CHANGELOG
    assert "4.24.1" in CHANGELOG
    assert "4.23.5" in CHANGELOG
    assert "AstrBot **>= 4.24.1**" in README


def test_size_fuse_claim_is_no_longer_absolute():
    assert "不会压出超预算的文件" not in README
    assert "不会压出超预算的文件" not in MEDIA_SOURCE
    assert "-fs" in README
    assert "硬上限" in README


def test_core_zero_dependency_claim_is_qualified():
    assert "core/ 下**不允许**出现 `import astrbot`" not in README
    assert "除 `injector.py` 从 `astrbot.api` 取 `logger`" in README
    assert "astrbot.api" in README


def test_hide_local_path_example_matches_the_real_output():
    assert "[Video: fileseg_xxx.mp4]" not in README
    real = _NEUTRALIZE_TEXT["gemini_native"].replace("{name}", "fileseg_xxx.mp4")
    assert squash(real) in squash(README)


def test_compat_table_describes_the_real_rewrite():
    assert "改写成不含路径的说明" in README


def test_videotest_example_matches_the_new_report():
    for needle in (
        "## `/videotest` 诊断指令",
        "模式   : gemini_native",
        "provider: type=",
        "闸门   :",
        "编码上限:",
        "凭据   : 视觉服务",
        "prompt :",
    ):
        assert needle in README, needle
    assert "l10_probe.mp4" not in README


def test_degradation_warning_promise_is_precise():
    hint = SCHEMA["provider_mode"]["hint"]
    assert "告警一次" in hint
    assert "钩子不再重复告警" in hint
    assert "真的只告警一次" in CHANGELOG


def test_cache_single_artifact_behaviour_is_documented():
    assert "不会进缓存" in SCHEMA["cache_max_mb"]["hint"]
    assert "不会进缓存" in README


def test_videotest_extension_check_is_documented():
    assert "不校验扩展名" in README


def test_truncate_semantics_are_documented():
    assert "按体积预算" in SCHEMA["oversize_action"]["hint"]
    assert "truncate_to" in SCHEMA["oversize_action"]["hint"]
    assert "按体积预算截断" in README or "按体积截断" in README


def test_data_uri_hard_limit_is_documented():
    assert "18MB" in README
    assert "18MB" in SCHEMA["max_size_mb"]["hint"]


def test_requirements_has_no_new_runtime_dependency():
    requirements = (REPO / "requirements.txt").read_text(encoding="utf-8")
    for line in requirements.splitlines():
        stripped = line.strip()
        assert not stripped or stripped.startswith("#"), stripped


def test_schema_hints_mention_the_new_guards():
    assert "墙钟" in SCHEMA["max_compress_seconds"]["hint"]
    assert "1.3-3.2" in SCHEMA["compress_max_mb"]["hint"]
