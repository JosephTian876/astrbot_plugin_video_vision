"""配置层：垃圾输入矩阵、上下界、normalized() 矛盾修正、schema 1:1。

对应修复项：B5（数值加固 / OverflowError / 上界 / 显式告警）、B18（未知键）。
"""

from __future__ import annotations

import json
from dataclasses import MISSING, fields
from pathlib import Path
from typing import get_type_hints, get_origin

import pytest

from vv_plugin.core import config as config_mod
from vv_plugin.core.config import (
    DEFAULT_MAX_SIZE_MB,
    VideoConfig,
)

# 常量用 getattr 取：这些上界是 0.2.1 新增的，用模块属性访问可以让本文件
# 在 0.2.0 上也能被收集（从而得到逐用例的基线对比），而不是收集期就 ImportError。
CACHE_MAX_MB_CEILING = getattr(config_mod, "CACHE_MAX_MB_CEILING", 65536)
MAX_COMPRESS_SECONDS_CEILING = getattr(
    config_mod, "MAX_COMPRESS_SECONDS_CEILING", 600.0
)
MAX_SIZE_MB_CEILING = getattr(config_mod, "MAX_SIZE_MB_CEILING", 2000.0)
MAX_VIDEO_COUNT_CEILING = getattr(config_mod, "MAX_VIDEO_COUNT_CEILING", 10)
OPENAI_MAX_TOKENS_CEILING = getattr(config_mod, "OPENAI_MAX_TOKENS_CEILING", 100000)
OPENAI_TIMEOUT_SEC_CEILING = getattr(
    config_mod, "OPENAI_TIMEOUT_SEC_CEILING", 3600.0
)

REPO_ROOT = Path(__file__).resolve().parents[1]


def load_schema() -> dict:
    return json.loads((REPO_ROOT / "_conf_schema.json").read_text(encoding="utf-8"))


# --------------------------------------------------------------- 垃圾输入矩阵

GARBAGE_VALUES = [
    None,
    "",
    "   ",
    "abc",
    "12abc",
    [],
    (),
    {},
    {"a": 1},
    b"bytes",
    object(),
    float("nan"),
    float("inf"),
    "1e999",
]

NUMERIC_FIELDS = (
    "max_size_mb",
    "compress_max_mb",
    "compress_crf",
    "max_duration_sec",
    "max_video_count",
    "max_compress_seconds",
    "cache_max_mb",
    "openai_timeout_sec",
    "openai_max_tokens",
)


@pytest.mark.parametrize("value", GARBAGE_VALUES, ids=repr)
def test_from_mapping_survives_garbage(value):
    """任何垃圾值都不能让加载抛异常，且结果必须落在合法范围内。"""
    raw = {name: value for name in NUMERIC_FIELDS}
    cfg = VideoConfig.from_mapping(raw)
    assert 0.5 <= cfg.max_size_mb <= MAX_SIZE_MB_CEILING
    assert 0.5 <= cfg.compress_max_mb <= cfg.max_size_mb
    assert cfg.compress_crf >= 0
    assert cfg.max_duration_sec >= 0
    assert 0 <= cfg.max_video_count <= MAX_VIDEO_COUNT_CEILING
    assert 1.0 <= cfg.max_compress_seconds <= MAX_COMPRESS_SECONDS_CEILING
    assert 16 <= cfg.cache_max_mb <= CACHE_MAX_MB_CEILING
    assert 10.0 <= cfg.openai_timeout_sec <= OPENAI_TIMEOUT_SEC_CEILING
    assert 64 <= cfg.openai_max_tokens <= OPENAI_MAX_TOKENS_CEILING
    assert isinstance(cfg.max_video_count, int)
    assert isinstance(cfg.cache_max_mb, int)


def test_from_mapping_accepts_non_mapping():
    for raw in (None, "nope", 42, [], ()):
        cfg = VideoConfig.from_mapping(raw)
        assert cfg.max_size_mb == DEFAULT_MAX_SIZE_MB


def test_json_1e999_does_not_break_plugin_loading():
    """回归：0.2.0 里这一行会抛 OverflowError，整个插件加载失败。"""
    raw = json.loads('{"max_video_count": 1e999, "cache_max_mb": 1e999}')
    cfg = VideoConfig.from_mapping(raw)
    assert cfg.max_video_count == MAX_VIDEO_COUNT_CEILING
    assert cfg.cache_max_mb == CACHE_MAX_MB_CEILING


@pytest.mark.parametrize(
    ("field_name", "value", "expected"),
    [
        ("max_size_mb", float("inf"), MAX_SIZE_MB_CEILING),
        ("max_size_mb", "1e999", MAX_SIZE_MB_CEILING),
        ("max_size_mb", 5000, MAX_SIZE_MB_CEILING),
        ("max_size_mb", 0.0, 0.5),
        ("max_video_count", float("inf"), MAX_VIDEO_COUNT_CEILING),
        ("max_video_count", 99, MAX_VIDEO_COUNT_CEILING),
        ("max_video_count", -3, 0),
        ("cache_max_mb", 10**9, CACHE_MAX_MB_CEILING),
        ("cache_max_mb", 1, 16),
        ("max_compress_seconds", 10**6, MAX_COMPRESS_SECONDS_CEILING),
        ("max_compress_seconds", 0, 1.0),
        ("openai_timeout_sec", 10**9, OPENAI_TIMEOUT_SEC_CEILING),
        ("openai_timeout_sec", 1, 10.0),
        ("openai_max_tokens", 10**9, OPENAI_MAX_TOKENS_CEILING),
        ("openai_max_tokens", 1, 64),
    ],
)
def test_numeric_bounds(field_name, value, expected):
    cfg = VideoConfig.from_mapping({field_name: value})
    assert getattr(cfg, field_name) == expected


def test_compress_budget_never_exceeds_gate():
    cfg = VideoConfig.from_mapping({"max_size_mb": 10, "compress_max_mb": 999})
    assert cfg.compress_max_mb <= cfg.max_size_mb


# ------------------------------------------------------------------ normalized


def test_normalized_lowers_compress_below_gate():
    cfg, warnings = VideoConfig.from_mapping(
        {"max_size_mb": 13, "compress_max_mb": 20}
    ).normalized()
    assert cfg.compress_max_mb < 13
    assert cfg.compress_max_mb == pytest.approx(13 * 0.6)
    assert any("已自动下调" in item for item in warnings)


def test_normalized_reports_unsolvable_tiny_gate():
    """0.5MB 闸门下 compress_max_mb 无解：必须显式告警，不能假装修好。"""
    cfg, warnings = VideoConfig(
        max_size_mb=0.5, compress_max_mb=0.5
    ).normalized()
    assert cfg.compress_max_mb == 0.5
    assert any("无法自动修正" in item for item in warnings)


def test_normalized_clamps_crf():
    cfg, warnings = VideoConfig(compress_crf=200).normalized()
    assert cfg.compress_crf == 51
    assert any("51" in item for item in warnings)


def test_normalized_warns_truncate_without_duration_limit():
    _, warnings = VideoConfig(
        oversize_action="truncate", max_duration_sec=0
    ).normalized()
    assert any("truncate" in item and "max_duration_sec" in item for item in warnings)


def test_normalized_warns_cache_smaller_than_compress_budget():
    _, warnings = VideoConfig(
        max_size_mb=30, compress_max_mb=20, cache_max_mb=16
    ).normalized()
    assert any("cache_max_mb" in item for item in warnings)


def test_normalized_is_idempotent():
    cfg, _ = VideoConfig.from_mapping(
        {"max_size_mb": 13, "compress_max_mb": 30}
    ).normalized()
    again, warnings = cfg.normalized()
    assert again == cfg
    assert warnings == []


# ------------------------------------------------------- schema <-> dataclass


def test_schema_keys_match_dataclass_fields():
    schema = load_schema()
    declared = {item.name for item in fields(VideoConfig)}
    assert set(schema) == declared
    assert len(schema) == len(declared)


def test_schema_defaults_match_dataclass_defaults():
    schema = load_schema()
    for item in fields(VideoConfig):
        assert item.name in schema, f"schema 缺少 {item.name}"
        if item.default is not MISSING:
            expected = item.default
        else:
            expected = item.default_factory()
        assert schema[item.name]["default"] == expected, item.name


_SCHEMA_TYPE_TO_PY = {
    "bool": bool,
    "int": int,
    "float": float,
    "string": str,
    "list": list,
}


@pytest.mark.parametrize("key", sorted(load_schema()))
def test_schema_type_matches_annotation(key):
    schema = load_schema()
    hints = get_type_hints(VideoConfig)
    annotation = hints[key]
    origin = get_origin(annotation) or annotation
    expected = _SCHEMA_TYPE_TO_PY[schema[key]["type"]]
    assert origin is expected, f"{key}: schema={schema[key]['type']} 注解={annotation}"


def test_every_schema_key_is_consumed_by_from_mapping():
    """双向 1:1：schema 里任何没被 from_mapping 消费的键都会变成"未知键"。"""
    schema = load_schema()
    raw = {key: value["default"] for key, value in schema.items()}
    cfg, notices = VideoConfig.from_mapping_detailed(raw)
    assert notices == []
    assert cfg == VideoConfig.from_mapping(raw)


# --------------------------------------------------------------- 未知键 / 收敛


def test_unknown_keys_are_reported_but_not_fatal():
    cfg, notices = VideoConfig.from_mapping_detailed(
        {"max_size_mbb": 5, "openai_api_kee": "x"}
    )
    assert cfg.max_size_mb == DEFAULT_MAX_SIZE_MB
    assert len(notices) == 1
    assert "max_size_mbb" in notices[0]
    assert "openai_api_kee" in notices[0]


def test_clamped_values_produce_a_readable_notice():
    raw = json.loads('{"max_video_count": 1e999}')
    cfg, notices = VideoConfig.from_mapping_detailed(raw)
    assert cfg.max_video_count == MAX_VIDEO_COUNT_CEILING
    assert any("max_video_count" in item and "上限" in item for item in notices)


def test_clean_config_has_no_notices():
    _, notices = VideoConfig.from_mapping_detailed(
        {"max_size_mb": 13.0, "compress_max_mb": 8.0}
    )
    assert notices == []


def test_private_helpers_are_not_public_api():
    """``_range_notices`` / ``_unknown_key_notices`` 只是内部实现。"""
    assert config_mod._as_int(float("inf"), 7, maximum=3) == 3
    assert config_mod._as_int(float("-inf"), 7, minimum=1) == 1


# ------------------------------------------------------------- 向后兼容


def test_legacy_013_config_still_loads():
    """0.1.x 的配置（没有 provider_mode 等新键）必须继续可用且自动降级。"""
    legacy = {
        "enable": True,
        "require_gemini_provider": True,
        "max_size_mb": 13.0,
        "compress_max_mb": 8.0,
        "compress_crf": 30,
        "compress_preset": "veryfast",
        "max_duration_sec": 60.0,
        "oversize_action": "compress",
        "max_video_count": 2,
        "handle_quoted_video": True,
        "treat_file_as_video": True,
        "hide_local_path": True,
        "notify_on_failure": True,
        "max_compress_seconds": 45.0,
        "ffmpeg_path": "",
        "cache_enabled": True,
        "cache_max_mb": 512,
        "debug_log": False,
    }
    cfg, notices = VideoConfig.from_mapping_detailed(legacy)
    assert notices == []
    assert cfg.provider_mode == "openai_compat"
    # 视觉服务三项为空 -> 运行时降级为 gemini_native（老配置零改动仍可用）
    assert cfg.effective_provider_mode == "gemini_native"
    assert cfg.max_size_mb == 13.0
    assert cfg.compress_max_mb == 8.0


def test_enum_values_still_fall_back_to_default():
    cfg = VideoConfig.from_mapping(
        {"oversize_action": "delete", "compress_preset": "turbo",
         "provider_mode": "magic"}
    )
    assert cfg.oversize_action == "compress"
    assert cfg.compress_preset == "veryfast"
    assert cfg.provider_mode == "openai_compat"
