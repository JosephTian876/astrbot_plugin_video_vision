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
#: 0.3.0 新增：ffmpeg 并发限流（AUDIT-030 §C.1）
FFMPEG_CONCURRENCY_CEILING = getattr(config_mod, "FFMPEG_CONCURRENCY_CEILING", 8)
QUEUE_TIMEOUT_CEILING = getattr(config_mod, "QUEUE_TIMEOUT_CEILING", 600.0)
#: 0.3.0 新增：视觉结果缓存（AUDIT-030 §C.2）
VISION_CACHE_TTL_CEILING = getattr(config_mod, "VISION_CACHE_TTL_CEILING", 86400.0)
VISION_CACHE_ENTRIES_CEILING = getattr(config_mod, "VISION_CACHE_ENTRIES_CEILING", 1024)

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
    "ffmpeg_max_concurrency",
    "ffmpeg_queue_timeout_sec",
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
    assert 0 <= cfg.ffmpeg_max_concurrency <= FFMPEG_CONCURRENCY_CEILING
    assert 1.0 <= cfg.ffmpeg_queue_timeout_sec <= QUEUE_TIMEOUT_CEILING
    assert isinstance(cfg.max_video_count, int)
    assert isinstance(cfg.cache_max_mb, int)
    assert isinstance(cfg.ffmpeg_max_concurrency, int)
    assert isinstance(cfg.ffmpeg_queue_timeout_sec, float)


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
        # ---- 0.3.0：ffmpeg 并发限流 ----
        ("ffmpeg_max_concurrency", 99, FFMPEG_CONCURRENCY_CEILING),
        ("ffmpeg_max_concurrency", float("inf"), FFMPEG_CONCURRENCY_CEILING),
        ("ffmpeg_max_concurrency", -3, 0),  # 0 = 关闭限流
        ("ffmpeg_max_concurrency", 0, 0),
        ("ffmpeg_queue_timeout_sec", 10**9, QUEUE_TIMEOUT_CEILING),
        ("ffmpeg_queue_timeout_sec", 0, 1.0),
    ],
)
def test_numeric_bounds(field_name, value, expected):
    cfg = VideoConfig.from_mapping({field_name: value})
    assert getattr(cfg, field_name) == expected


def test_compress_budget_never_exceeds_gate():
    cfg = VideoConfig.from_mapping({"max_size_mb": 10, "compress_max_mb": 999})
    assert cfg.compress_max_mb <= cfg.max_size_mb


# -------------------------------------------- 0.3.0 新增键：边界 / 垃圾值


#: 0.3.0 的三个 ffmpeg 限流键与其 (默认值, 下界, 上界)
NEW_CONCURRENCY_KEYS = {
    "ffmpeg_max_concurrency": (2, 0, FFMPEG_CONCURRENCY_CEILING),
    "ffmpeg_queue_timeout_sec": (30.0, 1.0, QUEUE_TIMEOUT_CEILING),
}

#: 明确"取不到数字"的垃圾值：必须**回落默认值**（不是收敛到边界）
NON_NUMERIC_GARBAGE = [
    None, "", "   ", "abc", [], (), {}, {"a": 1}, b"bytes", object(),
    float("nan"),
]


def test_new_keys_garbage_falls_back_to_the_default():
    """垃圾值一律回落默认（``2`` / ``30.0``），绝不抛异常。"""
    for key, (default, _low, _high) in NEW_CONCURRENCY_KEYS.items():
        for value in NON_NUMERIC_GARBAGE:
            cfg = VideoConfig.from_mapping({key: value})
            assert getattr(cfg, key) == default, (key, value)


def test_new_keys_out_of_range_is_clamped():
    """越界值收敛到边界（``_NUMERIC_BOUNDS`` 里必须有这几条）。"""
    for key, (_default, low, high) in NEW_CONCURRENCY_KEYS.items():
        for value in GARBAGE_VALUES:
            cfg = VideoConfig.from_mapping({key: value})
            assert low <= getattr(cfg, key) <= high, (key, value)
    bounds = config_mod._NUMERIC_BOUNDS
    assert bounds["ffmpeg_max_concurrency"] == (0, FFMPEG_CONCURRENCY_CEILING)
    assert bounds["ffmpeg_queue_timeout_sec"] == (1.0, QUEUE_TIMEOUT_CEILING)


def test_new_keys_clamping_produces_one_readable_notice():
    """与既有 ``_range_notices`` 风格一致：一条，写出原值与边界。"""
    cfg, notices = VideoConfig.from_mapping_detailed({"ffmpeg_max_concurrency": 99})
    assert cfg.ffmpeg_max_concurrency == FFMPEG_CONCURRENCY_CEILING
    hits = [item for item in notices if "ffmpeg_max_concurrency" in item]
    assert hits == [
        f"ffmpeg_max_concurrency=99 高于上限 {FFMPEG_CONCURRENCY_CEILING:g}，"
        f"已收敛为 {FFMPEG_CONCURRENCY_CEILING:g}"
    ]


def test_ffmpeg_singleflight_is_a_bool_that_defaults_to_true():
    """``ffmpeg_singleflight`` 默认 true；关闭必须是显式的 false。"""
    assert VideoConfig().ffmpeg_singleflight is True
    assert VideoConfig.from_mapping({}).ffmpeg_singleflight is True
    for value in (False, 0, "false", "off", "no", "否"):
        cfg = VideoConfig.from_mapping({"ffmpeg_singleflight": value})
        assert cfg.ffmpeg_singleflight is False, value
    # 非字符串/非数值类型 -> 回落默认 true（``_as_bool`` 对 ``enable`` 等键同源）
    for value in (None, [], {}, object()):
        assert VideoConfig.from_mapping(
            {"ffmpeg_singleflight": value}
        ).ffmpeg_singleflight is True
    # 字符串按 ``_as_bool`` 的词表判定：不在「开」词表里的一律算假，
    # 与 ``enable`` / ``cache_enabled`` 等既有布尔键完全一致。
    for value in ("", "abc", "0", "false"):
        assert VideoConfig.from_mapping(
            {"ffmpeg_singleflight": value}
        ).ffmpeg_singleflight is False


def test_queue_budget_warning_fires_only_when_it_really_eats_the_budget():
    """排队 + 压缩吃掉视觉超时预算的一半以上时必须显式告警（C.4-6）。

    默认值 30 + 45 = 75 < 180/2 的 90，**不许**无端告警。
    """
    default_cfg = VideoConfig(
        provider_mode="openai_compat",
        openai_base_url="https://example.invalid/v1",
        openai_api_key="sk-x",
    )
    _, warnings = default_cfg.normalized()
    assert warnings == []

    tight = VideoConfig(
        provider_mode="openai_compat",
        openai_base_url="https://example.invalid/v1",
        openai_api_key="sk-x",
        ffmpeg_queue_timeout_sec=60.0,
        max_compress_seconds=45.0,
        openai_timeout_sec=180.0,
    )
    _, tight_warnings = tight.normalized()
    assert any("ffmpeg_queue_timeout_sec" in item for item in tight_warnings), (
        tight_warnings
    )
    assert any("openai_timeout_sec" in item for item in tight_warnings)

    # gemini_native 不调用视觉服务：这条告警不适用
    native = VideoConfig(
        provider_mode="gemini_native",
        ffmpeg_queue_timeout_sec=60.0,
        max_compress_seconds=45.0,
        openai_timeout_sec=180.0,
    )
    _, native_warnings = native.normalized()
    assert not any("openai_timeout_sec" in item for item in native_warnings)


# ------------------------------------------------------------------ normalized


def test_normalized_lowers_compress_below_gate():
    """13/15 这组自相矛盾的配置：目标贴着闸门下方（90%），不是砍到 60%。"""
    cfg, notices = VideoConfig.from_mapping_detailed(
        {"max_size_mb": 13, "compress_max_mb": 15}
    )
    assert cfg.compress_max_mb < 13
    assert cfg.compress_max_mb == pytest.approx(11.7)
    # 单趟归一：这一步已经出最终值，normalized() 不再二次改动
    fixed, warnings = cfg.normalized()
    assert fixed == cfg
    assert warnings == []
    assert any("11.7" in item for item in notices)


def test_compress_budget_is_resolved_in_one_pass_with_one_notice():
    """回归（LIVE-TEST §10.2①）：15.0 配 13.0 闸门曾打出**两个**"最终值"。

    旧实现：`from_mapping_detailed` 先按闸门夹成 13.0（提示一条），
    `normalized()` 再降到 11.7（第二条提示）—— 同一个旋钮出现三个数字，
    生效值等于哪一条提示都没写。现在一次算完、只发一条，原值与最终值同句。
    """
    raw = {"max_size_mb": 13.0, "compress_max_mb": 15.0}
    cfg, notices = VideoConfig.from_mapping_detailed(raw)
    fixed, warnings = cfg.normalized()

    assert cfg.compress_max_mb == pytest.approx(11.7)
    assert fixed == cfg and warnings == []

    budget_notices = [item for item in notices if "compress_max_mb" in item]
    assert len(budget_notices) == 1, notices
    assert budget_notices[0] == (
        "compress_max_mb=15.0 超过体积闸门 max_size_mb=13.0，"
        "已按闸门的 90% 收敛为 11.7"
    )
    # 一个字段只出现在一条消息里（另一类消息里不许再提它）
    assert sum("compress_max_mb" in item for item in notices) == 1
    assert warnings == []


def test_resolving_the_compress_budget_is_idempotent():
    """幂等性由构造保证：``min(用户值, 闸门×0.9)`` 再喂回去是同一个值。"""
    for gate in (13.0, 0.6, 1.0, 2000.0):
        for asked in (0.1, 0.5, 5.0, 8.0, 12.0, 13.0, 15.0, 5000.0):
            cfg, _ = VideoConfig.from_mapping_detailed(
                {"max_size_mb": gate, "compress_max_mb": asked}
            )
            again, warnings = cfg.normalized()
            assert again == cfg, (gate, asked, cfg)
            # 压缩预算这条不变量必须已经满足：不会再"自动下调"或"不会自动修正"
            assert not [
                item
                for item in warnings
                if "已自动下调" in item or "不会自动修正" in item
            ], (gate, asked, warnings)
            # 再走一遍 from_mapping 也必须零提示（同一个值进、同一个值出）
            recycled, notices = VideoConfig.from_mapping_detailed(
                {"max_size_mb": cfg.max_size_mb, "compress_max_mb": cfg.compress_max_mb}
            )
            assert recycled.compress_max_mb == pytest.approx(cfg.compress_max_mb)
            assert [item for item in notices if "compress_max_mb" in item] == []


def test_compress_budget_below_the_lower_bound_is_reported_once():
    cfg, notices = VideoConfig.from_mapping_detailed(
        {"max_size_mb": 13.0, "compress_max_mb": 0.1}
    )
    assert cfg.compress_max_mb == pytest.approx(config_mod.MIN_COMPRESS_MB)
    budget_notices = [item for item in notices if "compress_max_mb" in item]
    assert budget_notices == [
        "compress_max_mb=0.1 低于下限 0.5，已收敛为 0.5"
    ]


def test_normalized_tiny_gate_stays_solvable_and_under_the_gate():
    """1.0MB 闸门：0.9 仍在 [MIN_COMPRESS_MB, 闸门) 内，不能越界也不能死循环。"""
    cfg, warnings = VideoConfig(
        max_size_mb=1.0, compress_max_mb=2.0
    ).normalized()
    assert cfg.compress_max_mb < 1.0
    assert cfg.compress_max_mb >= getattr(config_mod, "MIN_COMPRESS_MB", 0.5)
    assert any("已自动下调" in item for item in warnings)
    again, again_warnings = cfg.normalized()
    assert again == cfg
    assert again_warnings == []


def test_normalized_reports_unsolvable_tiny_gate():
    """0.5MB 闸门下 compress_max_mb 无解：必须显式告警，不能假装修好。

    措辞是「插件不会自动修正」而**不是**「无法自动修正」：后者会被陪伴插件的
    复核评语守卫判成「无法…修正」那种审核评语（`tests/test_interop.py`）。
    """
    cfg, warnings = VideoConfig(
        max_size_mb=0.5, compress_max_mb=0.5
    ).normalized()
    assert cfg.compress_max_mb == 0.5
    assert any("不会自动修正" in item for item in warnings)


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
    #: ``text`` 是 AstrBot 对"长字符串"的类型名（面板渲染成可拖拽的多行 textarea），
    #: 读进来仍然是 ``str``（``work/astrbot-upstream/docs/zh/dev/star/guides/
    #: plugin-config.md``：type 支持 string / text / int / float / …）。
    "text": str,
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


def test_api_key_field_is_masked_in_the_panel():
    """回归：插件在日志里给密钥打码，面板却明文显示（自相矛盾）。

    AstrBot 的 ``secret`` 只遮**显示**、不改值，且只对字符串字段生效
    （``work/astrbot-upstream/docs/zh/dev/star/guides/plugin-config.md``）。
    这里同时保证"只有字符串字段敢设 secret"，免得哪天设到数值字段上静默失效。
    """
    schema = load_schema()
    assert schema["openai_api_key"]["secret"] is True
    for key, field_schema in schema.items():
        if field_schema.get("secret"):
            assert field_schema["type"] in {"string", "text", "list"}, key


def test_long_prompt_templates_use_a_textarea():
    """回归：两个长模板挤在单行 input 里，改一个字要横向滚动。"""
    schema = load_schema()
    for key in ("vision_describe_prompt", "vision_qa_prompt"):
        assert schema[key]["type"] == "text", key
        # 长模板：单行 input 装不下（默认值本身就有 70+ 字符，改起来要横向滚动）
        assert len(schema[key]["default"]) > 60, key


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


def test_compress_budget_clamped_by_gate_reports_the_raw_value():
    """回归：15.0 配 13.0 闸门曾被**夹了两次**，用户却只看得到 13.0 或 11.7。

    旧实现两步都各发一条消息：第一步 15.0 → 13.0（notices），第二步 13.0 →
    11.7（warnings），两条都在说"已修正"，生效值却等于任何一条都没写全的数。
    现在只有一条，且原值 15.0 与最终值 11.7 出现在同一句里。
    """
    raw = {"max_size_mb": 13.0, "compress_max_mb": 15.0}
    cfg, notices = VideoConfig.from_mapping_detailed(raw)
    assert cfg.compress_max_mb == pytest.approx(11.7)

    gate_notices = [item for item in notices if "compress_max_mb" in item]
    assert len(gate_notices) == 1, notices
    assert "15.0" in gate_notices[0], notices  # 用户原值
    assert "13.0" in gate_notices[0], notices  # 体积闸门
    assert "11.7" in gate_notices[0], notices  # 最终生效值
    # 同一个字段只报一次，不再有第二条"自动修正"
    fixed, warnings = cfg.normalized()
    assert fixed.compress_max_mb == pytest.approx(11.7)
    assert warnings == []


def test_normalized_still_guards_a_directly_constructed_conflict():
    """``normalized()`` 仍是**不变量守卫**：直接构造的 13/13 也要被修正。

    ``from_mapping`` 那条路已经单趟算完（不会再触发），但绕过它的构造方式
    （测试、未来的配置来源）必须继续被兜住，并且告警说得出当前值。
    """
    cfg, warnings = VideoConfig(max_size_mb=13.0, compress_max_mb=13.0).normalized()
    assert cfg.compress_max_mb == pytest.approx(11.7)
    assert any("compress_max_mb(13.0)" in item for item in warnings), warnings
    assert any("已自动下调为 11.7" in item for item in warnings), warnings
    # 守卫本身也幂等
    again, again_warnings = cfg.normalized()
    assert again == cfg and again_warnings == []


def test_no_hidden_raw_value_state_on_the_frozen_dataclass():
    """回归：不得再用 ``object.__setattr__`` 往 frozen dataclass 塞隐状态。

    0.2.1 为了在告警里报"用户写的 15.0"，用一个 ``ClassVar`` + 实例属性记住原值
    —— 既污染了数据模型，又在 ``replace()`` 后丢失。单趟归一后不再需要它。
    """
    assert not hasattr(VideoConfig, "raw_compress_max_mb")
    source = (REPO_ROOT / "core" / "config.py").read_text(encoding="utf-8")
    assert "object.__setattr__" not in source
    assert "ClassVar" not in source
    assert "raw_compress_max_mb" not in source
    raw = {"max_size_mb": 13.0, "compress_max_mb": 15.0}
    remembered, _ = VideoConfig.from_mapping_detailed(raw)
    assert remembered == VideoConfig.from_mapping(raw)
    assert not hasattr(remembered, "raw_compress_max_mb")


def test_private_helpers_are_not_public_api():
    """``_resolve_compress_mb`` / ``_range_notices`` 只是内部实现。"""
    assert config_mod._as_int(float("inf"), 7, maximum=3) == 3
    assert config_mod._as_int(float("-inf"), 7, minimum=1) == 1
    final, notice = config_mod._resolve_compress_mb(15.0, 13.0)
    assert final == pytest.approx(11.7) and "11.7" in notice


def test_non_finite_megabyte_values_stay_readable():
    """回归（VERIFY2 P3-4）：``_mb_text(inf)`` 曾输出 ``"inf.0"``。

    ``1e999`` 是**合法 JSON**，所以 ``compress_max_mb=1e999`` 真的会走到提示里。
    旧实现把 ``inf`` 当作"没有小数点"再补一位小数，于是用户看到
    ``compress_max_mb=inf.0 超过体积闸门 …`` —— 既不是数字也不是中文。
    """
    assert config_mod._mb_text(float("inf")) == "∞"
    assert config_mod._mb_text(float("-inf")) == "-∞"
    assert config_mod._mb_text(float("nan")) == "非数值"
    # 有限值那几条不能因为 inf 分支而改变（常用路径）
    assert config_mod._mb_text(15.0) == "15.0"
    assert config_mod._mb_text(11.700000000000001) == "11.7"
    assert config_mod._mb_text(0.75) == "0.75"

    # 端到端：闸门 2000（上界）+ 预算 1e999 → 一条提示，且数字可读
    raw = json.loads('{"max_size_mb": 1e999, "compress_max_mb": 1e999}')
    cfg, notices = VideoConfig.from_mapping_detailed(raw)
    budget = [item for item in notices if "compress_max_mb" in item]
    assert len(budget) == 1, notices
    assert "inf" not in budget[0], budget[0]
    assert budget[0] == (
        "compress_max_mb=∞ 超过体积闸门 max_size_mb=2000.0，"
        "已按闸门的 90% 收敛为 1800.0"
    )
    assert cfg.compress_max_mb == pytest.approx(MAX_SIZE_MB_CEILING * 0.9)


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


# ---------------------------------------- 0.3.0：视觉缓存三个键（边界/垃圾值）


#: 视觉缓存的三个键与 (默认值, 下界, 上界)；``vision_cache_enabled`` 是布尔，另测
VISION_CACHE_KEYS = {
    "vision_cache_ttl_sec": (600.0, 60.0, VISION_CACHE_TTL_CEILING),
    "vision_cache_max_entries": (64, 1, VISION_CACHE_ENTRIES_CEILING),
}


def test_vision_cache_numeric_keys_bounds_and_garbage():
    for key, (default, low, high) in VISION_CACHE_KEYS.items():
        # 明确"取不到数字"的垃圾值 -> 回落默认
        for value in NON_NUMERIC_GARBAGE:
            assert getattr(VideoConfig.from_mapping({key: value}), key) == default, (
                key,
                value,
            )
        # 任何数值 -> 收敛进边界，绝不抛
        for value in GARBAGE_VALUES:
            resolved = getattr(VideoConfig.from_mapping({key: value}), key)
            assert low <= resolved <= high, (key, value, resolved)
        # 类型必须严格（int/float），schema 里 `test_schema_type_matches_annotation`
        # 还会再对一遍注解
        assert type(VideoConfig.from_mapping({key: default}).__dict__[key]) is type(
            default
        ), key
        bounds = config_mod._NUMERIC_BOUNDS
        assert bounds[key] == (low, high), key


def test_vision_cache_numeric_keys_clamp_with_one_notice():
    cfg, notices = VideoConfig.from_mapping_detailed({"vision_cache_ttl_sec": 5})
    assert cfg.vision_cache_ttl_sec == 60.0
    assert notices == [
        f"vision_cache_ttl_sec={5:g} 低于下限 {60.0:g}，已收敛为 {60.0:g}"
    ], notices

    cfg, notices = VideoConfig.from_mapping_detailed({"vision_cache_max_entries": 4096})
    assert cfg.vision_cache_max_entries == VISION_CACHE_ENTRIES_CEILING
    assert any("vision_cache_max_entries" in item for item in notices)


def test_vision_cache_enabled_defaults_to_true_and_false_disables_it():
    assert VideoConfig().vision_cache_enabled is True
    assert VideoConfig.from_mapping({}).vision_cache_enabled is True
    for value in (False, 0, "false", "off", "no", "否"):
        assert VideoConfig.from_mapping({"vision_cache_enabled": value}).vision_cache_enabled is False
    for value in (None, [], {}, object()):
        assert VideoConfig.from_mapping({"vision_cache_enabled": value}).vision_cache_enabled is True
