"""0.3.0 六个新键：类型垃圾不再静默（AUDIT-VV-R9 §6.2 / P3-b）。

旧实现的实测（``work/vv-r9/logs/e6_config.json``，84 个探针里 **58 个毫无提示**）：

* ``ffmpeg_max_concurrency="false"`` → **2**（用户想关掉限流，结果限流开着）；
* ``ffmpeg_max_concurrency="0.5"``   → **0**（限流被关掉）；
* ``vision_cache_enabled="abc"``     → **False**（垃圾字符串把缓存关了）。

修法只补**可观测性**，不动语义：生效值仍然是原来的值（偷偷改语义会踩到
``tests/test_config.py::test_ffmpeg_singleflight_is_a_bool_that_defaults_to_true``
锁住的词表行为），但会多一条"你填的 X 被当作 Y"的 WARNING。同一个
``(键名, 原值)`` 在进程内只出一条 —— 重复投递/重载不会刷屏。
"""

from __future__ import annotations

import pytest

from vv_plugin.core import config as config_mod
from vv_plugin.core.config import VideoConfig


@pytest.fixture(autouse=True)
def _fresh_ledger():
    """每个用例都从"还没提示过"开始（账本是进程级的，防止用例之间串味）。"""
    config_mod.reset_coercion_warnings()
    yield
    config_mod.reset_coercion_warnings()


def _notices(raw: dict) -> list[str]:
    return VideoConfig.from_mapping_detailed(raw)[1]


# ------------------------------------------------------------ 审计里的三个例子


@pytest.mark.parametrize(
    ("raw", "expected", "fragment"),
    [
        ({"ffmpeg_max_concurrency": "false"}, 2, "已当作默认值 2"),
        ({"ffmpeg_max_concurrency": "0.5"}, 0, "已当作整数 0"),
        ({"vision_cache_enabled": "abc"}, False, "已当作 false"),
    ],
)
def test_the_three_audited_examples_keep_their_value_but_now_warn(
    raw, expected, fragment
):
    """行为不变（生效值与旧实现逐字相同），只是**不再静默**。"""
    key = next(iter(raw))
    config, notices = VideoConfig.from_mapping_detailed(raw)

    assert getattr(config, key) == expected, "只补提示，不许偷偷改语义"
    assert len(notices) == 1, notices
    # 文案必须同时写出"用户填的原值"与"最终生效值"（AUDIT §6.2 的口径）
    assert key in notices[0]
    assert f"{key}={config_mod._raw_text(raw[key])}" in notices[0]
    assert fragment in notices[0], notices[0]


def test_the_other_three_keys_warn_too():
    """六个新键一个都不能漏（审计只举了 3 个例子）。"""
    cases = {
        "ffmpeg_queue_timeout_sec": ("false", 30.0, "已当作默认值 30.0"),
        "ffmpeg_singleflight": ("1e9", False, "已当作 false"),
        "vision_cache_ttl_sec": ("false", 600.0, "已当作默认值 600.0"),
        "vision_cache_max_entries": (4096.5, 1024, "高于上限 1024"),
    }
    for key, (raw, expected, fragment) in cases.items():
        config, notices = VideoConfig.from_mapping_detailed({key: raw})
        assert getattr(config, key) == expected, (key, raw)
        hits = [item for item in notices if key in item]
        assert len(hits) == 1, (key, notices)
        assert fragment in hits[0], (key, hits[0])


@pytest.mark.parametrize(
    ("key", "value", "expected", "fragment"),
    [
        ("ffmpeg_max_concurrency", True, 1, "已当作 1"),
        ("ffmpeg_max_concurrency", False, 0, "已当作 0"),
        ("ffmpeg_queue_timeout_sec", True, 1.0, "已当作 1.0"),
        ("vision_cache_max_entries", True, 1, "已当作 1"),
    ],
)
def test_a_boolean_written_into_a_numeric_key_warns(key, value, expected, fragment):
    """``True`` 会被 ``float()`` 悄悄当成 1：限流开关写成 ``true`` 不该静默变 1。"""
    config, notices = VideoConfig.from_mapping_detailed({key: value})
    assert getattr(config, key) == expected
    assert len(notices) == 1, notices
    assert fragment in notices[0], notices[0]


def test_a_boolean_below_the_lower_bound_is_reported_by_the_range_notice():
    """``False`` 落在下限以下时归 ``_range_notices``：同一字段绝不许报两条。"""
    _config, notices = VideoConfig.from_mapping_detailed(
        {"vision_cache_max_entries": False}
    )
    hits = [item for item in notices if "vision_cache_max_entries" in item]
    assert len(hits) == 1, notices
    assert "低于下限 1，已收敛为 1" in hits[0]


# ------------------------------------------------------------------ 不误报


@pytest.mark.parametrize(
    ("key", "value"),
    [
        ("ffmpeg_max_concurrency", 0),
        ("ffmpeg_max_concurrency", 1),
        ("ffmpeg_max_concurrency", 8),
        ("ffmpeg_max_concurrency", "3"),
        ("ffmpeg_queue_timeout_sec", 1.0),
        ("ffmpeg_queue_timeout_sec", 600),
        ("ffmpeg_singleflight", True),
        ("ffmpeg_singleflight", False),
        ("ffmpeg_singleflight", "true"),
        ("ffmpeg_singleflight", "off"),
        ("ffmpeg_singleflight", "是"),
        ("ffmpeg_singleflight", 1),
        ("ffmpeg_singleflight", 0),
        ("vision_cache_enabled", True),
        ("vision_cache_enabled", "false"),
        ("vision_cache_ttl_sec", 600.0),
        ("vision_cache_ttl_sec", 60),
        ("vision_cache_max_entries", 1),
        ("vision_cache_max_entries", 1024),
    ],
)
def test_a_legal_value_never_warns(key, value):
    """合法值（含字符串形式的数字、词表里的 true/false）一个字都不许多。"""
    _config, notices = VideoConfig.from_mapping_detailed({key: value})
    assert notices == [], notices


def test_a_missing_key_is_not_a_conversion():
    """没写这个键 = 接受默认值，不是"被转换"，不许提示。"""
    assert _notices({}) == []


def test_out_of_range_is_still_reported_by_the_range_notice_only():
    """越界收敛已有 ``_range_notices``：同一字段绝不能报两条。"""
    _config, notices = VideoConfig.from_mapping_detailed(
        {"ffmpeg_max_concurrency": 99}
    )
    hits = [item for item in notices if "ffmpeg_max_concurrency" in item]
    assert len(hits) == 1, notices
    assert "高于上限 8，已收敛为 8" in hits[0]


# ------------------------------------------------------------------ 只吵一次


def test_the_same_key_and_raw_value_is_reported_once():
    raw = {"ffmpeg_max_concurrency": "false"}
    assert len(_notices(raw)) == 1
    assert _notices(raw) == [], "同一个（键名, 原值）第二次不许再吵"
    # 换成一个**新的**错值 -> 必须重新提示（账本不会因"这个键报过"就闭嘴）
    assert len(_notices({"ffmpeg_max_concurrency": "0.5"})) == 1
    # 换成合法值 -> 更不该提示
    assert _notices({"ffmpeg_max_concurrency": 2}) == []


def test_from_mapping_does_not_consume_the_one_shot_ledger():
    """``from_mapping`` 丢弃提示，所以它也不许消费账本（否则启动告警会丢）。"""
    raw = {"vision_cache_enabled": "abc"}
    assert VideoConfig.from_mapping(raw).vision_cache_enabled is False
    assert len(_notices(raw)) == 1


# ------------------------------------------------------- 真正到 WARNING 日志


class _Context:
    async def get_using_provider_async(self, umo=None):
        return None


def test_plugin_startup_logs_the_warning_once_for_every_instance(stub_logger):
    """插件 ``__init__`` 走 ``from_mapping_detailed`` → 每条 notice 一条 WARNING。

    ``main.py:129-130`` 把 ``notices`` 原样打成 ``配置提示:``；重复构造（重载）
    时同一个错值只出现一次，绝不刷屏。
    """
    from vv_plugin.main import VideoVisionPlugin

    raw = {"ffmpeg_max_concurrency": "false"}
    VideoVisionPlugin(_Context(), dict(raw))
    VideoVisionPlugin(_Context(), dict(raw))

    lines = [item for item in stub_logger.messages("warning") if "配置提示:" in item]
    assert len(lines) == 1, lines
    assert 'ffmpeg_max_concurrency="false"' in lines[0]
    assert "已当作默认值 2" in lines[0]
