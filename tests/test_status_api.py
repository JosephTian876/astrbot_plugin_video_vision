"""宿主契约：``companion_v2_status()`` 必须让「陪伴」读得懂。

前四条用例只锁**协议形状**；后两组测 ``enabled`` / ``runtime_ready`` 的取值语义
（就绪 = 实际能不能干活，降级但可用仍算就绪）。逐条对应宿主
``astrbot_host/integration_registry.py:192-219`` 的四道校验（那个文件属于另一个
插件，本仓库不 import 它，所以这里把判据抄成断言）：

* ``:194`` ``inspect.iscoroutinefunction(method)`` 必须为假；
* ``:198-202`` 返回值不能是 awaitable（coroutine 会被 ``close()`` 后抛错）；
* ``:203`` ``type(status) is dict``（**严格**类型）且 ``schema`` 逐字相等；
* ``:204-206`` ``plugin_id`` 逐字相等；``enabled`` 必须 ``type() is bool``；
  ``runtime_ready`` 是 ``None`` 或 ``bool``。

为什么每条用例都构造**真插件**（而不是直接 ``VideoVisionStatusAPI(fake)``）：
``extension_api`` 是挂在 ``VideoVisionPlugin.__init__`` 末尾的**实例属性**，
用假对象测就绕过了"挂没挂上""挂的位置对不对"这两件最容易出错的事。构造本身是
纯内存操作（不联网、不建缓存目录、不跑 ffmpeg）。
"""

from __future__ import annotations

import inspect
import json

from vv_plugin.hooks.status import (
    PLUGIN_ID,
    STATUS_SCHEMA,
    VideoVisionStatusAPI,
)
from vv_plugin.main import VideoVisionPlugin


class _Context:
    """``Star`` 只保存 context，这里给一个空壳（不碰 provider 解析）。"""

    async def get_using_provider_async(self, umo=None):  # pragma: no cover - 用不到
        return None


def make_plugin(**raw_config) -> VideoVisionPlugin:
    """构造真插件；``provider_mode=gemini_native`` 默认不触发降级告警。"""
    options = {"provider_mode": "gemini_native"}
    options.update(raw_config)
    return VideoVisionPlugin(_Context(), options)


def make_status(**raw_config) -> dict:
    """走**生产挂载点**取状态（越像宿主，越能发现挂载问题）。"""
    return make_plugin(**raw_config).extension_api.companion_v2_status()


# --------------------------------------------------------------- 协议形状


def test_status_has_exactly_the_four_contract_fields():
    """**不多给**：宿主只把三个键写进清单（``:208-210``），多给没有收益只有风险。"""
    plugin = make_plugin()
    assert isinstance(plugin.extension_api, VideoVisionStatusAPI)
    # 宿主读的是 ``star_cls.extension_api``（**实例**，star_manager.py:1230-1246）：
    # 类属性拿不到运行中的 config，所以这里必须不是类属性。
    assert "extension_api" not in vars(VideoVisionPlugin)

    status = plugin.extension_api.companion_v2_status()
    assert set(status) == {"schema", "plugin_id", "enabled", "runtime_ready"}
    # 顺序无关，但"恰好四个"与"没有嵌套容器"是两件事，分开锁
    assert all(not isinstance(value, (dict, list, tuple, set)) for value in status.values())


def test_status_schema_and_plugin_id_are_the_host_literals():
    """``schema`` 与 ``plugin_id`` 是**逐字**比较（``:203-204``），不能有 / 或前缀。"""
    status = make_status()

    assert status["schema"] == "companion.provider-status.v1"
    assert status["schema"] == STATUS_SCHEMA
    assert status["plugin_id"] == "astrbot_plugin_video_vision"
    assert status["plugin_id"] == PLUGIN_ID
    # 防 ``author/name`` 形态混入：star_manager 会覆盖实例的 ``plugin_id``
    assert "/" not in status["plugin_id"]
    assert "\\" not in status["plugin_id"]


def test_types_match_the_host_strict_checks():
    """``type() is bool`` 不认 ``1`` / ``0`` / ``"true"`` —— 逐条复刻 ``:203-206``。"""
    status = make_status()

    assert type(status) is dict
    assert type(status["schema"]) is str
    assert type(status["plugin_id"]) is str
    assert type(status["enabled"]) is bool
    assert type(status["runtime_ready"]) is bool
    # 不是 1/0：真值相同的整数也不许过（宿主用的是 type() 而不是 isinstance）
    assert status["enabled"] is True
    assert status["runtime_ready"] is True


def test_method_is_sync_not_coroutine():
    """``async def`` 会被宿主判成"还没有适配器"（``:194``）。"""
    api = make_plugin().extension_api

    assert not inspect.iscoroutinefunction(api.companion_v2_status)
    assert not inspect.iscoroutinefunction(VideoVisionStatusAPI.companion_v2_status)
    status = api.companion_v2_status()
    assert not inspect.isawaitable(status)
    assert not inspect.isgenerator(status)


# ----------------------------------------------------- enabled / runtime_ready


def test_disabled_config_reports_enabled_false():
    """总开关关掉时宿主显示 ``disabled``（``:211-212``）。"""
    status = make_status(enable=False)

    assert status["enabled"] is False
    assert type(status["enabled"]) is bool
    # 关掉总开关时"就绪"必然为假：不装好过让宿主显示 connected
    assert status["runtime_ready"] is False


def test_degraded_but_usable_reports_runtime_ready():
    """``openai_compat`` 三要素为空 → 降级 ``gemini_native`` → **仍然就绪**。

    生产实测：这种配置下插件自动降级，视频理解每次都能正常工作，旧实现却因为
    ``mode_is_degraded`` 把宿主清单标成"尚未就绪"。``runtime_ready`` 的语义是
    "实际能不能干活"，不是"用户配置的模式有没有被原样采纳"。
    """
    degraded = make_status(provider_mode="openai_compat")

    assert degraded["enabled"] is True, "降级不是关闭，别把两件事混成一个字段"
    # 生效模式已是 gemini_native（宿主 provider 亲眼看视频），这条路径可用
    assert degraded["runtime_ready"] is True

    # 反面对照：三要素齐全时同一个字段也必须是 True，否则断言可能恒真
    ready = make_status(
        provider_mode="openai_compat",
        openai_base_url="https://vision.example.invalid/v1",
        openai_api_key="sk-test-not-a-real-key",
        openai_model="some-vision-model",
    )
    assert ready["enabled"] is True
    assert ready["runtime_ready"] is True


def test_both_modes_unusable_reports_not_runtime_ready():
    """两种模式都不可用（总开关关闭 + openai 三要素缺失）→ 未就绪。

    ``gemini_native`` 的依赖是宿主当前的 Gemini provider，插件侧读不到（钩子与
    状态接口都只有配置对象），所以 False 只能来自"插件自己知道不能跑"的情形：
    总开关关掉，而 openai 那条路又没配齐 —— 此时两条路都不通。
    """
    status = make_status(enable=False, provider_mode="openai_compat")

    assert status["enabled"] is False
    assert type(status["enabled"]) is bool
    assert status["runtime_ready"] is False
    assert type(status["runtime_ready"]) is bool


# ------------------------------------------------------------------ 隐私边界


def test_no_private_data_leaks_into_the_payload():
    """``:208-209``：只有标量能跨界，配置原文 / token / 路径一律不许出现。"""
    status = make_status(
        provider_mode="openai_compat",
        openai_base_url="https://vision.example.invalid/v1",
        openai_api_key="sk-super-secret-value",
        openai_model="private-model-name",
        ffmpeg_path="/opt/private/bin",
        vision_describe_prompt="请描述这段画面里的每一个细节",
    )
    payload = json.dumps(status, ensure_ascii=False)

    for secret in (
        "sk-super-secret",
        "vision.example.invalid",
        "private-model-name",
        "private/bin",
        "ffmpeg",
        "cache",
        "请描述这段画面",
    ):
        assert secret not in payload, secret
    # 值里不许有路径分隔符（schema / plugin_id 都是裸标识符）
    for value in status.values():
        assert "/" not in str(value) and "\\" not in str(value), value
    # 最后再收一次口：形状就是那四个键，没有"顺手"多带的诊断字段
    assert set(status) == {"schema", "plugin_id", "enabled", "runtime_ready"}
