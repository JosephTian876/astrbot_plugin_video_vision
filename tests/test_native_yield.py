"""核心自己会传视频时让位（AstrBot PR #10202 之后）。

为什么需要这一组
================

AstrBot 4.29.1 的 ``ProviderRequest`` **没有** ``video_urls`` 字段，核心只把视频
写成一行 ``[Video Attachment: name …, path …]`` 文本，本插件借 ``audio_url`` 槽把
视频块塞进去。PR #10202 合并后核心会自己在 ``video_urls`` 里带上视频，两边同时
生效的后果是**双份注入**（一份原生 video 块 + 一份借道的 audio 块），而模态清洗
只认 ``audio`` 时后者还会被替换成 ``[Audio]`` —— 用户看到的就是"视频又白传了"。

所以核心接管时我们让位：不注入、不压缩、不设 ``_DONE_ATTR``，但**仍然**改写占位
文本（核心自己不改写，模型看到本机路径会去跑 ``cv2`` 抽帧 / ``os.walk`` 找文件）。

判据为什么是两条
----------------

``video_urls`` 非空**不能**单独作为判据：核心在
``tool_loop_agent_runner._assemble_request_context_for_provider`` 里对**未声明**
``video`` 模态的 provider 会主动把 ``video_urls`` 清空（``modalities`` 为
``None`` / ``[]`` 时同样清空 —— "未配置 = 支持全部"的兼容约定显式排除了 video）。
那时让位等于把视频静默丢掉，比双份注入糟得多。因此必须
"``video_urls`` 非空 **且** ``modalities`` 是 list 且含 ``video``"。

两种环境都能跑
--------------

本文件不 import 真 ``ProviderRequest``，全部用 ``SimpleNamespace`` 造 req：

* 不传 ``video_urls`` = 4.29.1 形态（字段根本不存在），判据恒假；
* 传 ``video_urls`` = PR #10202 之后的形态。

这样同一份用例在装 4.29.1 的机器与装了 PR 的机器上结果完全一致，不需要真框架。
"""

from __future__ import annotations

import asyncio
from types import SimpleNamespace

import pytest

from vv_plugin.core.config import VideoConfig
from vv_plugin.hooks import llm_request as hooks

#: 闸门 1 认的 Gemini 系类型（``core/decide.GEMINI_PROVIDER_TYPES`` 的一员）。
GEMINI_TYPE = "googlegenai_chat_completion"
#: 让位判据要看的视频模态值。
WITH_VIDEO = ("text", "image", "audio", "video")
#: 4.29.1 上唯一可能的形态：没有 video 模态。
WITHOUT_VIDEO = ("text", "image", "audio")
#: 核心给的 ``video_urls`` 元素（PR 里既可能是本机路径，也可能是 http URL）。
CORE_VIDEO_URL = r"C:\tmp\a.mp4"


@pytest.fixture(autouse=True)
def _reset_warned():
    """每个用例都从"没告警过"开始，别让 ``_warned_*`` 的集合串场。

    ``_warned_yield`` 用 ``getattr`` 兜底：即使将来这一支被摘掉，本文件里其它
    用例也不会因为 ``AttributeError`` 一起红掉。
    """
    for name in ("_warned_yield", "_warned_gate", "_warned_modality", "_warned_config"):
        getattr(hooks, name, set()).clear()
    yield


class _TextPart:
    """核心产出的占位文本 part（真机上是 ``TextPart``）。"""

    def __init__(self, text: str) -> None:
        self.text = text


class _Context:
    """``plugin.context`` 的替身：只提供 ``get_using_provider_async``。"""

    def __init__(self, provider=None) -> None:
        self._provider = provider

    async def get_using_provider_async(self, umo=None):
        return self._provider


def make_provider(
    modalities: tuple[str, ...] = WITHOUT_VIDEO, *, declare_modalities: bool = True
):
    """造一个 provider；``declare_modalities=False`` 时连 ``modalities`` 键都没有。"""
    provider_config: dict[str, object] = {
        "type": GEMINI_TYPE,
        "id": "gemini-2.5-flash",
    }
    if declare_modalities:
        provider_config["modalities"] = list(modalities)
    return SimpleNamespace(provider_config=provider_config)


def make_plugin(provider, **options):
    config = VideoConfig(**{"provider_mode": "gemini_native", **options})
    return SimpleNamespace(context=_Context(provider), config=config, cache_root="")


def make_req(path: str, *, video_urls: list[str] | None = None) -> SimpleNamespace:
    """造一个请求。

    ``video_urls=None`` 表示**属性不存在**（4.29.1 的 ``ProviderRequest``），
    而不是空列表 —— 判据必须能区分这两者（4.29.1 上 ``getattr`` 取默认值）。
    """
    fields: dict[str, object] = {
        "extra_user_content_parts": [
            _TextPart(f"[Video Attachment: name a.mp4, path {path}]")
        ]
    }
    if video_urls is not None:
        fields["video_urls"] = video_urls
    return SimpleNamespace(**fields)


def run_hook(plugin, req) -> None:
    asyncio.run(
        hooks.handle_llm_request(
            plugin, SimpleNamespace(unified_msg_origin="umo"), req
        )
    )


def _video_blocks(req) -> list:
    """请求里真正被追加的块（占了超出一个占位符的那些）。"""
    return req.extra_user_content_parts[1:]


# ---------------------------------------------------------------- 不该让位的三条


def test_old_core_without_video_urls_still_injects(tiny_video, stub_logger):
    """4.29.1 形态：字段不存在 → 判据恒假 → 行为与 0.3.0 逐字一致。"""
    plugin = make_plugin(make_provider(WITHOUT_VIDEO))
    req = make_req(tiny_video)  # 刻意不传 video_urls

    run_hook(plugin, req)

    parts = req.extra_user_content_parts
    assert len(parts) == 2, [type(item).__name__ for item in parts]
    assert "data:video/mp4;base64," in parts[1].url
    assert parts[1].model_dump_for_context()["_no_save"] is True
    # 既有行为：占位文本被改写成不含本机路径的形式
    assert "path" not in parts[0].text
    assert "a.mp4" in parts[0].text
    assert tiny_video not in parts[0].text
    assert not any("让位" in item for item in stub_logger.messages())


def test_does_not_yield_when_modality_is_absent(tiny_video, stub_logger):
    """核心没声明 video 模态：它会清空 ``video_urls``，让位等于丢视频。"""
    plugin = make_plugin(make_provider(WITHOUT_VIDEO))
    req = make_req(tiny_video, video_urls=[CORE_VIDEO_URL])

    run_hook(plugin, req)

    assert len(_video_blocks(req)) == 1
    assert not any("让位" in item for item in stub_logger.messages())


def test_does_not_yield_when_modalities_is_unset(tiny_video, stub_logger):
    """未配置 ``modalities`` ≠ 支持 video：PR 的"未配置 = 全支持"显式排除了 video。"""
    plugin = make_plugin(make_provider(declare_modalities=False))
    req = make_req(tiny_video, video_urls=[CORE_VIDEO_URL])

    run_hook(plugin, req)

    assert len(_video_blocks(req)) == 1
    assert not any("让位" in item for item in stub_logger.messages())


def test_does_not_yield_when_modalities_is_empty_list(tiny_video, stub_logger):
    """空列表与"没配"同义（核心同样会清空 ``video_urls``）。"""
    plugin = make_plugin(make_provider(()))
    req = make_req(tiny_video, video_urls=[CORE_VIDEO_URL])

    run_hook(plugin, req)

    assert len(_video_blocks(req)) == 1
    assert not any("让位" in item for item in stub_logger.messages())


# ---------------------------------------------------------------- 该让位的五条


def test_yields_when_core_supplies_video_urls_and_video_modality(
    tiny_video, stub_logger
):
    """两条判据都成立：不注入、不压缩，但占位文本仍要抹掉本机路径。"""
    plugin = make_plugin(make_provider(WITH_VIDEO))
    req = make_req(tiny_video, video_urls=[CORE_VIDEO_URL])

    run_hook(plugin, req)

    parts = req.extra_user_content_parts
    assert len(parts) == 1, [type(item).__name__ for item in parts]
    # 核心已经交付视频，所以模型仍要知道"有个视频、叫什么"，只是不再被路径诱导
    assert "path" not in parts[0].text
    assert "a.mp4" in parts[0].text
    assert tiny_video not in parts[0].text
    infos = stub_logger.messages("info")
    assert any("让位" in item for item in infos), infos
    # 让位必须发生在压缩/编码之前：连"视频已备好"都不该出现
    assert not any("视频已备好" in item for item in infos), infos


def test_yield_is_idempotent_and_leaves_no_done_marker(tiny_video, stub_logger):
    """同一请求重复进入：结果不变，且**不设** ``_DONE_ATTR``。

    让位的语义是"本次请求不该由我们注入"，与 ``_DONE_ATTR`` 的"已经注入过"
    相反；用同一个标记表达两种状态会让后续读它的代码把让位误判成已注入。
    不设标记时也天然幂等 —— 改写后的文本不再匹配占位符正则。
    """
    plugin = make_plugin(make_provider(WITH_VIDEO))
    req = make_req(tiny_video, video_urls=[CORE_VIDEO_URL])

    run_hook(plugin, req)
    first_text = req.extra_user_content_parts[0].text
    run_hook(plugin, req)

    parts = req.extra_user_content_parts
    assert len(parts) == 1, [type(item).__name__ for item in parts]
    assert parts[0].text == first_text, "二次让位不该再改写一次"
    assert getattr(req, hooks._DONE_ATTR, None) is None


def test_yield_logs_only_once_per_provider(tiny_video, stub_logger):
    """INFO 每个 provider 只一次，之后降到 DEBUG（不能刷屏，也不能完全看不见）。"""
    plugin = make_plugin(make_provider(WITH_VIDEO))

    for _ in range(3):
        run_hook(plugin, make_req(tiny_video, video_urls=[CORE_VIDEO_URL]))

    infos = [item for item in stub_logger.messages("info") if "让位" in item]
    debugs = [item for item in stub_logger.messages("debug") if "让位" in item]
    assert len(infos) == 1, infos
    assert len(debugs) == 2, debugs


def test_yield_respects_hide_local_path_false(tiny_video, stub_logger):
    """``hide_local_path=false`` 时让位与注入一样**不动**占位文本。"""
    plugin = make_plugin(make_provider(WITH_VIDEO), hide_local_path=False)
    req = make_req(tiny_video, video_urls=[CORE_VIDEO_URL])

    run_hook(plugin, req)

    parts = req.extra_user_content_parts
    assert len(parts) == 1, [type(item).__name__ for item in parts]
    assert "path" in parts[0].text
    assert tiny_video in parts[0].text
    assert any("让位" in item for item in stub_logger.messages("info"))
