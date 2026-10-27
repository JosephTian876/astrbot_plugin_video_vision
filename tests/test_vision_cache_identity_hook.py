"""钩子层证据：同字节换后缀 / 换 api_key 必须 miss（AUDIT-VV-R9 §2.2/§2.3）。

这是 ``tests/test_vision_cache_identity.py`` 的**端到端补充**：那里只测
``core.vision_cache.cache_key`` 的载荷，这里测 ``hooks.vision_cache_key`` ——
也就是真正决定"查哪条缓存"的那个函数。少了它，key 的新维度就只是"纸面功能"。
"""

from __future__ import annotations

import asyncio

from vv_plugin.core.config import VideoConfig
from vv_plugin.core.injector import VideoInjection
from vv_plugin.core.openai_vision import VisionResult, build_vision_prompt
from vv_plugin.core.vision_cache import VisionCache
from vv_plugin.hooks import llm_request as hooks

SAME_BYTES = "as-is:bc98a7426643f2859c2f0c991b47f409"


def _injection(*, used_path: str = "a.mp4", content_key: str = SAME_BYTES):
    return VideoInjection(
        block={"type": "audio_url", "audio_url": {"url": "data:video/mp4;base64,AAAA"}},
        source_path=used_path,
        used_path=used_path,
        size_mb=1.0,
        duration_sec=2.0,
        action="as-is",
        content_key=content_key,
    )


def _config(**kwargs):
    options = {
        "provider_mode": "openai_compat",
        "openai_base_url": "http://127.0.0.1:9/v1",
        "openai_api_key": "sk-AAAAAA",
        "openai_model": "vision-model",
    }
    options.update(kwargs)
    return VideoConfig(**options)


def _counting_analyze(monkeypatch, calls: list) -> None:
    async def fake_analyze_videos(videos, question, *, config, logger=None):
        calls.append([name for name, _ in videos])
        return [
            VisionResult(True, name, analysis=f"分析{len(calls)}", elapsed_sec=1.0)
            for name, _ in videos
        ]

    monkeypatch.setattr(hooks, "analyze_videos", fake_analyze_videos)


def test_the_hook_keys_the_same_bytes_differently_for_mov_and_mp4():
    prompt = build_vision_prompt(_config(), "这视频讲了什么")
    mp4 = hooks.vision_cache_key(_injection(used_path="a.mp4"), prompt, _config())
    mov = hooks.vision_cache_key(_injection(used_path="a.mov"), prompt, _config())

    assert mp4 and mov
    assert mp4 != mov, "同一份字节改名 .mov 后必须换 key（否则 .mp4 的答案被复用）"


def test_the_hook_keys_differently_for_a_different_api_key():
    prompt = build_vision_prompt(_config(), "这视频讲了什么")
    first = hooks.vision_cache_key(_injection(), prompt, _config(openai_api_key="sk-AAAAAA"))
    second = hooks.vision_cache_key(_injection(), prompt, _config(openai_api_key="sk-BBBBBB"))

    assert first != second, "换 api_key 必须换 key（中转站按 key 路由时回答会变）"


def test_identical_deliveries_still_share_the_key():
    prompt = build_vision_prompt(_config(), "这视频讲了什么")
    assert hooks.vision_cache_key(_injection(), prompt, _config()) == hooks.vision_cache_key(
        _injection(), prompt, _config()
    )


def test_redelivery_as_mov_requests_the_service_again(monkeypatch, stub_logger):
    """端到端：同一个缓存对象，`.mp4` 之后换成 `.mov` 必须**再请求一次**。"""
    calls: list = []
    _counting_analyze(monkeypatch, calls)
    cache = VisionCache(ttl_sec=600.0, max_entries=4)

    async def deliver(name: str):
        prepared = [hooks.PreparedVideo(0, name, _injection(used_path=name))]
        return await hooks._analyze_prepared_videos(
            prepared, "这视频讲了什么", config=_config(), logger=stub_logger, cache=cache
        )

    asyncio.run(deliver("a.mp4"))
    second = asyncio.run(deliver("a.mov"))

    assert len(calls) == 2, "改后缀后必须重新请求视觉服务（旧实现在这里命中）"
    assert second[0].analysis == "分析2"
    assert cache.stats.hits == 0


def test_switching_the_api_key_requests_the_service_again(monkeypatch, stub_logger):
    calls: list = []
    _counting_analyze(monkeypatch, calls)
    cache = VisionCache(ttl_sec=600.0, max_entries=4)

    async def deliver(api_key: str):
        prepared = [hooks.PreparedVideo(0, "a.mp4", _injection())]
        return await hooks._analyze_prepared_videos(
            prepared,
            "这视频讲了什么",
            config=_config(openai_api_key=api_key),
            logger=stub_logger,
            cache=cache,
        )

    asyncio.run(deliver("sk-AAAAAA"))
    second = asyncio.run(deliver("sk-BBBBBB"))

    assert len(calls) == 2, "换 key 后必须重新请求视觉服务（旧实现在这里命中）"
    assert second[0].analysis == "分析2"
