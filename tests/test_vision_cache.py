"""视觉结果缓存：key 六个维度、TTL、LRU、三态、以及"命中真的省了一次 HTTP"。

对应 ``AUDIT-030`` §C.2 与 §D.1 #14-#19。全部确定性：时间走可注入的 ``clock``，
HTTP 全部打本机 ``http.server`` 假端点（**不联网**），视频用真的 ffmpeg 生成。

这个文件刻意分成两层：

* ``VisionCache`` / ``cache_key`` 的纯单测（不需要 ffmpeg、不需要 AstrBot）；
* 端到端：真 ``httpx`` + 真本机端点 + 真 ``prepare_video`` + 真钩子，
  用**端点侧的请求计数**证明"第二次完全相同投递没有发 HTTP"。
"""

from __future__ import annotations

import asyncio
import dataclasses
import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from types import SimpleNamespace

import pytest

from vv_plugin.core import vision_cache as vision_cache_module
from vv_plugin.core.config import VideoConfig
from vv_plugin.core.vision_cache import (
    KEY_VERSION,
    CacheHit,
    VisionCache,
    cache_key,
)
from vv_plugin.hooks import llm_request as hooks


# --------------------------------------------------------------- 工具与夹具


class _Clock:
    """可手动推进的单调时钟。"""

    def __init__(self, now: float = 1000.0) -> None:
        self.now = now

    def __call__(self) -> float:
        return self.now


def fields(**overrides):
    base = {
        "video": "as-is:aaa",
        "prompt": "描述这段视频",
        "proto": "chat",
        "url": "https://example.invalid/v1/chat/completions",
        "model": "vision-model",
        "max_tokens": 512,
    }
    base.update(overrides)
    return base


class _TextPart:
    def __init__(self, text: str) -> None:
        self.text = text


def make_part(path: str, name: str = "a.mp4") -> _TextPart:
    return _TextPart(f"[Video Attachment: name {name}, path {path}]")


def make_plugin(base_url: str, *, vision_cache, **kwargs) -> SimpleNamespace:
    options = {
        "provider_mode": "openai_compat",
        "openai_base_url": base_url,
        "openai_api_key": "sk-unit-test",
        "openai_model": "vision-model",
        "openai_timeout_sec": 10.0,
        "openai_max_tokens": 512,
    }
    options.update(kwargs)
    return SimpleNamespace(
        context=None,
        config=VideoConfig(**options),
        cache_root="",
        vision_cache=vision_cache,
    )


def run_hook(plugin, path: str, question: str = "这视频讲了什么") -> list:
    """跑一次真实钩子，返回注入后的 parts（每个请求都是新的 req）。"""
    req = SimpleNamespace(extra_user_content_parts=[make_part(path)])
    event = SimpleNamespace(
        unified_msg_origin="umo", get_message_str=lambda: question
    )
    asyncio.run(hooks.handle_llm_request(plugin, event, req))
    return req.extra_user_content_parts


# ------------------------------------------------------- 假端点（真 HTTP）


class _State:
    def __init__(self) -> None:
        self.requests: list[dict] = []
        self.status = 200
        self.lock = threading.Lock()


STATE = _State()


class _Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def do_POST(self) -> None:  # BaseHTTPRequestHandler 的约定名（N802 豁免）
        length = int(self.headers.get("Content-Length") or 0)
        raw = self.rfile.read(length)
        try:
            payload = json.loads(raw.decode("utf-8")) if raw else {}
        except Exception:  # noqa: BLE001
            payload = {"_raw": raw.decode("utf-8", "replace")}
        with STATE.lock:
            STATE.requests.append({"path": self.path, "json": payload})
            attempt = len(STATE.requests)
            status = STATE.status
        if status != 200:
            body = b'{"error": "boom"}'
            self._send(status, body, "application/json")
            return
        body = json.dumps(
            {
                "choices": [
                    {
                        "message": {
                            "role": "assistant",
                            "content": f"第 {attempt} 次分析：画面里有一只橘猫",
                        }
                    }
                ],
                "usage": {"prompt_tokens": 262, "completion_tokens": 17},
            }
        ).encode("utf-8")
        self._send(200, body, "application/json")

    def _send(self, status: int, data: bytes, content_type: str) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Connection", "close")
        self.end_headers()
        try:
            self.wfile.write(data)
        except OSError:  # 客户端已取消
            pass

    def log_message(self, *args, **kwargs) -> None:  # 静音
        pass


@pytest.fixture(scope="session")
def _server() -> str:
    server = ThreadingHTTPServer(("127.0.0.1", 0), _Handler)
    server.daemon_threads = True
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_address[1]}"
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=10)


@pytest.fixture
def endpoint(_server):
    STATE.requests.clear()
    STATE.status = 200
    yield SimpleNamespace(url=_server, requests=STATE.requests, state=STATE)
    STATE.requests.clear()
    STATE.status = 200


# ============================================ cache_key：六个维度 + 版本


def test_key_is_stable_for_the_same_input():
    assert cache_key(**fields()) == cache_key(**fields())
    # 纯 hex（sha256），不夹带任何原文
    key = cache_key(**fields())
    assert len(key) == 64 and all(ch in "0123456789abcdef" for ch in key)


@pytest.mark.parametrize(
    ("changed", "value"),
    [
        ("video", "compressed:bbb"),
        ("prompt", "只描述画面里的动物"),
        ("proto", "responses"),
        ("url", "https://example.invalid/v1/responses"),
        ("model", "another-model"),
        ("max_tokens", 1024),
    ],
)
def test_key_isolates_every_dimension(changed, value):
    """少一个维度就"串味"：用 A 问题的答案回答 B 问题（AUDIT-030 §E-6）。"""
    base = cache_key(**fields())
    assert cache_key(**fields(**{changed: value})) != base, changed


def test_key_is_versioned(monkeypatch):
    """payload 结构变化（换字段形态）必须让旧条目失效（``KEY_VERSION``）。"""
    assert KEY_VERSION == 1
    key_v1 = cache_key(**fields())
    monkeypatch.setattr(vision_cache_module, "KEY_VERSION", 2)
    assert cache_key(**fields()) != key_v1


def test_key_separates_field_boundaries():
    """字段边界必须无歧义：prompt 里含分隔符不许和别的字段"拼"出同一个 key。

    朴素的 ``"|".join(...)`` 会在这里撞车（``video="a|b", prompt="c"`` 与
    ``video="a", prompt="b|c"`` 拼出同一个字符串），所以 key 走规范化 JSON。
    """
    left = cache_key(**fields(video="a|b", prompt="c"))
    right = cache_key(**fields(video="a", prompt="b|c"))
    assert left != right


def test_key_survives_non_ascii_and_newlines():
    first = cache_key(**fields(prompt="描述\n第二行｜中文"))
    second = cache_key(**fields(prompt="描述\n第二行｜中文 "))
    assert first != second  # 空格也是内容的一部分，不许 strip 掉造成串味


# ==================================================== VisionCache：TTL/LRU


def test_hit_returns_the_same_analysis():
    cache = VisionCache(ttl_sec=600.0, max_entries=4, clock=_Clock())
    cache.put("k", "画面里有一只猫")
    assert cache.get("k") == "画面里有一只猫"
    assert cache.stats.hits == 1
    assert cache.stats.misses == 0
    assert cache.stats.stores == 1


def test_miss_returns_none():
    cache = VisionCache(ttl_sec=600.0, max_entries=4, clock=_Clock())
    assert cache.get("nope") is None
    assert cache.stats.misses == 1


def test_lookup_reports_the_remaining_ttl_without_refreshing_it():
    clock = _Clock()
    cache = VisionCache(ttl_sec=600.0, max_entries=4, clock=clock)
    cache.put("k", "x")
    clock.now += 100.0
    hit = cache.lookup("k")
    assert isinstance(hit, CacheHit)
    assert hit.analysis == "x"
    assert hit.ttl_left == pytest.approx(500.0)
    clock.now += 400.0  # 命中**不**续期：仍然按写入时刻算
    assert cache.lookup("k").ttl_left == pytest.approx(100.0)


def test_ttl_expiry():
    clock = _Clock()
    cache = VisionCache(ttl_sec=600.0, max_entries=4, clock=clock)
    cache.put("k", "x")
    clock.now += 599.9
    assert cache.get("k") == "x"
    clock.now += 0.2  # 越过 600s
    assert cache.get("k") is None
    assert cache.stats.expirations == 1
    assert len(cache) == 0, "过期条目必须真的被删掉（不是只查不到）"


def test_lru_eviction_by_capacity():
    clock = _Clock()
    cache = VisionCache(ttl_sec=600.0, max_entries=2, clock=clock)
    cache.put("a", "A")
    cache.put("b", "B")
    cache.put("c", "C")  # 超容量 -> 淘汰最久未用的 a
    assert cache.get("a") is None
    assert cache.get("b") == "B"
    assert cache.get("c") == "C"
    assert cache.stats.evictions == 1

    # 命中会刷新使用次序：现在 b 最久未用
    assert cache.get("b") == "B"
    cache.put("d", "D")
    assert cache.get("b") == "B", "刚命中过的条目不许被淘汰"
    assert cache.get("c") is None, "命中刷新后，最久未用的应该是 c"
    assert cache.get("d") == "D"


def test_put_overwrites_and_refreshes_the_write_clock():
    clock = _Clock()
    cache = VisionCache(ttl_sec=600.0, max_entries=2, clock=clock)
    cache.put("k", "old")
    clock.now += 500.0
    cache.put("k", "new")
    clock.now += 400.0  # 距**上一次写入** 400s < 600s
    assert cache.get("k") == "new"
    assert cache.stats.stores == 2
    assert cache.stats.evictions == 0


def test_empty_analysis_is_never_stored():
    cache = VisionCache(ttl_sec=600.0, max_entries=4, clock=_Clock())
    for value in ("", "   ", "\n\t "):
        cache.put("k", value)
    assert len(cache) == 0
    assert cache.stats.stores == 0


def test_clear_resets_entries_but_keeps_counters():
    cache = VisionCache(ttl_sec=600.0, max_entries=4, clock=_Clock())
    cache.put("k", "x")
    cache.get("k")
    cache.clear()
    assert len(cache) == 0
    assert cache.stats.hits == 1


def test_thread_safety_under_concurrent_put_and_get():
    cache = VisionCache(ttl_sec=600.0, max_entries=8)
    errors: list[BaseException] = []

    def worker(index: int) -> None:
        try:
            for step in range(200):
                key = f"k{index % 4}-{step % 3}"
                cache.put(key, f"v{index}-{step}")
                cache.get(key)
        except BaseException as exc:  # pragma: no cover - 失败即线程安全问题
            errors.append(exc)

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(8)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(10)

    assert errors == []
    assert len(cache) <= 8


def test_stats_start_at_zero_and_are_additive():
    cache = VisionCache(ttl_sec=600.0, max_entries=1, clock=_Clock())
    stats = cache.stats
    assert (stats.hits, stats.misses, stats.stores, stats.evictions) == (0, 0, 0, 0)
    cache.put("a", "A")
    cache.put("b", "B")  # 淘汰 a
    cache.get("a")  # miss
    cache.get("b")  # hit
    assert (cache.stats.hits, cache.stats.misses) == (1, 1)
    assert cache.stats.evictions == 1


# ============================== VideoInjection.content_key（喂给 key 的第 1 维）


def test_injector_content_key_only_when_openai_mode_with_the_cache(
    media_tools, tiny_video, make_placeholder, stub_logger
):
    from vv_plugin.core.injector import prepare_video
    from vv_plugin.core.media import content_hash

    def run(**kwargs):
        config = VideoConfig(provider_mode="gemini_native", **kwargs)
        return prepare_video(
            make_placeholder(tiny_video),
            config=config,
            cache_root="",
            logger=stub_logger,
        )

    # gemini_native：不发视觉请求，没有可缓存的文本结果 -> 不算
    assert run().injection.content_key == ""
    # openai_compat + 缓存开 -> 直通时是 as-is:<源文件哈希>
    config = VideoConfig(
        provider_mode="openai_compat",
        openai_base_url="http://127.0.0.1:9/v1",
        openai_api_key="sk-x",
        openai_model="m",
        vision_cache_enabled=True,
    )
    outcome = prepare_video(
        make_placeholder(tiny_video), config=config, cache_root="", logger=stub_logger
    )
    assert outcome.ok
    assert outcome.injection.content_key == f"as-is:{content_hash(tiny_video)}"
    # 缓存关掉：明确不花这次哈希
    off = prepare_video(
        make_placeholder(tiny_video),
        config=dataclasses.replace(config, vision_cache_enabled=False),
        cache_root="",
        logger=stub_logger,
    )
    assert off.injection.content_key == ""


def test_content_key_is_reproducible_for_compressed_videos(
    media_tools, ffmpeg_dir, noisy_video, tmp_path, make_placeholder, stub_logger
):
    """压缩路径的 content_key = **源内容哈希 + 压缩参数**（不是产物哈希）。

    回归（ARCH-VV-C2 P1-1）：产物字节**不可复现** —— 同一源 + 同一组参数每次压
    出来的 mp4 都不一样（熵编码 + 封装时间戳 + ``-fs`` 在 GOP 边界截断），于是
    ``compressed:<产物哈希>`` 每次都变：``vision_cache_enabled=true`` +
    ``cache_enabled=false`` 时视觉缓存**永不命中**（实测 3 次同源同参 = 3 个 key）。

    现在 key 与压缩缓存 key **同源**（``core/injector.py`` 里同一个 signature：
    源内容哈希 | compress_max_mb | crf | preset | truncate_to），于是：

    * 同源同参 -> 同一个 key（重复投递才可能命中）；
    * 换压缩参数 -> 换 key（模型看到的画面变了，不许共用缓存）。

    用**真 ffmpeg**：这里证明的是"参数进 key、字节不进 key"。
    """
    from vv_plugin.core.injector import content_hash_from_text, prepare_video
    from vv_plugin.core.media import content_hash

    def compress(**kwargs):
        config = VideoConfig(
            provider_mode="openai_compat",
            openai_base_url="http://127.0.0.1:9/v1",
            openai_api_key="sk-x",
            openai_model="m",
            ffmpeg_path=ffmpeg_dir,
            max_size_mb=0.25,
            compress_max_mb=0.2,
            **kwargs,
        )
        return prepare_video(
            make_placeholder(noisy_video),
            config=config,
            cache_root=str(tmp_path / "cache"),
            logger=stub_logger,
        )

    def expected(crf: int) -> str:
        """独立算一遍签名（不调生产代码里的帮助函数，避免自证）。"""
        return "compressed:" + content_hash_from_text(
            f"{content_hash(noisy_video)}|0.2|{crf}|veryfast|0.0"
        )

    outcome = compress()
    assert outcome.ok, outcome.reason
    injection = outcome.injection
    assert injection.action == "compressed"
    assert injection.content_key == expected(30)
    # 产物确实不是源文件（这也正是"产物哈希"当初看起来正确的来源）
    assert content_hash(injection.used_path) != content_hash(noisy_video)

    # 同源同参再压一次：content_key 必须**完全相同**（否则重复投递永不命中）
    again = compress()
    assert again.ok, again.reason
    assert again.injection.content_key == injection.content_key

    # 换一组压缩参数 -> 画面变了 -> content_key 必须不同（否则串味）
    other = compress(compress_crf=40)
    assert other.ok, other.reason
    assert other.injection.content_key == expected(40)
    assert other.injection.content_key != injection.content_key


def test_content_key_ignores_the_non_reproducible_artifact_bytes(
    media_tools, ffmpeg_dir, noisy_video, tmp_path, make_placeholder, stub_logger,
    monkeypatch,
):
    """产物字节**每次都不一样**时，content_key 仍必须稳定（P1-1 的根因锁）。

    真 ffmpeg 的产物是"通常不同"，本用例把它变成**确定性**事实：桩每次写出不同
    的字节（带调用序号）。若 key 取产物哈希，这里就会得到两个 key；取
    「源内容哈希 + 压缩参数」则只有一个 —— 而这正是视觉缓存能否命中的全部区别。
    """
    from vv_plugin.core import media
    from vv_plugin.core.injector import prepare_video

    calls: list[list[str]] = []

    def fake_run(args, timeout):
        calls.append(list(args))
        with open(args[-1], "wb") as handle:
            # 非空 + 每次不同：模拟"同一源同参压出来的字节不可复现"
            handle.write(b"\x00\x00\x00\x18ftypmp42" + f"run{len(calls)}".encode())

    monkeypatch.setattr(media, "_run_ffmpeg", fake_run)

    def compress():
        config = VideoConfig(
            provider_mode="openai_compat",
            openai_base_url="http://127.0.0.1:9/v1",
            openai_api_key="sk-x",
            openai_model="m",
            ffmpeg_path=ffmpeg_dir,
            max_size_mb=0.25,
            compress_max_mb=0.2,
            cache_enabled=False,
        )
        return prepare_video(
            make_placeholder(noisy_video),
            config=config,
            cache_root="",
            logger=stub_logger,
        )

    first = compress()
    second = compress()
    assert first.ok and second.ok, (first.reason, second.reason)
    assert len(calls) == 2, "前提：确实压了两次（关掉压缩缓存）"
    assert first.injection.content_key, "视觉缓存开启时必须算得出 key"
    assert first.injection.content_key == second.injection.content_key


def test_compress_cache_key_is_unchanged_and_is_the_vision_key(
    media_tools, ffmpeg_dir, noisy_video, tmp_path, make_placeholder, stub_logger
):
    """压缩缓存 key 必须与 0.3.0 第一版**逐字节相同**，且就是视觉 key 的正文。

    两件事一起锁：

    1. **升级不许让已有缓存失效**：用旧实现的公式手工预热一个产物，新实现必须
       命中它（P1-1 只改"视觉缓存取什么"，不许顺带改压缩缓存的 key）；
    2. **同源承诺**：视觉 key = ``<action>:<压缩缓存 key>`` —— 两处由同一个
       signature 产生，``cache_enabled`` 的开关因此不会改变视觉缓存的 key。
    """
    import shutil

    from vv_plugin.core.injector import content_hash_from_text, prepare_video
    from vv_plugin.core.media import OUTPUT_SUFFIX, content_hash

    cache_root = tmp_path / "cache"
    cache_root.mkdir()
    # 旧实现的公式（0.3.0 第一版，逐字抄）：源内容哈希 | compress_max_mb | crf |
    # preset | truncate_to。truncate_to 在"只超体积、不超时长"时是 0.0。
    legacy_key = content_hash_from_text(
        f"{content_hash(noisy_video)}|0.2|30|veryfast|0.0"
    )
    shutil.copyfile(noisy_video, cache_root / f"{legacy_key}{OUTPUT_SUFFIX}")

    config = VideoConfig(
        provider_mode="openai_compat",
        openai_base_url="http://127.0.0.1:9/v1",
        openai_api_key="sk-x",
        openai_model="m",
        ffmpeg_path=ffmpeg_dir,
        max_size_mb=0.25,
        compress_max_mb=0.2,
        vision_cache_enabled=True,
    )
    outcome = prepare_video(
        make_placeholder(noisy_video),
        config=config,
        cache_root=str(cache_root),
        logger=stub_logger,
    )

    assert outcome.ok, outcome.reason
    assert "缓存命中" in outcome.injection.note, outcome.injection.note
    assert outcome.injection.content_key == f"compressed:{legacy_key}"


# ============================== 端到端：命中真的省了一次 HTTP（真 httpx）


def test_cache_hit_skips_the_service_call(
    endpoint, tiny_video, stub_logger
):
    """同一视频 + 同一提问 + 同一模型/prompt 的重复投递：端点只收到 1 次。"""
    cache = VisionCache(ttl_sec=600.0, max_entries=8, clock=_Clock())
    plugin = make_plugin(endpoint.url, vision_cache=cache)

    first = run_hook(plugin, tiny_video)
    second = run_hook(plugin, tiny_video)

    assert len(endpoint.requests) == 1, "第二次相同投递不该再发一次 HTTP"
    first_text = first[-1].text
    second_text = second[-1].text
    assert "第 1 次分析" in first_text
    assert "第 1 次分析" in second_text, "命中时必须复用第一次的分析原文"
    assert "第 2 次分析" not in second_text, "命中不该产生新的服务端回答"
    assert cache.stats.hits == 1
    assert cache.stats.stores == 1
    # 命中日志必须显式出现（否则运维会以为接口变快了）
    hits = [m for m in stub_logger.messages("info") if "视觉缓存命中" in m]
    assert len(hits) == 1, stub_logger.records
    assert "TTL 剩" in hits[0]


def test_compressed_video_hits_even_with_the_compress_cache_off(
    media_tools, ffmpeg_dir, noisy_video, endpoint, stub_logger
):
    """``cache_enabled=false`` + ``vision_cache_enabled=true``：同源同参必须命中。

    回归（ARCH-VV-C2 P1-1 的**端到端**面）：压缩分支的 key 曾经取产物哈希，而
    同一源 + 同一组参数每次压出来的字节都不同 -> 第 2 次投递换一个 key ->
    **再发一次 HTTP**（实测 3 次投递 = 3 次 POST、hits=0 misses=3 stores=3）。

    修好之后：关掉压缩缓存只是"每次都要重压一遍本地视频"（省不掉的本地开销），
    视觉缓存照常命中 —— 第 2 次投递的服务端 POST 增量必须是 **0**。
    """
    cache = VisionCache(ttl_sec=600.0, max_entries=8, clock=_Clock())
    plugin = make_plugin(
        endpoint.url,
        vision_cache=cache,
        cache_enabled=False,
        ffmpeg_path=ffmpeg_dir,
        max_size_mb=0.25,
        compress_max_mb=0.2,
    )

    first = run_hook(plugin, noisy_video)
    second = run_hook(plugin, noisy_video)

    assert len(endpoint.requests) == 1, "同源同参的重复投递不许再发一次 HTTP"
    assert cache.stats.hits == 1
    assert cache.stats.stores == 1
    assert len(cache) == 1, "只该有一个 key（3 个 key 正是修复前的症状）"
    assert "第 1 次分析" in first[-1].text
    assert "第 1 次分析" in second[-1].text, "命中时必须复用第一次的分析原文"
    assert "第 2 次分析" not in second[-1].text
    # 命中日志照样要出现（与 as-is 路径同一条文案）
    assert len([m for m in stub_logger.messages("info") if "视觉缓存命中" in m]) == 1


def test_a_different_question_misses_and_calls_again(endpoint, tiny_video, stub_logger):
    """换一个提问 -> key 变 -> 必须重新调用（缓存串味才是 bug）。"""
    cache = VisionCache(ttl_sec=600.0, max_entries=8, clock=_Clock())
    plugin = make_plugin(endpoint.url, vision_cache=cache)

    run_hook(plugin, tiny_video, "这视频讲了什么")
    run_hook(plugin, tiny_video, "画面里有几个人")

    assert len(endpoint.requests) == 2
    assert cache.stats.hits == 0
    assert cache.stats.stores == 2
    # 两个请求的 prompt 确实不同（证明是 prompt 这一维生效，不是碰巧）
    prompts = {item["json"]["messages"][0]["content"][0]["text"] for item in endpoint.requests}
    assert len(prompts) == 2


def test_a_different_model_misses(endpoint, tiny_video, stub_logger):
    cache = VisionCache(ttl_sec=600.0, max_entries=8, clock=_Clock())
    plugin = make_plugin(endpoint.url, vision_cache=cache)
    run_hook(plugin, tiny_video)
    other = make_plugin(endpoint.url, vision_cache=cache, openai_model="another-model")
    run_hook(other, tiny_video)
    assert len(endpoint.requests) == 2


def test_ttl_expiry_calls_the_service_again(endpoint, tiny_video, stub_logger):
    clock = _Clock()
    cache = VisionCache(ttl_sec=60.0, max_entries=8, clock=clock)
    plugin = make_plugin(endpoint.url, vision_cache=cache)

    run_hook(plugin, tiny_video)
    clock.now += 30.0
    run_hook(plugin, tiny_video)  # 仍在 TTL 内 -> 命中
    assert len(endpoint.requests) == 1

    clock.now += 31.0  # 越过 60s
    run_hook(plugin, tiny_video)
    assert len(endpoint.requests) == 2
    assert cache.stats.expirations == 1


def test_failures_are_never_cached(endpoint, tiny_video, stub_logger):
    """503 不许被缓存：否则一次抖动会被固定成 TTL 内的持续故障。"""
    endpoint.state.status = 503
    cache = VisionCache(ttl_sec=600.0, max_entries=8, clock=_Clock())
    plugin = make_plugin(endpoint.url, vision_cache=cache)

    run_hook(plugin, tiny_video)
    run_hook(plugin, tiny_video)

    assert len(endpoint.requests) == 2, "失败必须每次重试"
    assert cache.stats.stores == 0
    assert len(cache) == 0
    # 失败仍然走原有的"通知模型"路径，不是静默
    assert any("视觉服务" in m for m in stub_logger.messages("warning"))


def test_disabled_cache_calls_the_service_every_time(endpoint, tiny_video, stub_logger):
    """``vision_cache_enabled=false``：没有 cache 对象，逐字节等于 0.2.1。"""
    plugin = make_plugin(endpoint.url, vision_cache=None)

    run_hook(plugin, tiny_video)
    run_hook(plugin, tiny_video)

    assert len(endpoint.requests) == 2
    assert not [m for m in stub_logger.messages("info") if "视觉缓存命中" in m]


def test_cache_hit_still_marks_the_part_as_temp(endpoint, tiny_video, stub_logger):
    """命中注入的 TextPart 仍然 ``_no_save``：历史不许被污染。"""
    cache = VisionCache(ttl_sec=600.0, max_entries=8, clock=_Clock())
    plugin = make_plugin(endpoint.url, vision_cache=cache)

    run_hook(plugin, tiny_video)
    parts = run_hook(plugin, tiny_video)

    injected = parts[-1]
    dumped = injected.model_dump_for_context()
    assert dumped["_no_save"] is True
    assert "第 1 次分析" in injected.text
