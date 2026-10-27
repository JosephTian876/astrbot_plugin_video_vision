"""视觉请求的**并发**去重（VERIFY-B P2-3）。

`VisionCache` 只挡得住**先后**发生的重复投递；两个群同时发同一段视频时，
两边都查不到缓存、各发一次 60-120s 的上传 + 推理 —— 实测 4 并发 = 4 次 POST。
这里锁住修好之后的行为：同 key 的并发调用只发**一次** HTTP，其余等结果。

三层证据，互相独立：

* ``_Handler`` 在本机 ``http.server`` 上**自己数**真实 TCP POST（不是插件自报）；
* ``cache.stats``（stores / hits）证明只有 leader 走完 "调服务 + 写缓存"；
* 失败/异常用例证明"失败不缓存、异常向等待者传播"。

所有用例不联网（``127.0.0.1``）、不需要 ffmpeg（直接构造 ``PreparedVideo``）。
"""

from __future__ import annotations

import asyncio
import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from vv_plugin.core.config import VideoConfig
from vv_plugin.core.injector import VideoInjection
from vv_plugin.core.openai_vision import VisionResult
from vv_plugin.core.vision_cache import VisionCache
from vv_plugin.hooks import llm_request as hooks


# ------------------------------------------------------------------ 计数端点


class _State:
    """服务端状态：POST 计数 + 一个"扣住响应"的闸门（制造确定的并发窗口）。"""

    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.posts = 0
        self.status = 200
        #: 默认 None = 不阻塞；测试可塞一个 ``threading.Event`` 扣住响应
        self.gate: threading.Event | None = None
        self.gate_reached: threading.Event | None = None


class _Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def do_POST(self) -> None:  # BaseHTTPRequestHandler 的约定名（N802 豁免）
        length = int(self.headers.get("Content-Length") or 0)
        self.rfile.read(length)
        with STATE.lock:
            STATE.posts += 1
            seq = STATE.posts
            status = STATE.status
            gate = STATE.gate
            reached = STATE.gate_reached
        if reached is not None:
            reached.set()
        if gate is not None:
            gate.wait(10)
        if status != 200:
            self._send(status, b'{"error": "boom"}')
            return
        body = json.dumps(
            {
                "choices": [
                    {"message": {"role": "assistant", "content": f"STUB#{seq} 画面里有一只橘猫"}}
                ],
                "usage": {"prompt_tokens": 262, "completion_tokens": 17},
            }
        ).encode("utf-8")
        self._send(200, body)

    def _send(self, status: int, data: bytes) -> None:
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Connection", "close")
        self.end_headers()
        try:
            self.wfile.write(data)
        except OSError:  # 客户端已取消
            pass

    def log_message(self, *args, **kwargs) -> None:  # 静音
        pass


STATE = _State()


@pytest.fixture(scope="module")
def endpoint():
    server = ThreadingHTTPServer(("127.0.0.1", 0), _Handler)
    server.daemon_threads = True
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_address[1]}/v1"
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=10)


@pytest.fixture(autouse=True)
def _reset_state():
    with STATE.lock:
        STATE.posts = 0
        STATE.status = 200
        STATE.gate = None
        STATE.gate_reached = None
    yield
    with STATE.lock:
        STATE.status = 200
        STATE.gate = None
        STATE.gate_reached = None


# ------------------------------------------------------------------ 小工具


def _config(base_url: str) -> VideoConfig:
    return VideoConfig(
        provider_mode="openai_compat",
        openai_base_url=base_url,
        openai_api_key="sk-unit-test",
        openai_model="vision-model",
        openai_max_tokens=512,
        openai_timeout_sec=10.0,
    )


def _injection(content_key: str = "as-is:same-bytes", name: str = "a.mp4"):
    return VideoInjection(
        block={
            "type": "audio_url",
            "audio_url": {"url": "data:video/mp4;base64,AAAA"},
        },
        source_path=name,
        used_path=name,
        size_mb=1.0,
        duration_sec=2.0,
        action="as-is",
        content_key=content_key,
    )


def _prepared(name: str = "a.mp4", content_key: str = "as-is:same-bytes"):
    return [hooks.PreparedVideo(0, name, _injection(content_key, name))]


async def _wait_until(predicate, *, timeout: float = 10.0, what: str = "") -> None:
    loop = asyncio.get_running_loop()
    deadline = loop.time() + timeout
    while not predicate():
        if loop.time() >= deadline:
            raise AssertionError(f"等了 {timeout}s 仍未成立：{what or predicate}")
        await asyncio.sleep(0.005)


def _hold_response() -> threading.Event:
    """让下一次响应卡住，直到测试放行（制造 4 个请求**真正重叠**的窗口）。"""
    release = threading.Event()
    reached = threading.Event()
    with STATE.lock:
        STATE.gate = release
        STATE.gate_reached = reached
    return release


async def _await_gate() -> None:
    assert await asyncio.to_thread(STATE.gate_reached.wait, 10), "leader 没有发出请求"


# ------------------------------------------------------- 1. 4 并发 -> 1 次 HTTP


def test_four_concurrent_identical_deliveries_send_exactly_one_http(endpoint):
    """4 个并发相同投递：恰好 1 次 POST，4 份结果相同，只有 leader 写缓存。"""
    cache = VisionCache(ttl_sec=600.0, max_entries=8)
    config = _config(endpoint)
    prepared = _prepared()

    async def scenario() -> list[list[VisionResult]]:
        release = _hold_response()
        tasks = [
            asyncio.create_task(
                hooks._analyze_prepared_videos(
                    prepared, "这视频讲了什么", config=config, logger=None, cache=cache
                )
            )
            for _ in range(4)
        ]
        await _await_gate()
        # 响应还扣在服务端：此刻若去重失效，另外 3 个会各自再发一次 POST
        await asyncio.sleep(0.2)
        assert STATE.posts == 1, f"并发去重失效：扣住响应期间收到 {STATE.posts} 次 POST"
        release.set()
        return await asyncio.gather(*tasks)

    results = asyncio.run(scenario())

    assert STATE.posts == 1, f"4 个并发相同投递应只发 1 次 POST，实测 {STATE.posts}"
    assert [item[0].analysis for item in results] == [results[0][0].analysis] * 4
    assert "STUB#1" in results[0][0].analysis
    # 只有 leader 走完"调服务 + 写缓存"：等待者不会各写一次
    assert (cache.stats.stores, cache.stats.hits) == (1, 0), cache.stats
    assert len(cache) == 1
    # 没有泄漏的 flight
    assert cache.flights.pending == 0


def test_duplicate_videos_in_one_message_share_one_request_and_keep_their_names(
    endpoint,
):
    """同一条消息里两份**字节相同**的视频：1 次 HTTP，但各自保留自己的文件名。"""
    cache = VisionCache(ttl_sec=600.0, max_entries=8)
    config = _config(endpoint)
    prepared = [
        hooks.PreparedVideo(0, "a.mp4", _injection()),
        hooks.PreparedVideo(1, "b.mp4", _injection()),
    ]

    results = asyncio.run(
        hooks._analyze_prepared_videos(
            prepared, "这视频讲了什么", config=config, logger=None, cache=cache
        )
    )

    assert STATE.posts == 1
    assert [item.name for item in results] == ["a.mp4", "b.mp4"]
    assert results[0].analysis == results[1].analysis


# ------------------------------------------------- 2. 失败不缓存、异常向等待者传播


def test_concurrent_failures_are_shared_but_never_cached(endpoint):
    """4 并发同 key 遇到 503：只发 1 次 POST、都拿到失败、缓存保持空；下次重试。"""
    STATE.status = 503
    cache = VisionCache(ttl_sec=600.0, max_entries=8)
    config = _config(endpoint)
    prepared = _prepared()

    async def scenario() -> list[list[VisionResult]]:
        release = _hold_response()
        tasks = [
            asyncio.create_task(
                hooks._analyze_prepared_videos(
                    prepared, "这视频讲了什么", config=config, logger=None, cache=cache
                )
            )
            for _ in range(4)
        ]
        await _await_gate()
        await asyncio.sleep(0.2)
        release.set()
        return await asyncio.gather(*tasks)

    results = asyncio.run(scenario())

    assert STATE.posts == 1, "并发的同一次失败也要只发一次请求"
    assert all(item[0].ok is False for item in results)
    assert "视觉服务返回 HTTP 503" in results[0][0].reason
    # 失败不缓存：条目 0 条、stores 0
    assert (len(cache), cache.stats.stores) == (0, 0)

    # 失败不被"记忆"：下一次投递必须重新发请求
    asyncio.run(
        hooks._analyze_prepared_videos(
            prepared, "这视频讲了什么", config=config, logger=None, cache=cache
        )
    )
    assert STATE.posts == 2, "失败被缓存了（下一次投递没有再发请求）"


def test_flight_exceptions_propagate_to_every_waiter_and_are_not_kept():
    """``work`` 抛异常：所有等待者都收到同一个异常，flight 表清空，下次重新真跑。"""
    from vv_plugin.core.vision_cache import VisionFlights

    flights = VisionFlights()
    entered = threading.Event()
    calls: list[int] = []

    async def boom() -> str:
        calls.append(1)
        entered.set()
        await asyncio.sleep(0)  # 让出：确保 3 个等待者已经挂上
        raise RuntimeError("端点炸了")

    async def scenario() -> list[BaseException | str]:
        tasks = [asyncio.create_task(flights.run("k", boom)) for _ in range(4)]
        return await asyncio.gather(*tasks, return_exceptions=True)

    errors = asyncio.run(scenario())

    assert len(calls) == 1, f"同 key 只该真跑一次，实测 {len(calls)} 次"
    assert all(isinstance(item, RuntimeError) for item in errors), errors
    assert {str(item) for item in errors} == {"端点炸了"}
    assert flights.pending == 0, "失败后 flight 必须摘干净（否则下次会等一个死 future）"

    # 失败不缓存 → 下一次重新真跑
    calls.clear()

    async def ok() -> str:
        calls.append(1)
        return "fine"

    assert asyncio.run(flights.run("k", ok)) == "fine"
    assert len(calls) == 1


# ------------------------------------------------------- 3. 不同 key 不合并


def test_different_keys_are_not_merged(endpoint):
    """两个不同 key 的并发调用仍各发各的（去重不能把不同视频串成一个答案）。"""
    cache = VisionCache(ttl_sec=600.0, max_entries=8)
    config = _config(endpoint)

    async def scenario() -> None:
        release = _hold_response()
        tasks = [
            asyncio.create_task(
                hooks._analyze_prepared_videos(
                    _prepared("a.mp4", "as-is:aaa"),
                    "问题",
                    config=config,
                    logger=None,
                    cache=cache,
                )
            ),
            asyncio.create_task(
                hooks._analyze_prepared_videos(
                    _prepared("b.mp4", "as-is:bbb"),
                    "问题",
                    config=config,
                    logger=None,
                    cache=cache,
                )
            ),
        ]
        await _await_gate()
        await asyncio.sleep(0.2)
        assert STATE.posts == 2, f"不同 key 不该被合并，实测 {STATE.posts} 次 POST"
        release.set()
        await asyncio.gather(*tasks)

    asyncio.run(scenario())
    assert cache.stats.stores == 2


# ------------------------------------------------ 4. leader 取消 / 空 key / 关闭


def test_cancelled_leader_hands_the_work_over_to_a_waiter():
    """leader 被取消：等待者自己接管，绝不永远等一个没有结果的 flight。"""
    from vv_plugin.core.vision_cache import VisionFlights

    flights = VisionFlights()
    calls: list[str] = []

    async def work() -> str:
        calls.append("call")
        if len(calls) == 1:
            await asyncio.sleep(10)  # 只有会被取消的 leader 会走到这里
        return "artifact"

    async def scenario() -> str:
        leader = asyncio.create_task(flights.run("k", work))
        await asyncio.sleep(0)  # leader 进 flight
        assert flights.pending == 1
        waiter = asyncio.create_task(flights.run("k", work))
        await asyncio.sleep(0)  # waiter 挂上
        leader.cancel()
        assert isinstance(
            (await asyncio.gather(leader, return_exceptions=True))[0],
            asyncio.CancelledError,
        )
        return await asyncio.wait_for(waiter, timeout=5)

    assert asyncio.run(scenario()) == "artifact"
    assert len(calls) == 2, calls  # 被取消的那次 + 接管的这次
    assert flights.pending == 0


def test_empty_key_bypasses_the_flight():
    """算不出 key（``content_key=""``）时不去重：直接跑，绝不登记 flight（fail-open）。"""
    from vv_plugin.core.vision_cache import VisionFlights

    flights = VisionFlights()
    calls: list[int] = []

    async def work() -> str:
        calls.append(1)
        await asyncio.sleep(0)
        return "x"

    async def scenario() -> list[str]:
        return await asyncio.gather(*(flights.run("", work) for _ in range(4)))

    assert asyncio.run(scenario()) == ["x"] * 4
    assert len(calls) == 4, "空 key 不许被合并"
    assert flights.pending == 0


def test_cache_disabled_means_no_dedupe_at_all(endpoint):
    """``cache=None``（``vision_cache_enabled=false``）：不去重，4 并发 = 4 次 POST。"""
    config = _config(endpoint)

    async def scenario() -> None:
        release = _hold_response()
        tasks = [
            asyncio.create_task(
                hooks._analyze_prepared_videos(
                    _prepared(), "这视频讲了什么", config=config, logger=None, cache=None
                )
            )
            for _ in range(4)
        ]
        await _await_gate()
        await asyncio.sleep(0.2)
        assert STATE.posts == 4, "关缓存时不允许有任何额外去重（要能回到 0.2.1）"
        release.set()
        await asyncio.gather(*tasks)

    asyncio.run(scenario())
