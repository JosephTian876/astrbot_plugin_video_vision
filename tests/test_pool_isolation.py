"""默认线程池被占满时，视频准备必须在预算内**要么开始、要么明确 busy**（P1-1）。

AUDIT-VV-R9 §1.5 的机制：``asyncio.to_thread`` 排的是**进程级**默认执行器
（``min(32, cpu+4)``）。池满时任务只是躺在池队列里，而槽位已经先被取走了：

* 当前 LLM 请求无限挂起（``core/limits.py`` 的 ``wait_for_worker`` 对 worker 的
  等待没有超时）；
* 取消也不会还槽：``_hand_off`` 把还槽挂到 worker 的 done-callback 上，而
  "还没启动的 worker" 永远不会有 done —— 槽位被一个**不存在的并发**占住。

本文件只在**默认池被真正占满**的条件下断言三件事：

1. 视频准备在 ``ffmpeg_queue_timeout_sec`` 量级的预算内结束（不许挂起）；
2. 阻塞工作体跑在 ffmpeg **专属**线程里（不再和核心的 media / TTS / ASR 抢池）；
3. ``ffmpeg_max_concurrency=0``（``limit<=0``）仍然走默认池 —— 逐字节等价 0.2.1。
"""

from __future__ import annotations

import asyncio
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Any

from vv_plugin.core.config import VideoConfig
from vv_plugin.core.injector import PrepareOutcome, VideoInjection
from vv_plugin.core.limits import SlotLimiter
from vv_plugin.hooks import llm_request as hooks

#: 与配置默认值一致的量级（配置下限也是 1.0；见 ``core/config.py``）
LIMIT = 2
QUEUE_TIMEOUT = 1.0
#: 占满默认池的线程名前缀（用例用它证明"工作体没跑在这里"）
SHARED_PREFIX = "shared-pool"
#: 视频准备的专属执行器前缀（``core/limits.SlotLimiter``）
DEDICATED_PREFIX = "vv-ffmpeg"


class _TextPart:
    def __init__(self, text: str) -> None:
        self.text = text


def _video_part(path: str, name: str = "a.mp4") -> _TextPart:
    return _TextPart(f"[Video Attachment: name {name}, path {path}]")


def _ok_outcome(path: str) -> PrepareOutcome:
    return PrepareOutcome(
        True,
        VideoInjection(
            block={
                "type": "audio_url",
                "audio_url": {"url": "data:video/mp4;base64,AAAA"},
            },
            source_path=path,
            used_path=path,
            size_mb=1.0,
            duration_sec=8.0,
            action="as-is",
        ),
    )


async def _wait_event(event: threading.Event, timeout: float) -> bool:
    """等一个**线程侧**事件。

    刻意不用 ``asyncio.to_thread(event.wait, …)``：本文件的前提就是默认池被占满，
    那种写法自己也会排队（连"等事件"都等不到）。
    """
    deadline = time.monotonic() + timeout
    while not event.is_set():
        if time.monotonic() >= deadline:
            return False
        await asyncio.sleep(0.005)
    return True


class _SaturatedPool:
    """把 loop 的默认执行器换成一个小池并**占满**它（模拟核心的 media/TTS/ASR）。"""

    def __init__(self, loop: asyncio.AbstractEventLoop, size: int) -> None:
        self.pool = ThreadPoolExecutor(
            max_workers=size, thread_name_prefix=SHARED_PREFIX
        )
        self._started = [threading.Event() for _ in range(size)]
        self._release = threading.Event()
        self._futures: list[Any] = []
        self._size = size
        self._loop = loop

    def occupy(self) -> None:
        self._loop.set_default_executor(self.pool)
        for index in range(self._size):
            self._futures.append(
                self._loop.run_in_executor(None, self._block, index)
            )

    def _block(self, index: int) -> None:
        self._started[index].set()
        self._release.wait(60)

    async def wait_until_full(self, timeout: float = 5.0) -> None:
        for event in self._started:
            assert await _wait_event(event, timeout), "默认池没被占满（用例前提不成立）"

    def release(self) -> None:
        """放行 blocker：默认池的线程回到空闲（用例收尾必须调用）。"""
        self._release.set()


def test_a_full_default_pool_never_blocks_video_preparation(monkeypatch):
    """默认池 20/20（这里 2/2）被占满时，视频准备仍必须在预算内完成。

    旧行为（AUDIT-VV-R9 §1.5 实测）：20 线程池 → ``>=6.0s`` 未完成；
    6 线程池（2 核等价）→ 一直挂到外部 blocker 到期（58.36s）。
    """
    threads: list[str] = []

    def fake_prepare(placeholder: Any, **kwargs: Any) -> PrepareOutcome:
        threads.append(threading.current_thread().name)
        return _ok_outcome(placeholder.path)

    monkeypatch.setattr(hooks, "prepare_video", fake_prepare)
    parts = [_video_part(r"C:\videos\a.mp4")]
    config = VideoConfig(provider_mode="gemini_native")

    async def scenario():
        loop = asyncio.get_running_loop()
        saturated = _SaturatedPool(loop, LIMIT)
        saturated.occupy()
        await saturated.wait_until_full()
        limiter = SlotLimiter(LIMIT, queue_timeout=QUEUE_TIMEOUT)
        task = asyncio.create_task(
            hooks.build_video_blocks(
                parts, config=config, cache_root="", limiter=limiter
            )
        )
        started_at = time.monotonic()
        try:
            done, _pending = await asyncio.wait({task}, timeout=5.0)
            elapsed = time.monotonic() - started_at
            assert done, (
                f"默认池被占满时视频准备 {elapsed:.1f}s 仍未结束"
                "（P1-1：无限挂起）"
            )
            return elapsed, task.result(), limiter, saturated
        finally:
            saturated.release()
            if not task.done():
                task.cancel()
            await asyncio.gather(task, return_exceptions=True)
            # 让红色路径里那些"排在线程池里"的 to_thread 收尾，别把噪音留到下一个用例
            await asyncio.sleep(0.1)
            limiter.shutdown()

    elapsed, result, limiter, saturated = asyncio.run(scenario())
    saturated.pool.shutdown(wait=True)
    prepared, consumed, failures = result

    assert failures == []
    assert consumed == {0}
    assert len(prepared) == 1
    assert elapsed < QUEUE_TIMEOUT + 0.5, (
        f"{elapsed:.2f}s 才结束（预算 {QUEUE_TIMEOUT}s，旧行为是无限挂起）"
    )
    # 工作体必须在**专属**线程里跑：默认池被占满不该影响我们
    assert threads, "prepare_video 根本没被调用"
    assert all(name.startswith(DEDICATED_PREFIX) for name in threads), threads
    # 槽位与令牌都不许留在"未启动的 worker"手里
    assert limiter.inflight == 0
    assert limiter.available == limiter.limit


def test_cancelling_never_leaves_a_slot_with_an_unstarted_worker(monkeypatch):
    """取消只许"等 worker 结束再还槽"，不许把槽留给一个**还没启动**的 worker。

    旧行为：worker 排在被占满的默认池里，取消后 ``inflight`` 一直是 1
    （AUDIT-VV-R9 §1.5 S1：``after_cancel(0.31s): inflight=1, available=0``）。
    """
    started = threading.Event()
    release = threading.Event()
    entered: list[str] = []
    cancelled_while_running: list[int] = []

    def fake_prepare(placeholder: Any, **kwargs: Any) -> PrepareOutcome:
        entered.append(threading.current_thread().name)
        started.set()
        assert release.wait(10), "用例没有放行 worker"
        return PrepareOutcome(False, reason="stub：本用例不需要真产物", kind="internal")

    monkeypatch.setattr(hooks, "prepare_video", fake_prepare)
    parts = [_video_part(r"C:\videos\a.mp4")]
    config = VideoConfig(provider_mode="gemini_native")

    async def scenario() -> None:
        loop = asyncio.get_running_loop()
        saturated = _SaturatedPool(loop, LIMIT)
        saturated.occupy()
        await saturated.wait_until_full()
        limiter = SlotLimiter(LIMIT, queue_timeout=QUEUE_TIMEOUT)
        task = asyncio.create_task(
            hooks.build_video_blocks(
                parts, config=config, cache_root="", limiter=limiter
            )
        )
        try:
            ran_in_time = await _wait_event(started, QUEUE_TIMEOUT + 2.0)
            assert ran_in_time, (
                "worker 在默认池被占满时没有开始执行"
                "（旧行为：排在线程池队列里，槽位被未启动的 worker 占住）"
            )
            cancelled_while_running.append(limiter.inflight)
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)
            # 取消不许提前还槽：worker 还在跑
            assert limiter.inflight == 1, "取消把槽位提前还了"
            release.set()
            deadline = time.monotonic() + 5.0
            while limiter.inflight != 0 and time.monotonic() < deadline:
                await asyncio.sleep(0.005)
            assert limiter.inflight == 0, "worker 结束后槽位没有归还"
            assert limiter.available == limiter.limit
        finally:
            saturated.release()
            release.set()
            if not task.done():
                task.cancel()
            await asyncio.gather(task, return_exceptions=True)
            limiter.shutdown()

    asyncio.run(scenario())
    assert cancelled_while_running == [1]
    assert len(entered) == 1, entered
    assert entered[0].startswith(DEDICATED_PREFIX), entered


def test_limit_zero_still_uses_the_shared_pool():
    """``ffmpeg_max_concurrency=0``：不建专属池，仍走默认池（逐字节等价 0.2.1）。"""
    names: list[str] = []

    def worker() -> str:
        names.append(threading.current_thread().name)
        return "ok"

    async def scenario() -> str:
        loop = asyncio.get_running_loop()
        pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix=SHARED_PREFIX)
        loop.set_default_executor(pool)
        limiter = SlotLimiter(0)
        assert limiter._executor is None, "关闭限流时不许建专属执行器"
        try:
            return await limiter.submit(worker)
        finally:
            limiter.shutdown()

    assert asyncio.run(scenario()) == "ok"
    assert names == [f"{SHARED_PREFIX}_0"], names


def test_the_dedicated_pool_is_lazy_and_sized_by_the_limit():
    """专属池必须**惰性**创建（不处理视频就不建线程），且 ``max_workers == limit``。

    这是"拿到槽就一定有线程"的根据：同时最多 ``limit`` 个 worker，
    池也有 ``limit`` 个线程 —— 于是永远没有"排在池队列里"的状态。
    """
    limiter = SlotLimiter(2)
    try:
        assert limiter._executor is None, "构造时不许建线程池"
        pool = limiter._pool()
        assert pool._max_workers == limiter.limit
        assert limiter._pool() is pool, "每次 submit 不许重建池"
    finally:
        limiter.shutdown()
    assert limiter._executor is None
    # 关掉之后再 submit 不许炸，也不许悄悄重建一个池（退回默认池）
    assert limiter._closed is True
