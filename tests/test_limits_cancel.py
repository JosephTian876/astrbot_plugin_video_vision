"""取消"占槽中的 await"不许提前归还令牌（VERIFY-A §4 / P2-1 的回归用例）。

背景：``asyncio.to_thread`` 的线程**不可取消**。还槽若只写在
``_Slot.__aexit__`` 里，取消持有槽位的协程时令牌立刻回到队列，而 worker 线程
里的 ffmpeg 还在跑（最长 ``max_compress_seconds``）—— 真实并发于是可以
> ``limit``（VERIFY-A 实测 ``limit=1`` 时峰值 2）。

本文件锁住四件事：

1. 取消**持有者**：取消瞬间 ``inflight`` 仍是 ``limit``，直到 worker 真的结束
   才归还；期间另一个任务的**真实并发**（在线程里独立计数）不超限；
2. worker 以异常收场时同样恰好归还一次（不多还、不漏还）；
3. 取消**排队等待**中的任务：令牌数不变、没有任何 worker 被启动；
4. 连续 50 次取消后 ``inflight`` / ``available`` 回到一致状态。

判定口径：``inflight + available == limit`` 是令牌守恒式；:class:`_ThreadMeter`
在 worker 线程里计数（与限流器自己的账本**互相独立**），用来测真实并发。
"""

from __future__ import annotations

import ast
import asyncio
import threading
from pathlib import Path
from typing import Any

import pytest

from vv_plugin.core.config import VideoConfig
from vv_plugin.core.injector import PrepareOutcome
from vv_plugin.core.limits import SlotLimiter
from vv_plugin.hooks import llm_request as hooks

REPO = Path(__file__).resolve().parents[1]


# ------------------------------------------------------------------ 辅助


class _ThreadMeter:
    """在 worker 线程里数"此刻真有几个 worker 在跑"（含峰值与累计进入数）。"""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self.alive = 0
        self.peak = 0
        self.entered = 0

    def enter(self) -> None:
        with self._lock:
            self.alive += 1
            self.entered += 1
            self.peak = max(self.peak, self.alive)

    def leave(self) -> None:
        with self._lock:
            self.alive -= 1


class _TextPart:
    """与线上同形的文本块（``iter_placeholders`` 只读 ``text``）。"""

    def __init__(self, text: str) -> None:
        self.text = text


def _video_part(path: str, name: str = "a.mp4") -> _TextPart:
    return _TextPart(f"[Video Attachment: name {name}, path {path}]")


async def _wait_until(predicate: Any, *, timeout: float = 10.0, what: str = "") -> None:
    """推进事件循环直到 ``predicate()`` 成立；超时直接失败（不吞）。

    只用于等**单调收敛**的外部事实（worker 真的结束、回调真的跑过），不用来
    制造竞争：所有"谁先谁后"都由显式的 ``Event`` 决定。
    """
    loop = asyncio.get_running_loop()
    deadline = loop.time() + timeout
    while not predicate():
        if loop.time() >= deadline:
            raise AssertionError(f"等了 {timeout}s 仍未成立：{what or predicate}")
        await asyncio.sleep(0.001)


async def _wait_thread_event(event: threading.Event, *, timeout: float = 10.0) -> None:
    """等一个线程侧 ``threading.Event``（在线程里阻塞等待，不轮询睡觉）。"""
    assert await asyncio.wait_for(asyncio.to_thread(event.wait, timeout), timeout + 5), (
        "worker 线程没有在超时内起来"
    )


# --------------------------------------------------- 1. 取消持有者（生产路径）


def test_cancelling_the_holder_keeps_the_slot_until_the_worker_finishes(
    monkeypatch, stub_logger
):
    """``limit=1``：第一条"消息"进槽后在 ffmpeg 期间被取消。

    断言 (a) 取消瞬间 ``inflight`` 仍为 1（没有提前还槽）；(b) worker 真正结束
    后 ``inflight`` 回到 0；(c) 期间第二条消息的真实并发没有越过 ``limit``。

    走的是**生产路径** ``hooks.build_video_blocks``（不是复刻）：只把
    ``prepare_video`` 换成一个在 worker 线程里阻塞的桩。
    """
    limiter = SlotLimiter(1, queue_timeout=30.0)
    meter = _ThreadMeter()
    started = threading.Event()
    release = threading.Event()

    def fake_prepare(placeholder: Any, **kwargs: Any) -> PrepareOutcome:
        meter.enter()
        try:
            started.set()
            assert release.wait(10), "用例没有放行 worker"
            return PrepareOutcome(
                False, reason="stub：本条用例不需要真产物", kind="internal"
            )
        finally:
            meter.leave()

    monkeypatch.setattr(hooks, "prepare_video", fake_prepare)
    config = VideoConfig(provider_mode="gemini_native")

    async def scenario() -> None:
        first = asyncio.create_task(
            hooks.build_video_blocks(
                [_video_part(r"C:\videos\a.mp4")],
                config=config,
                cache_root="",
                limiter=limiter,
            )
        )
        await _wait_thread_event(started)
        assert limiter.inflight == 1

        first.cancel()
        result = await asyncio.gather(first, return_exceptions=True)
        assert isinstance(result[0], asyncio.CancelledError), result

        # (a) 取消瞬间：worker 线程还在跑，槽位**不许**归还
        assert meter.alive == 1, "worker 线程已结束，用例前提不成立"
        assert limiter.inflight == 1, "取消把槽位提前还了（P2-1 复发）"
        assert limiter.available == 0

        # (c) 取消期间再来一条"消息"：必须排队，真实并发不许越过 limit=1
        second = asyncio.create_task(
            hooks.build_video_blocks(
                [_video_part(r"C:\videos\b.mp4", "b.mp4")],
                config=config,
                cache_root="",
                limiter=limiter,
            )
        )
        await _wait_until(lambda: limiter.waiters == 1, what="第二条消息没有去排队")
        assert meter.entered == 1, "worker 提前启动了（槽位被提前归还）"
        assert meter.peak == 1, f"取消期间真实并发峰值 {meter.peak} > limit=1"

        release.set()
        await asyncio.wait_for(second, timeout=10)

        # (b) worker 结束后槽位与令牌都回到原位
        assert meter.alive == 0
        assert meter.peak == 1, f"真实并发峰值 {meter.peak} > limit=1"
        assert limiter.inflight == 0
        assert limiter.available == 1
        assert limiter.inflight + limiter.available == limiter.limit

    asyncio.run(scenario())


# ------------------------------------------- 2. 取消后 worker 以异常收场


def test_cancelled_holder_returns_the_token_exactly_once_when_the_worker_raises():
    """worker 抛异常也必须恰好还一次：不多还（令牌超发）也不漏还（槽位永久 -1）。

    顺带锁住"没人 await 的那条 worker 异常不许变成 loop 噪音"：没人取走的
    异常会被 ``asyncio`` 报成 ``Task exception was never retrieved``。
    """
    limiter = SlotLimiter(1, queue_timeout=30.0)
    meter = _ThreadMeter()
    started = threading.Event()
    release = threading.Event()
    noise: list[tuple[str, object]] = []

    def worker() -> str:
        meter.enter()
        try:
            started.set()
            assert release.wait(10), "用例没有放行 worker"
            raise RuntimeError("worker 炸了")
        finally:
            meter.leave()

    async def scenario() -> None:
        loop = asyncio.get_running_loop()
        previous = loop.get_exception_handler()
        loop.set_exception_handler(lambda _l, ctx: noise.append((ctx["message"], None)))

        async def holder() -> Any:
            async with limiter.slot(name="boom") as slot:
                return await slot.wait_for_worker(asyncio.to_thread(worker))

        task = asyncio.create_task(holder())
        try:
            await _wait_thread_event(started)
            task.cancel()
            result = await asyncio.gather(task, return_exceptions=True)
            assert isinstance(result[0], asyncio.CancelledError), result
            assert limiter.inflight == 1

            release.set()
            await _wait_until(
                lambda: limiter.inflight == 0, what="worker 异常结束后令牌没有回来"
            )
            assert limiter.available == 1
            assert meter.alive == 0
            # 再让事件循环跑几轮：多还一次的话令牌会变成 2（守恒式被破坏）
            for _ in range(5):
                await asyncio.sleep(0)
            assert limiter.available == 1, "worker 收场后被还了不止一次"
            assert limiter.inflight + limiter.available == limiter.limit
        finally:
            loop.set_exception_handler(previous)
        # 取走异常之后不许再报"never retrieved"（否则每条取消都刷一条日志）
        assert noise == [], noise

    asyncio.run(scenario())


# ---------------------------------------------- 3. 取消排队等待中的任务


def test_cancelling_a_queued_waiter_never_starts_a_worker_and_keeps_the_tokens():
    """排队阶段被取消：令牌数不变（``available == 0`` 仍属于持有者），无泄漏。"""
    limiter = SlotLimiter(1, queue_timeout=30.0)
    meter = _ThreadMeter()
    gate = asyncio.Event()

    def worker() -> str:  # pragma: no cover - 真跑起来就说明用例被绕过
        meter.enter()
        try:
            raise AssertionError("排队者不该真的启动 worker")
        finally:
            meter.leave()

    async def scenario() -> None:
        async def holder() -> None:
            async with limiter.slot(name="holder"):
                await gate.wait()

        holding = asyncio.create_task(holder())
        await _wait_until(lambda: limiter.inflight == 1, what="持有者没有进槽")

        async def waiter() -> Any:
            async with limiter.slot(name="waiter") as slot:
                return await slot.wait_for_worker(asyncio.to_thread(worker))

        queued = asyncio.create_task(waiter())
        await _wait_until(lambda: limiter.waiters == 1, what="等待者没有进队列")

        queued.cancel()
        result = await asyncio.gather(queued, return_exceptions=True)
        assert isinstance(result[0], asyncio.CancelledError), result

        assert limiter.inflight == 1, "取消等待者动了持有者的槽位"
        assert limiter.available == 0
        assert limiter.waiters == 0
        assert meter.entered == 0, "等待者被取消后仍启动了 worker"

        gate.set()
        await asyncio.wait_for(holding, timeout=10)
        assert limiter.inflight == 0
        assert limiter.available == 1
        assert limiter.inflight + limiter.available == limiter.limit

    asyncio.run(scenario())


# ------------------------------------------------- 4. 连续 50 次取消


def test_fifty_consecutive_cancellations_leave_the_limiter_consistent():
    """50 轮"进槽 -> 取消 -> worker 结束"之后 ``inflight`` / ``available`` 一致。"""
    limiter = SlotLimiter(1, queue_timeout=30.0)
    meter = _ThreadMeter()
    rounds = 50

    async def scenario() -> None:
        for round_index in range(rounds):
            gate = threading.Event()
            started = threading.Event()

            def worker(gate: threading.Event = gate, started: threading.Event = started):
                meter.enter()
                try:
                    started.set()
                    assert gate.wait(10), "用例没有放行 worker"
                    return f"artifact-{round_index}"
                finally:
                    meter.leave()

            async def holder(worker: Any = worker) -> Any:
                async with limiter.slot(name="round") as slot:
                    return await slot.wait_for_worker(asyncio.to_thread(worker))

            task = asyncio.create_task(holder())
            await _wait_thread_event(started)
            assert limiter.inflight == 1

            task.cancel()
            result = await asyncio.gather(task, return_exceptions=True)
            assert isinstance(result[0], asyncio.CancelledError), result
            assert limiter.inflight == 1, f"第 {round_index} 轮提前还了槽"
            assert limiter.available == 0

            gate.set()
            await _wait_until(
                lambda: limiter.inflight == 0,
                what=f"第 {round_index} 轮 worker 结束后没还槽",
            )
            assert limiter.available == 1, f"第 {round_index} 轮令牌数不对"
            assert limiter.inflight + limiter.available == limiter.limit

        assert meter.entered == rounds
        assert meter.alive == 0
        assert meter.peak == 1, f"连续取消期间真实并发峰值 {meter.peak} > limit=1"
        assert limiter.stats.peak_inflight == 1

    asyncio.run(scenario())


# ------------------------------------------------ 5. 未被取消的既有路径


def test_normal_path_still_returns_the_value_and_releases_once():
    """没有取消时行为不变：拿到 worker 的返回值，出槽还一次令牌。"""
    limiter = SlotLimiter(1)

    async def scenario() -> None:
        async with limiter.slot(name="ok") as slot:
            value = await slot.wait_for_worker(asyncio.to_thread(lambda: "artifact"))
            assert value == "artifact"
            assert limiter.inflight == 1, "还没出槽就把令牌还了"
        assert limiter.inflight == 0
        assert limiter.available == 1

    asyncio.run(scenario())


def test_worker_exception_propagates_and_releases_once():
    """worker 抛异常：异常照常传给调用方，令牌仍然恰好还一次。"""
    limiter = SlotLimiter(1)

    def boom() -> str:
        raise ValueError("worker 炸了")

    async def scenario() -> None:
        with pytest.raises(ValueError):
            async with limiter.slot(name="boom") as slot:
                await slot.wait_for_worker(asyncio.to_thread(boom))
        assert limiter.inflight == 0
        assert limiter.available == 1

    asyncio.run(scenario())


def test_unlimited_limiter_waits_without_shielding(monkeypatch):
    """``limit=0``：直接 await，不建 task、不 shield（0 = 关闭限流的既有承诺）。"""
    limiter = SlotLimiter(0)

    def no_shield(*args: Any, **kwargs: Any) -> Any:  # pragma: no cover - 命中即失败
        raise AssertionError("关闭限流时不许用 asyncio.shield 包一层")

    monkeypatch.setattr(asyncio, "shield", no_shield)

    async def scenario() -> str:
        async with limiter.slot(name="off") as slot:
            return await slot.wait_for_worker(asyncio.to_thread(lambda: "raw"))

    assert asyncio.run(scenario()) == "raw"
    # 不计账、不建队列（与 ``NULL_LIMITER`` 的既有语义一致）
    assert limiter.inflight == 0
    assert limiter.available == 0
    assert limiter._queue is None


# -------------------------------------------------- 6. 结构守卫（调用点）


def _is_slot_call(node: ast.AsyncWith) -> bool:
    return bool(node.items) and all(
        isinstance(item.context_expr, ast.Call)
        and isinstance(item.context_expr.func, ast.Attribute)
        and item.context_expr.func.attr == "slot"
        for item in node.items
    )


def _calls(node: ast.AST, attr: str) -> list[ast.Call]:
    return [
        child
        for child in ast.walk(node)
        if isinstance(child, ast.Call)
        and isinstance(child.func, ast.Attribute)
        and child.func.attr == attr
    ]


def test_every_slot_wrapped_worker_is_awaited_through_wait_for_worker():
    """结构守卫：``to_thread(prepare_video, ...)`` 必须交给 ``slot.wait_for_worker``。

    只把 ``to_thread`` 放进 ``async with slot()`` 是不够的（P2-1 就是这么漏的）：
    还槽必须绑定到 **worker 真正结束**，否则取消时槽位提前归还。
    """
    offenders: list[str] = []
    checked = 0
    for name in ("hooks/llm_request.py", "main.py"):
        tree = ast.parse((REPO / name).read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if not isinstance(node, ast.AsyncWith) or not _is_slot_call(node):
                continue
            spawns = [
                call
                for call in _calls(node, "to_thread")
                if call.args
                and isinstance(call.args[0], ast.Name)
                and call.args[0].id == "prepare_video"
            ]
            if not spawns:
                continue
            checked += len(spawns)
            guarded = len(_calls(node, "wait_for_worker"))
            if guarded != len(spawns):
                offenders.append(
                    f"{name}: {len(spawns)} 处 to_thread(prepare_video)，"
                    f"{guarded} 处交给 wait_for_worker"
                )
    assert checked == 2, f"预期 2 个生产调用点（hooks + /videotest），找到 {checked} 个"
    assert offenders == [], offenders
