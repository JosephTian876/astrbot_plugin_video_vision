"""并发限流（async 令牌）与 singleflight（线程事件）的确定性单测。

对应审计 ``AUDIT-030`` §C.1 与 §D.1 的 #1-#13。全部用例**不靠 sleep 竞争**：
等待用 ``asyncio.Event`` / ``threading.Event`` 显式放行，计数用可观察的
``inflight`` / ``waiters`` / 桩 logger。唯一的时间来源是可注入的 ``clock``。
"""

from __future__ import annotations

import ast
import asyncio
import threading
from pathlib import Path

import pytest

from vv_plugin.core.limits import (
    DETERMINISTIC_FAILURE_KINDS,
    QueueTimeout,
    SingleFlight,
    SlotLimiter,
)

REPO = Path(__file__).resolve().parents[1]


# ------------------------------------------------------------------ 辅助


async def _spin(predicate, *, limit: int = 5000) -> None:
    """让事件循环推进到 ``predicate()`` 为真（**不睡墙钟**）。"""
    for _ in range(limit):
        if predicate():
            return
        await asyncio.sleep(0)
    raise AssertionError("等待条件在 5000 次调度后仍未成立")


class _Clock:
    """可手动推进的时钟（``SlotLimiter(clock=...)``）。"""

    def __init__(self, now: float = 0.0) -> None:
        self.now = now

    def __call__(self) -> float:
        return self.now


class _Recorder:
    """收集日志的最小记录器（插件 logger 的鸭子类型）。"""

    def __init__(self) -> None:
        self.records: list[tuple[str, str]] = []
        self._lock = threading.Lock()

    def _add(self, level: str, message: object, *args: object) -> None:
        text = message % args if args else str(message)
        with self._lock:
            self.records.append((level, text))

    def debug(self, message: object, *args: object, **kwargs: object) -> None:
        self._add("debug", message, *args)

    def info(self, message: object, *args: object, **kwargs: object) -> None:
        self._add("info", message, *args)

    def warning(self, message: object, *args: object, **kwargs: object) -> None:
        self._add("warning", message, *args)

    def error(self, message: object, *args: object, **kwargs: object) -> None:
        self._add("error", message, *args)

    def messages(self, level: str | None = None) -> list[str]:
        return [
            text for lvl, text in self.records if level is None or lvl == level
        ]


class _MediaToolError(RuntimeError):
    """``MediaToolError`` 的替身：``SingleFlight`` 只认 ``.kind``，不 import 它。"""

    def __init__(self, message: str, kind: str) -> None:
        super().__init__(message)
        self.kind = kind


def _boom(kind: str):
    """构造一个"一调用就抛"的 work。"""

    def work() -> str:
        raise _MediaToolError(kind, kind=kind)

    return work


# ================================================== SlotLimiter（1-7）


def test_slot_limiter_admits_at_most_limit():
    """#1：8 个协程、limit=2 —— 同时在槽内的永远不超过 2 个。"""

    async def scenario() -> int:
        limiter = SlotLimiter(2)
        gate = asyncio.Event()
        entered: list[int] = []
        peak = 0
        two_in = asyncio.Event()

        async def job(index: int) -> None:
            nonlocal peak
            async with limiter.slot(name=f"job-{index}"):
                entered.append(index)
                peak = max(peak, limiter.inflight)
                if len(entered) == 2:
                    two_in.set()
                await gate.wait()

        tasks = [asyncio.create_task(job(i)) for i in range(8)]
        await asyncio.wait_for(two_in.wait(), timeout=5)
        # 前两个已经占满：此刻不可能有第三个进来
        assert peak == 2
        assert limiter.available == 0
        assert limiter.waiters == 6
        gate.set()
        await asyncio.wait_for(asyncio.gather(*tasks), timeout=5)
        assert len(entered) == 8
        assert limiter.inflight == 0
        assert limiter.available == 2
        assert limiter.stats.peak_inflight == 2
        return peak

    assert asyncio.run(scenario()) == 2


def test_slot_limiter_is_fifo():
    """#2：等待者按**提交顺序**拿槽（3 个等待者 -> 0,1,2）。"""

    async def scenario() -> list[int]:
        limiter = SlotLimiter(1)
        order: list[int] = []
        release = asyncio.Event()

        async def holder() -> None:
            async with limiter.slot():
                await release.wait()

        async def waiter(index: int) -> None:
            async with limiter.slot():
                order.append(index)

        holder_task = asyncio.create_task(holder())
        await _spin(lambda: limiter.inflight == 1)
        waiters = [asyncio.create_task(waiter(i)) for i in range(3)]
        await _spin(lambda: limiter.waiters == 3)
        # 让三个等待者的 queue.get() 真正登记进队列（FIFO 顺序 = 创建顺序）
        for _ in range(5):
            await asyncio.sleep(0)
        release.set()
        await asyncio.wait_for(asyncio.gather(holder_task, *waiters), timeout=5)
        return order

    assert asyncio.run(scenario()) == [0, 1, 2]


def test_slot_limiter_unlimited_when_zero():
    """#3：limit=0 = 关闭限流 —— 峰值 == N，且**没有**任何等待/日志/统计。"""

    async def scenario() -> tuple[int, int]:
        recorder = _Recorder()
        limiter = SlotLimiter(0, logger=recorder, debug=True)
        gate = asyncio.Event()
        active = 0
        peak = 0
        entered = 0
        all_in = asyncio.Event()

        async def job() -> None:
            nonlocal active, peak, entered
            async with limiter.slot(name="x"):
                active += 1
                entered += 1
                peak = max(peak, active)
                if entered == 5:
                    all_in.set()
                await gate.wait()
                active -= 1

        tasks = [asyncio.create_task(job()) for _ in range(5)]
        await asyncio.wait_for(all_in.wait(), timeout=5)
        gate.set()
        await asyncio.wait_for(asyncio.gather(*tasks), timeout=5)
        assert limiter.enabled is False
        assert limiter.stats.waits == 0
        assert limiter.stats.timeouts == 0
        assert recorder.records == []
        return peak, entered

    assert asyncio.run(scenario()) == (5, 5)


def test_slot_limiter_raises_queue_timeout():
    """#4：limit=1 + 占用不放 + timeout=0.05 -> QueueTimeout，令牌不丢。"""

    async def scenario() -> None:
        limiter = SlotLimiter(1, queue_timeout=0.05)
        release = asyncio.Event()

        async def holder() -> None:
            async with limiter.slot():
                await release.wait()

        holder_task = asyncio.create_task(holder())
        await _spin(lambda: limiter.inflight == 1)

        with pytest.raises(QueueTimeout) as excinfo:
            async with limiter.slot(name="late"):
                pass
        error = excinfo.value
        assert error.limit == 1
        assert error.timeout == pytest.approx(0.05)
        assert error.waited >= 0.0
        assert limiter.stats.timeouts == 1

        release.set()
        await asyncio.wait_for(holder_task, timeout=5)
        assert limiter.available == 1
        assert limiter.inflight == 0

    asyncio.run(scenario())


def test_slot_limiter_releases_on_exception():
    """#5：槽内抛异常后令牌必须归位（``available == limit``）。"""

    async def scenario() -> None:
        limiter = SlotLimiter(1)
        with pytest.raises(ValueError):
            async with limiter.slot():
                raise ValueError("boom")
        assert limiter.available == 1
        assert limiter.inflight == 0
        # 归位之后还能再拿到（没有死锁）
        async with limiter.slot():
            assert limiter.inflight == 1
        assert limiter.available == 1

    asyncio.run(scenario())


def test_slot_limiter_cancellation_does_not_leak_tokens():
    """#6：取消 3 个等待者后槽位不减少（**防死锁的核心用例**）。"""

    async def scenario() -> None:
        limiter = SlotLimiter(2, queue_timeout=30.0)
        release = asyncio.Event()

        async def holder() -> None:
            async with limiter.slot():
                await release.wait()

        holders = [asyncio.create_task(holder()) for _ in range(2)]
        await _spin(lambda: limiter.inflight == 2)
        waiters = [asyncio.create_task(holder()) for _ in range(3)]
        await _spin(lambda: limiter.waiters == 3)
        for task in waiters:
            task.cancel()
        await asyncio.gather(*waiters, return_exceptions=True)
        assert limiter.available == 0  # 两个槽仍被 holders 占着
        assert limiter.waiters == 0
        release.set()
        await asyncio.wait_for(asyncio.gather(*holders), timeout=5)
        assert limiter.available == 2
        assert limiter.inflight == 0

        async def quick() -> None:
            async with limiter.slot():
                await asyncio.sleep(0)

        await asyncio.wait_for(asyncio.gather(*(quick() for _ in range(4))), timeout=5)
        assert limiter.available == 2
        assert limiter.inflight == 0

    asyncio.run(scenario())


def test_slot_limiter_survives_loop_recreation():
    """#7：同一实例跨两次 ``asyncio.run`` 仍可用（``asyncio.Queue`` 是 loop 绑定的）。"""

    limiter = SlotLimiter(2)

    async def round_one() -> None:
        async with limiter.slot():
            assert limiter.inflight == 1
        assert limiter.available == 2

    asyncio.run(round_one())
    asyncio.run(round_one())
    assert limiter.available == 2
    assert limiter.inflight == 0
    assert limiter.stats.waits == 0


def test_slot_limiter_logs_a_single_queue_line_when_the_wait_is_long():
    """等待 >= 1.0s 才打一条 INFO（每个视频最多一条），立即拿到只打 debug。"""

    async def scenario() -> _Recorder:
        recorder = _Recorder()
        clock = _Clock()
        limiter = SlotLimiter(1, logger=recorder, clock=clock, debug=True)
        release = asyncio.Event()

        async def holder() -> None:
            async with limiter.slot():
                await release.wait()

        holder_task = asyncio.create_task(holder())
        await _spin(lambda: limiter.inflight == 1)

        async def waiter() -> None:
            async with limiter.slot(name="slow"):
                pass

        waiter_task = asyncio.create_task(waiter())
        await _spin(lambda: limiter.waiters == 1)
        clock.now = 2.5  # 排队 2.5s
        release.set()
        await asyncio.wait_for(asyncio.gather(holder_task, waiter_task), timeout=5)
        return recorder

    recorder = asyncio.run(scenario())
    queued = [item for item in recorder.messages("info") if "ffmpeg 排队" in item]
    assert len(queued) == 1, recorder.records
    assert "2.5s" in queued[0] and "limit=1" in queued[0]
    assert "前方 0 个" in queued[0]


# ================================================== SingleFlight（8-13）


def test_singleflight_runs_the_work_once():
    """#8：4 线程同 key -> 工作体只跑 1 次，4 个返回值相同，joined == 3。"""
    flight = SingleFlight()
    calls: list[str] = []
    store: dict[str, str] = {}
    joined: list[int] = []
    all_joined = threading.Event()
    lock = threading.Lock()

    def on_join() -> None:
        with lock:
            joined.append(1)
            if len(joined) == 3:
                all_joined.set()

    def work() -> str:
        with lock:
            calls.append(threading.current_thread().name)
        # 等 3 个 follower 都真的加入了再产出（不靠 sleep 竞争）
        assert all_joined.wait(5), "3 个 follower 没有全部加入"
        store["k"] = "artifact"
        return "artifact"

    def probe() -> str | None:
        return store.get("k")

    results: list[str] = []
    errors: list[BaseException] = []

    def run() -> None:
        try:
            results.append(
                flight.run("k", work, probe=probe, wait_timeout=5.0, on_join=on_join)
            )
        except BaseException as exc:  # pragma: no cover - 失败时把原因带出来
            errors.append(exc)

    leader = threading.Thread(target=run, name="leader")
    leader.start()
    followers = [threading.Thread(target=run, name=f"f{i}") for i in range(3)]
    for thread in followers:
        thread.start()
    for thread in [leader, *followers]:
        thread.join(10)

    assert errors == []
    assert len(calls) == 1, calls
    assert results == ["artifact"] * 4
    assert len(joined) == 3
    assert flight.pending == 0


def test_singleflight_distinct_keys_run_concurrently():
    """#9：不同 key 必须并行（两个 barrier 都放行才算过；串行会超时）。"""
    flight = SingleFlight()
    barrier = threading.Barrier(2, timeout=5)
    results: list[str] = []
    errors: list[BaseException] = []

    def make_work(value: str):
        def work() -> str:
            barrier.wait()
            return value

        return work

    def run(key: str) -> None:
        try:
            results.append(flight.run(key, make_work(key), wait_timeout=5.0))
        except BaseException as exc:  # pragma: no cover
            errors.append(exc)

    threads = [threading.Thread(target=run, args=(key,)) for key in ("a", "b")]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(10)

    assert errors == []
    assert sorted(results) == ["a", "b"]
    assert flight.pending == 0


def test_singleflight_propagates_deterministic_failure():
    """#10：leader 抛确定性失败 -> follower 原样抛同类，工作体只跑 1 次。"""
    assert "not_found" in DETERMINISTIC_FAILURE_KINDS
    flight = SingleFlight()
    calls: list[int] = []
    kinds: list[str] = []
    joined = threading.Event()

    def work() -> str:
        calls.append(1)
        assert joined.wait(5), "follower 没有加入"
        raise _MediaToolError("未找到 ffmpeg", kind="not_found")

    def run() -> None:
        try:
            flight.run("k", work, probe=lambda: None, wait_timeout=5.0,
                       on_join=joined.set)
        except BaseException as exc:
            kinds.append(getattr(exc, "kind", type(exc).__name__))

    leader = threading.Thread(target=run)
    leader.start()
    follower = threading.Thread(target=run)
    follower.start()
    try:
        leader.join(10)
        follower.join(10)
    finally:
        joined.set()

    assert len(calls) == 1
    assert kinds == ["not_found", "not_found"]
    assert flight.pending == 0


def test_singleflight_follower_retries_transient_failure_once():
    """#11：leader 抛瞬时失败（timeout）-> follower 接管，工作体**恰好** 2 次。"""
    flight = SingleFlight()
    calls: list[str] = []
    results: list[str] = []
    takeover: list[BaseException | None] = []
    failures: list[BaseException] = []
    joined = threading.Event()
    lock = threading.Lock()

    def work() -> str:
        with lock:
            calls.append(threading.current_thread().name)
            attempt = len(calls)
        if attempt == 1:
            assert joined.wait(5), "follower 没有加入"
            raise _MediaToolError("ffmpeg 超时", kind="timeout")
        return "second-attempt"

    def run() -> None:
        try:
            results.append(
                flight.run(
                    "k",
                    work,
                    probe=lambda: "should-not-be-used",
                    wait_timeout=5.0,
                    on_join=joined.set,
                    on_takeover=takeover.append,
                )
            )
        except BaseException as exc:  # leader 的失败在这里被消化掉
            failures.append(exc)

    leader = threading.Thread(target=run, name="leader")
    leader.start()
    follower = threading.Thread(target=run, name="follower")
    follower.start()
    try:
        leader.join(10)
        follower.join(10)
    finally:
        joined.set()

    assert len(calls) == 2, calls
    assert len(takeover) == 1
    assert getattr(takeover[0], "kind", "") == "timeout"
    assert results == ["second-attempt"]
    assert [getattr(exc, "kind", "") for exc in failures] == ["timeout"]
    assert flight.pending == 0


def test_singleflight_follower_timeout_takes_over():
    """#12：leader 卡住不返回 -> follower 等到上限后接管。"""
    flight = SingleFlight()
    started = threading.Event()
    in_second_work = threading.Event()
    calls: list[str] = []
    results: list[str] = []
    release = threading.Event()
    lock = threading.Lock()

    def work() -> str:
        name = threading.current_thread().name
        with lock:
            calls.append(name)
        if name == "leader":
            started.set()
            assert release.wait(10)
            return "leader-done"
        in_second_work.set()
        return "follower-took-over"

    def run_leader() -> None:
        results.append(flight.run("k", work, probe=lambda: "never", wait_timeout=10.0))

    leader = threading.Thread(target=run_leader, name="leader")
    leader.start()
    assert started.wait(5)

    def run_follower() -> None:
        results.append(flight.run("k", work, probe=lambda: "never", wait_timeout=0.05))

    follower = threading.Thread(target=run_follower, name="follower")
    follower.start()
    follower.join(10)
    assert in_second_work.is_set()
    assert results == ["follower-took-over"], results
    assert calls == ["leader", "follower"], calls
    release.set()
    leader.join(10)
    assert flight.pending == 0


def test_singleflight_registry_is_empty_after_the_flight():
    """#13：结束后登记表必须清空（不泄漏 Event / 异常对象）。"""
    flight = SingleFlight()

    assert flight.run("k", lambda: "v", wait_timeout=5.0) == "v"
    assert flight.pending == 0

    with pytest.raises(_MediaToolError):
        flight.run("boom", _boom("corrupt"), wait_timeout=5.0)
    assert flight.pending == 0


def test_a_finished_leader_does_not_remove_a_newer_flight():
    """``finally`` 只摘自己那条：后来的 flight 不能被先结束的调用删掉。"""
    flight = SingleFlight()
    release_first = threading.Event()
    first_started = threading.Event()
    second_started = threading.Event()
    release_second = threading.Event()
    takeover_seen: list[BaseException | None] = []

    def first_work() -> str:
        first_started.set()
        assert release_first.wait(10)
        return "first"

    def second_work() -> str:
        second_started.set()
        assert release_second.wait(10)
        return "second"

    results: list[str] = []

    leader = threading.Thread(
        target=lambda: results.append(
            flight.run("k", first_work, probe=lambda: None, wait_timeout=10.0)
        )
    )
    leader.start()
    assert first_started.wait(5)

    follower = threading.Thread(
        target=lambda: results.append(
            flight.run(
                "k",
                second_work,
                probe=lambda: None,
                wait_timeout=0.05,
                on_takeover=takeover_seen.append,
            )
        )
    )
    follower.start()
    assert second_started.wait(5)  # follower 已经接管并登记了新 flight

    release_first.set()
    leader.join(10)
    # 第一条 flight 结束，但登记表里仍应有 follower 自己那条
    assert flight.pending == 1

    release_second.set()
    follower.join(10)
    assert flight.pending == 0
    assert sorted(results) == ["first", "second"]


# ======================================= 结构守卫：限流只在 async 层


def _slot_context_nodes(tree: ast.AST) -> set[int]:
    """``async with <limiter>.slot(...):`` 体内的全部节点 id。"""
    inside: set[int] = set()
    for node in ast.walk(tree):
        if not isinstance(node, ast.AsyncWith) or not node.items:
            continue
        contexts = [item.context_expr for item in node.items]
        if not all(
            isinstance(ctx, ast.Call)
            and isinstance(ctx.func, ast.Attribute)
            and ctx.func.attr == "slot"
            for ctx in contexts
        ):
            continue
        for child in ast.walk(node):
            inside.add(id(child))
    return inside


def test_every_prepare_video_call_runs_inside_a_limiter_slot():
    """RISK #3 + P1-1：``prepare_video`` 必须在槽内、且提交到**专属**执行器。

    两条一起钉：

    1. 每次提交都在 ``async with limiter.slot()`` 里 —— 否则取消时没有槽可以还，
       真实并发会越过 ``limit``（VERIFY-A P2-1）；
    2. 提交必须走 ``slots.submit``（``core.limits.SlotLimiter`` 的专属执行器）。
       **不许**再出现 ``asyncio.to_thread(prepare_video, …)``：它排的是进程级
       默认执行器，池满时任务在池队列里无限期排队，而槽位已被取走 ——
       实测 20 线程池占满时视频请求 **>=6.0s 未完成**（VERIFY-VV-R9 P1-1）。

    绝不能把 ``threading.Semaphore`` 塞进 ``prepare_video``：默认执行器是全进程
    共享的（2 核只有 6 个线程），在那里阻塞会饿死核心的 media / TTS / ASR。
    """
    off_slot: list[str] = []
    not_dedicated: list[str] = []
    checked = 0
    for name in ("hooks/llm_request.py", "main.py"):
        tree = ast.parse((REPO / name).read_text(encoding="utf-8"))
        inside = _slot_context_nodes(tree)
        total = 0
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call) or not node.args:
                continue
            first = node.args[0]
            if not (isinstance(first, ast.Name) and first.id == "prepare_video"):
                continue
            total += 1
            func = node.func
            attr = (
                func.attr
                if isinstance(func, ast.Attribute)
                else getattr(func, "id", "")
            )
            if id(node) not in inside:
                off_slot.append(f"{name}:{node.lineno} {attr}(prepare_video, …)")
            if attr != "submit":
                not_dedicated.append(
                    f"{name}:{node.lineno} {attr}(prepare_video, …)"
                    " —— 必须走 slots.submit（ffmpeg 专属执行器）"
                )
        checked += total
    assert checked >= 2, f"没找到预期的调用点（只找到 {checked} 处）"
    assert off_slot == [], off_slot
    assert not_dedicated == [], not_dedicated


def _code_string_literals(path: Path) -> list[str]:
    """源码里的字符串字面量（**跳过 docstring**：说明文字不算实现）。"""
    tree = ast.parse(path.read_text(encoding="utf-8"))
    skip: set[int] = set()
    for node in ast.walk(tree):
        body = getattr(node, "body", None)
        if not isinstance(body, list) or not body:
            continue
        first = body[0]
        if (
            isinstance(first, ast.Expr)
            and isinstance(first.value, ast.Constant)
            and isinstance(first.value.value, str)
        ):
            skip.add(id(first.value))
    return [
        node.value
        for node in ast.walk(tree)
        if isinstance(node, ast.Constant)
        and isinstance(node.value, str)
        and id(node) not in skip
    ]


def test_limits_module_stays_free_of_threading_blocking_for_the_caller():
    """结构：``limits.py`` 只用标准库，且**不 import astrbot**（``core/`` 契约）。

    ``concurrent`` / ``contextvars`` / ``functools`` 是 P1-1 的专属执行器要用的
    （``SlotLimiter.submit``：与 ``asyncio.to_thread`` 逐字等价地转交阻塞函数，
    只是换成 ffmpeg 自己的池），不是"顺手多用几个库"。
    """
    path = REPO / "core" / "limits.py"
    tree = ast.parse(path.read_text(encoding="utf-8"))
    imported: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.level == 0:
            imported.add((node.module or "").split(".")[0])
    assert "astrbot" not in imported
    assert imported <= {
        "__future__",
        "asyncio",
        "collections",
        "concurrent",
        "contextvars",
        "dataclasses",
        "functools",
        "threading",
        "time",
        "typing",
    }, sorted(imported)

    # 不许出现任何"在 worker 里阻塞"的原语；也不许用 time.sleep 排队
    literals = "\n".join(_code_string_literals(path))
    for forbidden in ("threading.Semaphore", "time.sleep"):
        assert forbidden not in literals, forbidden
    # 不许真的调用 Semaphore
    semaphore_calls = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and (
            (isinstance(node.func, ast.Attribute) and node.func.attr == "Semaphore")
            or (isinstance(node.func, ast.Name) and node.func.id == "Semaphore")
        )
    ]
    assert semaphore_calls == [], "限流只能用 asyncio.Queue 令牌，不许用 Semaphore"
    # 交给执行器的**只许**是专属池：``run_in_executor(None, …)`` 会退回进程级
    # 默认池，那正是 P1-1（池满时任务排在池队列里、槽位被未启动的 worker 占住）
    executor_calls = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "run_in_executor"
    ]
    assert len(executor_calls) == 1, [node.lineno for node in executor_calls]
    only = executor_calls[0]
    assert only.args, "run_in_executor 必须显式给出执行器（不许 None）"
    target = only.args[0]
    assert (
        isinstance(target, ast.Call)
        and isinstance(target.func, ast.Attribute)
        and target.func.attr == "_pool"
    ), f"只许跑在专属执行器上，实际第一个参数是 {ast.dump(target)}"



def test_queue_info_is_at_most_one_line_per_video():
    """每个视频**最多 1 条**排队 INFO（AUDIT-030 §C.1「日志不许刷屏」）。

    4 个任务、limit=2、时钟一次性推进 3s：前两个立即拿到槽（waited=0），
    后两个真的排队（waited=3s）—— 于是恰好 2 条 INFO，一条不少也一条不多。
    ``debug=False`` 保证没有额外的 debug 行来"凑数"。
    """

    async def scenario() -> _Recorder:
        recorder = _Recorder()
        clock = _Clock()
        limiter = SlotLimiter(2, logger=recorder, clock=clock)
        release = asyncio.Event()

        async def holder(index: int) -> None:
            async with limiter.slot(name=f"v{index}"):
                await release.wait()

        holders = [asyncio.create_task(holder(i)) for i in range(2)]
        await _spin(lambda: limiter.inflight == 2)

        waiters = [asyncio.create_task(holder(i)) for i in range(2, 4)]
        await _spin(lambda: limiter.waiters == 2)
        clock.now = 3.0
        release.set()
        await asyncio.wait_for(asyncio.gather(*holders, *waiters), timeout=5)
        return recorder

    recorder = asyncio.run(scenario())
    queued = [item for item in recorder.messages("info") if "ffmpeg 排队" in item]
    assert len(queued) == 2, recorder.records  # 只有真的等过的那两个
    assert all("3.0s" in item and "limit=2" in item for item in queued), queued
    # 每个视频最多一条：4 次取槽，INFO 只可能出现在等过的那两次上
    assert len(recorder.records) == 2, recorder.records


def test_null_limiter_never_builds_a_queue_and_logs_nothing():
    """``limit=0`` 的空实现：不建令牌队列、不动 inflight、不打任何日志。

    这是"``ffmpeg_max_concurrency=0`` 逐字节等价 0.2.1"最直接的结构证据 ——
    没有队列就没有 ``await``，也就没有排队。
    """

    recorder = _Recorder()

    async def scenario() -> None:
        limiter = SlotLimiter(0, logger=recorder, debug=True)
        assert limiter._queue is None
        async with limiter.slot(name="v"):
            assert limiter.inflight == 0  # 不计入在飞
            assert limiter.available == 0  # 也没有令牌可数
        assert limiter._queue is None, "关闭限流时不许建 asyncio.Queue"
        assert limiter.stats.waits == 0
        assert limiter.stats.peak_inflight == 0

    asyncio.run(scenario())
    assert recorder.records == []
    # 模块级单例同样必须是"未建队列"的状态（钩子在没有 limiter 时用它）
    from vv_plugin.core.limits import NULL_LIMITER

    assert NULL_LIMITER.limit == 0
    assert NULL_LIMITER._queue is None
