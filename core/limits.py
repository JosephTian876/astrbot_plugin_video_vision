"""并发限流（async 令牌）与 singleflight（线程事件）—— 纯标准库，零 AstrBot 依赖。

为什么排队必须在 **async 层**（AUDIT-030 §C.1-1 / §E-3）
-------------------------------------------------------
``asyncio.to_thread`` 的实现是 ``loop.run_in_executor(None, …)``，也就是
**全进程共享**的默认 ``ThreadPoolExecutor``，大小是 ``min(32, cpu+4)``
（2 核机器 = 6 个线程；16 核 = 20）。AstrBot 核心自己就在这个池里跑图片/音视频
转码（``core/utils/media_utils.py``）、TTS、自托管 ASR 和工具调用。

因此这里的等待只用 ``asyncio.Queue`` 令牌：2 核机器上 6 个线程全都在**干活**，
而不是 4 个在 worker 里"睡觉"、把核心功能一起饿死。**禁止**把
``threading.Semaphore`` 塞进 ``prepare_video`` —— 那等于把默认线程池当成自己的
私有池用（实测 2 核 = 6 个线程，核心的 media/TTS/ASR 全在里面）。

为什么不用 ``asyncio.Semaphore``
--------------------------------
1. ``Queue.get()`` 被取消/超时打断时不会**吞掉**令牌：令牌只在 ``get_nowait()``
   里出队，队列里的元素先于"唤醒信号"存在，因此令牌要么还在队列里、要么已经
   真的交到某个调用方手上（后者由 :meth:`SlotLimiter._reclaim` 还回去）；
2. 顺手能报"前方还有几个"给日志（``waiters``）；
3. 本项目要支持 3.10-3.13：``Queue`` 的语义跨版本变化比 ``Semaphore`` 少。

为什么 singleflight 只用 ``threading.Event``
--------------------------------------------
``prepare_video`` 跑在 worker 线程里；在 worker 里碰 ``asyncio.Future`` /
``asyncio.Semaphore`` 会 ``attached to a different loop``（测试里 ``asyncio.run``
反复重建循环时必炸）。这里全是 ``threading.Lock`` / ``threading.Event``，
与事件循环无关。
"""

from __future__ import annotations

import asyncio
import threading
import time
from collections.abc import Hashable
from dataclasses import dataclass
from typing import Any, Awaitable, Callable

#: leader 的失败里，这些是**确定性**的：重试只会把同一个 `max_compress_seconds`
#: 超时乘以 N，follower 必须原样承接失败（立即返回，不浪费时间）。
#: 其余（``timeout`` / ``exit_code`` / ``other``）属瞬时失败，允许 follower
#: **自己接管一次**，避免一次网络/磁盘抖动波及同一文件的 N 个用户。
DETERMINISTIC_FAILURE_KINDS: frozenset[str] = frozenset(
    {"not_found", "duration", "corrupt", "empty", "too_large"}
)

#: 排队达到这个秒数才打 INFO（每个视频最多一条 —— 不许在成功路径上刷屏）
QUEUE_LOG_THRESHOLD_SEC = 1.0

#: 队列里的令牌（内容无意义，只表示"一个空闲槽位"）
_TOKEN = object()


class QueueTimeout(Exception):
    """等待 ffmpeg 槽位超时 —— **不是** ffmpeg 自己超时。

    两者必须可区分：``timeout`` 会劝用户去调 ``max_compress_seconds``，
    而这里真正该调的是 ``ffmpeg_max_concurrency`` / ``ffmpeg_queue_timeout_sec``。
    """

    def __init__(self, waited: float, timeout: float, limit: int, ahead: int) -> None:
        self.waited = waited
        self.timeout = timeout
        self.limit = limit
        self.ahead = ahead
        super().__init__(
            f"排队等待转码超过 {timeout:.0f}s（limit={limit}，前方 {ahead} 个）"
        )


@dataclass
class LimiterStats:
    """限流器的可观测计数（只用于 DEBUG 日志与单测，不进 ``/videotest``）。"""

    #: 真的排过队的次数（立即拿到槽不计）
    waits: int = 0
    waited_sec_total: float = 0.0
    max_wait_sec: float = 0.0
    timeouts: int = 0
    peak_inflight: int = 0


class SlotLimiter:
    """进程内**全局**（跨会话）的 ffmpeg 槽位限流器。

    ``limit <= 0`` 表示**关闭限流**：不建队列、不加任何 await、不打任何日志 ——
    逐字节等价 0.2.1（默认线程池仍是唯一的上限来源）。

    队列按事件循环惰性重建：``asyncio.Queue`` 是 loop 绑定的，而测试（以及
    插件的重载）会反复 ``asyncio.run``。换 loop 时重建一份填满 ``limit`` 个令牌
    的队列，``inflight`` 归零。
    """

    def __init__(
        self,
        limit: int,
        *,
        queue_timeout: float = 30.0,
        logger: Any | None = None,
        debug: bool = False,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self.limit = max(0, int(limit))
        self.queue_timeout = max(0.0, float(queue_timeout))
        self.stats = LimiterStats()
        self._logger = logger
        self._debug = bool(debug)
        self._clock = clock
        self._lock = threading.Lock()
        self._loop: asyncio.AbstractEventLoop | None = None
        self._queue: asyncio.Queue[Any] | None = None
        self._waiters = 0
        self._inflight = 0

    # ------------------------------------------------------------ 只读状态

    @property
    def enabled(self) -> bool:
        """``limit > 0`` 才真的限流。"""
        return self.limit > 0

    @property
    def inflight(self) -> int:
        """当前占用槽位的调用数。"""
        return self._inflight

    @property
    def available(self) -> int:
        """队列里还剩几个空闲令牌（``available + inflight == limit``）。"""
        queue = self._queue
        return 0 if queue is None else queue.qsize()

    @property
    def waiters(self) -> int:
        """正在排队等槽位的调用数。"""
        return self._waiters

    # ---------------------------------------------------------------- 取还槽

    def slot(self, name: str = "") -> _Slot:
        """返回一个 ``async with`` 上下文管理器（``name`` 只用于日志）。"""
        return _Slot(self, name)

    async def _acquire(self, name: str) -> None:
        if not self.enabled:
            return
        queue = self._queue_for(asyncio.get_running_loop())
        started = self._clock()
        ahead = 0
        try:
            queue.get_nowait()
        except asyncio.QueueEmpty:
            ahead = self._waiters
            self._waiters += 1
            try:
                await self._await_token(queue, ahead)
            finally:
                self._waiters -= 1
            self.stats.waits += 1
        waited = self._clock() - started
        self._inflight += 1
        self.stats.waited_sec_total += waited
        if waited > self.stats.max_wait_sec:
            self.stats.max_wait_sec = waited
        if self._inflight > self.stats.peak_inflight:
            self.stats.peak_inflight = self._inflight
        self._log_acquisition(waited, ahead)

    def _release(self) -> None:
        if self._inflight > 0:
            self._inflight -= 1
        queue = self._queue
        if queue is None:
            return
        try:
            queue.put_nowait(_TOKEN)
        except asyncio.QueueFull:  # 不该发生：只有归还会放令牌
            pass

    # ---------------------------------------------------------------- 内部

    def _queue_for(
        self, loop: asyncio.AbstractEventLoop
    ) -> asyncio.Queue[Any]:
        """取当前事件循环的令牌队列（换 loop 时重建）。"""
        queue = self._queue
        if queue is not None and self._loop is loop:
            return queue
        with self._lock:
            if self._queue is not None and self._loop is loop:
                return self._queue
            queue = asyncio.Queue(maxsize=self.limit)
            for _ in range(self.limit):
                queue.put_nowait(_TOKEN)
            self._queue = queue
            self._loop = loop
            self._inflight = 0
            self._waiters = 0
            return queue

    async def _await_token(self, queue: asyncio.Queue[Any], ahead: int) -> None:
        """等一个令牌；超时抛 :class:`QueueTimeout`，取消时令牌不丢。"""
        started = self._clock()
        getter = asyncio.ensure_future(queue.get())
        try:
            await asyncio.wait_for(getter, timeout=self.queue_timeout)
        except asyncio.CancelledError:
            self._reclaim(getter, queue)
            raise
        except (asyncio.TimeoutError, TimeoutError):
            self._reclaim(getter, queue)
            self.stats.timeouts += 1
            waited = self._clock() - started
            if self._logger is not None:
                self._logger.warning(
                    "[VideoVision] ffmpeg 排队超时（%.0fs，limit=%d），本视频跳过",
                    self.queue_timeout,
                    self.limit,
                )
            raise QueueTimeout(
                waited=waited,
                timeout=self.queue_timeout,
                limit=self.limit,
                ahead=ahead,
            ) from None

    @staticmethod
    def _reclaim(getter: asyncio.Future[Any], queue: asyncio.Queue[Any]) -> None:
        """超时/取消时，若令牌**已经真的**交到手上就必须还回队列。

        不还说会怎样：槽位永久 -1，重复几次之后全站卡死（RISK #4）。这也是
        ``wait_for`` + ``ensure_future`` 这套写法的唯一理由 —— 我们要能检查
        "那个 future 到底拿到令牌没有"。
        """
        try:
            if getter.done() and not getter.cancelled() and getter.exception() is None:
                queue.put_nowait(getter.result())
                return
            if not getter.done():
                getter.cancel()
        except (asyncio.QueueFull, asyncio.CancelledError):
            pass
        except Exception:  # noqa: BLE001 - 兜底：任何异常都不能掩盖真正的超时
            pass

    def _log_acquisition(self, waited: float, ahead: int) -> None:
        log = self._logger
        if log is None:
            return
        if waited >= QUEUE_LOG_THRESHOLD_SEC:
            log.info(
                "[VideoVision] ffmpeg 排队 %.1fs（limit=%d，前方 %d 个）",
                waited,
                self.limit,
                ahead,
            )
        elif self._debug:
            # 措辞必须诚实：等了 0.3s 不是"立即获取"（只在 <1.0s 时走到这里）
            log.debug(
                "[VideoVision][debug] ffmpeg 槽位获取（等待 %.2fs，inflight=%d）",
                waited,
                self._inflight,
            )


class _Slot:
    """``SlotLimiter.slot()`` 的返回值：保证进出成对。"""

    __slots__ = ("_handed_off", "_held", "_limiter", "_name")

    def __init__(self, limiter: SlotLimiter, name: str) -> None:
        self._limiter = limiter
        self._name = name
        self._held = False
        #: 还槽已移交给 worker 的完成回调（见 :meth:`wait_for_worker`）
        self._handed_off = False

    async def __aenter__(self) -> _Slot:
        await self._limiter._acquire(self._name)
        self._held = True
        return self

    async def __aexit__(self, exc_type: Any, exc: Any, tb: Any) -> bool:
        if self._held and not self._handed_off:
            self._held = False
            self._limiter._release()
        return False

    async def wait_for_worker(self, worker: Awaitable[Any]) -> Any:
        """等 worker 结束；本任务被取消时，槽位**留在 worker 手里**。

        ``asyncio.to_thread`` 的线程不可取消：协程被 cancel 时 ``__aexit__``
        会当场还令牌，而 ffmpeg 子进程还在跑（最长 ``max_compress_seconds``），
        真实并发于是可以越过 ``limit``（VERIFY-A P2-1：``limit=1`` 实测峰值 2）。

        所以取消路径把还槽移交给 worker 的 done-callback —— 它在事件循环线程里
        于 worker 真正结束（成功/失败/异常）之后**恰好**执行一次：既不会提前还，
        也不会漏还。``__aexit__`` 看到 ``_handed_off`` 就不再还第二次。
        （一个槽位只对应一个 worker：同一个 ``async with`` 里别调两次。）

        ``limit=0``（关闭限流）时直接 ``await``：不建 task、不 shield，
        与 0.2.1 逐字节等价。
        """
        if not self._limiter.enabled:
            return await worker
        task = asyncio.ensure_future(worker)
        try:
            return await asyncio.shield(task)
        except asyncio.CancelledError:
            self._hand_off(task)
            raise

    def _hand_off(self, task: asyncio.Future[Any]) -> None:
        """把还槽移交给 ``task`` 的完成回调（幂等：只移交一次）。"""
        if self._handed_off or not self._held:
            return
        self._handed_off = True
        task.add_done_callback(self._release_after_worker)

    def _release_after_worker(self, task: asyncio.Future[Any]) -> None:
        """worker 真正结束时还槽（done-callback 跑在事件循环线程里）。"""
        if not task.cancelled():
            task.exception()  # 取走异常：已经没人 await 它，别让 asyncio 报噪音
        self._held = False
        self._limiter._release()


@dataclass
class _Flight:
    """一次在飞的共享工作。"""

    event: threading.Event
    error: BaseException | None = None

    @classmethod
    def new(cls) -> _Flight:
        return cls(event=threading.Event())


def failure_kind(error: BaseException | None) -> str:
    """从异常上取机器可读分类；没有分类的按 ``other``（瞬时）处理。"""
    if error is None:
        return ""
    return str(getattr(error, "kind", "") or "other")


class SingleFlight:
    """同一 key 的并发调用只让一个真跑，其余等它把产物发布到共享位置。

    契约（AUDIT-030 §C.1-8）：

    * leader 成功 -> follower 醒来后用 ``probe()`` 取**同一个产物**
      （``probe`` 就是 ``cache.get``；它拿不到时才会自己重跑）；
    * leader 抛确定性失败（``not_found`` / ``duration`` / ``corrupt`` /
      ``empty`` / ``too_large``）-> follower 原样抛出**同一个异常对象**；
    * leader 抛瞬时失败（``timeout`` / ``exit_code`` / ``other``）或 follower
      等待超时 -> follower **自己接管一次**（成为新 leader），最多一次；
    * 无论成功失败，``finally`` 里摘掉的永远是**自己那条** flight。
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._flights: dict[Hashable, _Flight] = {}

    @property
    def pending(self) -> int:
        """在飞的 flight 数（正常情况下任何时刻之后都必须回到 0）。"""
        with self._lock:
            return len(self._flights)

    def run(
        self,
        key: Hashable,
        work: Callable[[], Any],
        *,
        probe: Callable[[], Any] | None = None,
        wait_timeout: float = 60.0,
        on_join: Callable[[], None] | None = None,
        on_takeover: Callable[[BaseException | None], None] | None = None,
    ) -> Any:
        """执行 ``work``（同 key 只一个真跑）；返回 leader 或缓存里的产物。"""
        timeout = max(0.0, float(wait_timeout))
        attempts = 0
        while True:
            flight, is_leader = self._enter(key)
            if is_leader:
                return self._lead(key, flight, work)

            if on_join is not None:
                on_join()
            finished = flight.event.wait(timeout)

            if finished and flight.error is None:
                if probe is not None:
                    value = probe()
                    if value is not None:
                        return value
                # 产物没进共享位置（例如超过 cache_max_mb）：没有可共享的东西
            elif finished:
                kind = failure_kind(flight.error)
                if kind in DETERMINISTIC_FAILURE_KINDS:
                    raise flight.error
                if on_takeover is not None:
                    on_takeover(flight.error)

            if attempts >= 1:
                # 第二次机会也不成：自己跑，且**不再**登记 flight ——
                # 这时行为等价于没有 singleflight，绝不会无限重试。
                return work()
            attempts += 1
            self._discard(key, flight)

    # ---------------------------------------------------------------- 内部

    def _enter(self, key: Hashable) -> tuple[_Flight, bool]:
        with self._lock:
            flight = self._flights.get(key)
            if flight is None:
                flight = _Flight.new()
                self._flights[key] = flight
                return flight, True
            return flight, False

    def _lead(
        self, key: Hashable, flight: _Flight, work: Callable[[], Any]
    ) -> Any:
        try:
            return work()
        except BaseException as exc:  # noqa: BLE001 - leader 的失败必须原样传给 follower
            flight.error = exc
            raise
        finally:
            flight.event.set()
            self._discard(key, flight)

    def _discard(self, key: Hashable, flight: _Flight) -> None:
        """只摘自己那条：防止把后登记的同 key flight 删掉。"""
        with self._lock:
            if self._flights.get(key) is flight:
                del self._flights[key]


__all__ = [
    "DETERMINISTIC_FAILURE_KINDS",
    "NULL_LIMITER",
    "QUEUE_LOG_THRESHOLD_SEC",
    "LimiterStats",
    "QueueTimeout",
    "SingleFlight",
    "SlotLimiter",
    "failure_kind",
]


#: 关闭限流时用的**空实现**（``limit=0``）：不建队列、不等待、不打日志。
#:
#: 它让调用方可以**无条件**写 ``async with limiter.slot(…)`` —— 于是
#: "每个 ``to_thread(prepare_video)`` 都在槽内"这条结构约束能被 AST 锁死
#: （``tests/test_limits.py``）。``async with`` 一个立即返回的 ``__aenter__``
#: 不会让出事件循环，因此"关闭限流"仍然逐字节等价 0.2.1。
NULL_LIMITER = SlotLimiter(0)
