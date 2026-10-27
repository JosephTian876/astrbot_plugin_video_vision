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

为什么 ffmpeg 用**专属**执行器（VERIFY-VV-R9 P1-1）
--------------------------------------------------
上面那条禁令说的是"**不许一边占用默认池的线程、一边等槽**"，不是"必须用默认池"。
恰恰相反：把 ``prepare_video`` 交给默认池有一个致命的排队状态 —— 池满时任务只是
躺在**池队列**里，而调用方已经先取走了槽位，于是

* 当前请求无限挂起（:meth:`_Slot.wait_for_worker` 对 worker 的等待没有超时）；
* 取消也还不回槽位：还槽挂在 worker 的 done-callback 上，而"还没启动的 worker"
  永远不会有 done —— 一个并不存在的并发把槽位占住。

实测（AUDIT-VV-R9 §1.5）：20 线程池占满时视频请求 **>=6.0s 未完成**；
6 线程池（2 核等价）**58.36s**（靠外部 blocker 到期）才结束。

:meth:`SlotLimiter.submit` 因此把阻塞工作交给**专属** ``ThreadPoolExecutor``，
尺寸就是 ``limit``：限流器保证同一时刻最多 ``limit`` 个 worker 在跑（取槽 ->
提交 -> worker 结束才还槽），于是"拿到槽就一定有线程"，池队列在结构上为空。
默认池被核心的 media / TTS / ASR 占满与我们无关，我们也不去抢它。

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
import contextvars
import functools
import threading
import time
from collections.abc import Hashable
from concurrent.futures import ThreadPoolExecutor
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

#: ffmpeg 专属执行器的线程名前缀（测试据此断言"没跑在默认池里"）
EXECUTOR_THREAD_PREFIX = "vv-ffmpeg"

#: 队列里的令牌（内容无意义，只表示"一个空闲槽位"）
_TOKEN = object()


def _name_suffix(name: str) -> str:
    """日志里的视频名后缀（空名不打印，避免尾巴上挂一个空括号）。"""
    text = (name or "").strip()
    return f"，视频={text}" if text else ""


def format_seconds(value: float) -> str:
    """把秒数渲染成人读文本 —— 小于 1 秒时**绝不**写成「0s」。

    ``%.0f`` 会把 ``queue_timeout=0.01`` 渲染成「排队等待转码超过 0s」，把还剩
    0.9s 的 TTL 渲染成「TTL 剩 0s」（VERIFY-A P3-2 / VERIFY-B P3）。两处都在说
    "没有等待"和"马上过期"，而实际是"等了 10 毫秒""还剩 900 毫秒"。所以：
    ``>=60s`` 整数秒、``>=1s`` 一位小数、``>0s`` 毫秒（至少 1ms），只有 0 才是
    ``0s``。

    Args:
        value: 秒数（负数按 0 处理）。

    Returns:
        ``"600s"`` / ``"30.0s"`` / ``"2.5s"`` / ``"900ms"`` / ``"1ms"`` / ``"0s"``。
    """
    seconds = max(0.0, float(value))
    if seconds >= 60.0:
        return f"{seconds:.0f}s"
    if seconds >= 1.0:
        return f"{seconds:.1f}s"
    if seconds > 0.0:
        return f"{max(1.0, seconds * 1000.0):.0f}ms"
    return "0s"


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
        #: 超时预算的人读文本（``"30s"`` / ``"10ms"``）：文案只有这一份真相，
        #: 钩子与 ``/videotest`` 都走 :meth:`short_reason`，不各写一遍格式串。
        self.timeout_text = format_seconds(timeout)
        super().__init__(
            f"排队等待转码超过 {self.timeout_text}（limit={limit}，前方 {ahead} 个）"
        )

    def short_reason(self) -> str:
        """注入给模型 / ``/videotest`` 的短原因（不含「前方 N 个」这类现场细节）。"""
        return f"排队等待转码超过 {self.timeout_text}（limit={self.limit}）"


@dataclass
class LimiterStats:
    """限流器的可观测计数（DEBUG 日志 + ``/videotest`` 的一行摘要）。"""

    #: 真的排过队的次数（立即拿到槽不计）
    waits: int = 0
    waited_sec_total: float = 0.0
    max_wait_sec: float = 0.0
    timeouts: int = 0
    peak_inflight: int = 0

    def summary(self) -> str:
        """``/videotest`` 里的那一行摘要；**没有任何排队/超时**时返回空串。

        正常路径不新增任何行（报告格式保持原样），只有运维真的需要看的时候
        （有人在排队、或有人被排队超时拒掉）才多一行。
        """
        if self.waits <= 0 and self.timeouts <= 0:
            return ""
        if self.timeouts and self.waits <= 0:
            # 只有"排队超时"、没有一次排队成功时，别渲染成
            # 「排队 0 次，最长 0ms，累计 0ms，超时 1 次」——那三个 0 与后面的
            # 「超时 1 次」自相矛盾（VERIFY-VV-R7 P2-3 实测复现）。
            return f"超时 {self.timeouts} 次（没有一次排队成功）"
        parts = [
            f"排队 {self.waits} 次",
            f"最长 {format_seconds(self.max_wait_sec)}",
            f"累计 {format_seconds(self.waited_sec_total)}",
        ]
        if self.timeouts:
            parts.append(f"超时 {self.timeouts} 次")
        return "，".join(parts)


class SlotLimiter:
    """进程内**全局**（跨会话）的 ffmpeg 槽位限流器 + **专属**执行器。

    ``limit <= 0`` 表示**关闭限流**：不建队列、不加任何 await、不打任何日志，
    也不建专属执行器 —— 逐字节等价 0.2.1（默认线程池仍是唯一的上限来源）。

    队列按事件循环惰性重建：``asyncio.Queue`` 是 loop 绑定的，而测试（以及
    插件的重载）会反复 ``asyncio.run``。换 loop 时重建一份填满 ``limit`` 个令牌
    的队列，``inflight`` 归零。

    专属执行器同理**惰性**创建（在 :meth:`submit` 里），尺寸 == ``limit``，
    :meth:`shutdown` 收掉 —— 见模块说明里"为什么 ffmpeg 用专属执行器"。
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
        #: 专属执行器（惰性创建；``limit<=0`` 时永远是 ``None``）
        self._executor: ThreadPoolExecutor | None = None
        self._pool_lock = threading.Lock()
        #: :meth:`shutdown` 之后不再接受新任务（退回默认池，见 :meth:`submit`）
        self._closed = False

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
                await self._await_token(queue, ahead, name)
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
        self._log_acquisition(waited, ahead, name)

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

    # -------------------------------------------------------- 专属执行器

    def submit(self, fn: Callable[..., Any], /, *args: Any, **kwargs: Any) -> Any:
        """把**阻塞**函数排到 ffmpeg 专属执行器，返回可直接 ``await`` 的对象。

        为什么不 ``asyncio.to_thread``（AUDIT-VV-R9 §1.5 实测）：它排的是
        **进程级**默认执行器，池满时任务只是躺在**池队列**里，而槽位已经被取走
        —— 当前请求无限挂起，取消也还不回槽位（还槽挂在 worker 的 done-callback
        上，而还没启动的 worker 永远没有 done）。

        这里排到专属池（``max_workers == limit``）。因为限流器保证同一时刻最多
        ``limit`` 个 worker 在跑（取槽 -> 提交 -> worker 结束才还槽），
        "拿到槽就一定有线程"，池队列在结构上为空。

        ``limit <= 0``（关闭限流）或已经 :meth:`shutdown` 时退回
        ``asyncio.to_thread``：前者就是 0.2.1 的行为，后者是为了重载途中的
        在飞钩子不至于挂死。

        Args:
            fn: 要在线程里跑的**同步**函数。
            *args: 位置参数。
            **kwargs: 关键字参数。

        Returns:
            ``asyncio.Future``（专属池）或 coroutine（退回默认池），都可 ``await``。
        """
        if not self.enabled or self._closed:
            return asyncio.to_thread(fn, *args, **kwargs)
        loop = asyncio.get_running_loop()
        # 与 ``asyncio.to_thread`` 逐字等价（含 contextvars 传递），只换执行器
        call = functools.partial(contextvars.copy_context().run, fn, *args, **kwargs)
        try:
            return loop.run_in_executor(self._pool(), call)
        except RuntimeError:
            # shutdown 正在发生（重载途中的在飞钩子）：退回默认池，绝不挂起
            return asyncio.to_thread(fn, *args, **kwargs)

    def _pool(self) -> ThreadPoolExecutor:
        """取专属执行器（惰性创建：没处理过视频就不建线程）。"""
        pool = self._executor
        if pool is not None:
            return pool
        with self._pool_lock:
            if self._executor is None:
                self._executor = ThreadPoolExecutor(
                    max_workers=self.limit,
                    thread_name_prefix=EXECUTOR_THREAD_PREFIX,
                )
            return self._executor

    def shutdown(self, *, wait: bool = False) -> None:
        """收掉专属执行器（``main.terminate`` 调用）：不打断在飞的 worker。

        线程只在第一次 :meth:`submit` 时创建，所以"没处理过视频"的实例没有线程
        可收；这里只把已经建出来的那些收掉，插件重载不会留下一个永不回收的池。

        Args:
            wait: ``True`` 时等 worker 跑完。默认 ``False``：ffmpeg 子进程有自己
                的超时，卸载流程不该跟着等（在飞的 worker 会跑完自己那一次）。
        """
        with self._pool_lock:
            pool, self._executor = self._executor, None
            self._closed = True
        if pool is not None:
            pool.shutdown(wait=wait)

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

    async def _await_token(
        self, queue: asyncio.Queue[Any], ahead: int, name: str = ""
    ) -> None:
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
                    "[VideoVision] ffmpeg 排队超时（%s，limit=%d），本视频跳过%s",
                    format_seconds(self.queue_timeout),
                    self.limit,
                    _name_suffix(name),
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

    def _log_acquisition(self, waited: float, ahead: int, name: str = "") -> None:
        """排队日志：**带上视频名** —— 否则线上无法判断是哪条视频排了 28 秒。

        ``slot(name=…)`` 一路传到这里，是这条日志唯一的定位信息（VERIFY-A P3-1）。
        """
        log = self._logger
        if log is None:
            return
        if waited >= QUEUE_LOG_THRESHOLD_SEC:
            log.info(
                "[VideoVision] ffmpeg 排队 %.1fs（limit=%d，前方 %d 个）%s",
                waited,
                self.limit,
                ahead,
                _name_suffix(name),
            )
        elif self._debug:
            # 措辞必须诚实：等了 0.3s 不是"立即获取"（只在 <1.0s 时走到这里）
            log.debug(
                "[VideoVision][debug] ffmpeg 槽位获取（等待 %.2fs，inflight=%d）%s",
                waited,
                self._inflight,
                _name_suffix(name),
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

        worker（``SlotLimiter.submit`` 排到专属执行器 / 默认池的那个线程）不可
        取消：协程被 cancel 时 ``__aexit__`` 会当场还令牌，而 ffmpeg 子进程还在跑
        （最长 ``max_compress_seconds``），真实并发于是可以越过 ``limit``
        （VERIFY-A P2-1：``limit=1`` 实测峰值 2）。

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
    "format_seconds",
]


#: 关闭限流时用的**空实现**（``limit=0``）：不建队列、不等待、不打日志。
#:
#: 它让调用方可以**无条件**写 ``async with limiter.slot(…)`` —— 于是
#: "每个 ``to_thread(prepare_video)`` 都在槽内"这条结构约束能被 AST 锁死
#: （``tests/test_limits.py``）。``async with`` 一个立即返回的 ``__aenter__``
#: 不会让出事件循环，因此"关闭限流"仍然逐字节等价 0.2.1。
NULL_LIMITER = SlotLimiter(0)
