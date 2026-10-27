"""视觉理解结果的进程内缓存（TTL + LRU）—— 纯标准库，零 AstrBot 依赖。

它解决的是**哪一半**问题（``AUDIT-030`` §B.2）
------------------------------------------------
"群友对同一视频连续追问，每次都要重调视觉接口"——这一半**不成立**：追问时如果
不重发/不引用视频，插件的钩子根本不会运行（``req.extra_user_content_parts`` 是
每个回合现搭的，历史里既没有视频占位符，也没有注入的分析文本，后者被
``mark_as_temp()`` 在落库时丢弃）。而"追问时重新引用视频"通常**换了提问** ——
那时缓存命中才是 bug（用 A 问题的答案回答 B 问题）。

成立的是另一半：**完全相同**的重复投递（同一份视频字节 + 同一个最终 prompt +
同一模型 + 同一端点 + 同一 ``max_tokens``）在 TTL 内重复发生时，会各花一次
60-120s 的上传 + 推理。省掉这一次是真实收益，且不需要落盘、不需要新依赖。

key 为什么是"规范化 JSON 的 sha256"而不是 ``"|".join(...)``
-----------------------------------------------------------
六个维度里 ``prompt`` 是**用户可控文本**，可能含 ``|``：朴素拼接会让
``video="a|b", prompt="c"`` 与 ``video="a", prompt="b|c"`` 拼出同一个字符串
（``tests/test_vision_cache.py::test_key_separates_field_boundaries`` 锁住这一点）。
``json.dumps(..., sort_keys=True, ensure_ascii=False)`` 给每个字段加了转义边界，
同时 ``KEY_VERSION`` 让将来 payload 结构变化时旧条目自动失效。

明确**不**进 key 的东西：``openai_api_key``（换密钥不改变语义）、
``openai_timeout_sec``、``vision_verify_ssl``、视频文件名（QQ 每次落地名都不同，
放进去等于永不命中）、会话 ID（跨会话共享正是收益来源）。

``video`` 这一维：**"源 + 参数"，不是产物字节**
------------------------------------------------
压缩/截断过的视频，这一维是 ``<action>:<压缩签名>``，签名 = 源内容哈希 +
压缩参数（``compress_max_mb`` / ``crf`` / ``preset`` / ``truncate_to``），与
**压缩缓存 key 完全同源**（``core/injector.py`` 的 ``compress_signature``）。

为什么不用"真正发出去的字节"的哈希：产物**不可复现** —— 同一源 + 同一组参数
每次压出来的 mp4 都不一样（熵编码 + 封装时间戳 + ``-fs`` 在 GOP 边界截断），
实测 3 次同源同参 = 3 个不同的产物哈希。取产物哈希会让
``vision_cache_enabled=true`` + ``cache_enabled=false`` 时**永不命中**
（ARCH-VV-C2 P1-1）。"源 + 参数"决定模型看到的画面，既稳定、又能在压缩之前
算出来；直通（as-is）视频的签名就是源文件内容哈希。
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import threading
import time
from collections import OrderedDict
from collections.abc import Awaitable
from dataclasses import dataclass
from typing import Any, Callable

#: key 的载荷版本。任何"让旧条目语义失效"的改动（例如换发送形态）都必须 +1。
KEY_VERSION = 1


@dataclass(frozen=True)
class CacheHit:
    """一次命中：分析原文 + 剩余 TTL（秒，用于日志，不用于续期）。"""

    analysis: str
    ttl_left: float


@dataclass
class VisionCacheStats:
    """可观测计数（只用于 DEBUG 日志与单测；``/videotest`` 不渲染它们）。"""

    hits: int = 0
    misses: int = 0
    stores: int = 0
    evictions: int = 0
    expirations: int = 0


#: ``work()`` 返回这个哨兵表示"没有结果"（与 ``None`` 区分开）
_MISSING = object()


@dataclass
class _Flight:
    """一次在飞的异步视觉请求（同 key 的后来者等它的结果）。"""

    loop: asyncio.AbstractEventLoop
    event: asyncio.Event
    value: Any = _MISSING
    error: BaseException | None = None


class VisionFlights:
    """同 key 的并发视觉请求只跑一次（asyncio 级 singleflight，fail-open）。

    :class:`VisionCache` 只挡得住**先后**的重复投递；同一时刻的重复投递会各发一次
    HTTP（VERIFY-B P2-3：4 并发 = 4 次 POST）。契约：leader 成功 → 等待者共享同一
    结果；leader 抛异常 → 同一个异常向等待者传播（调用方据此不写缓存）；leader 被
    取消 → 等待者自己接管；空 key → 直接跑。与 ``core.limits.SingleFlight`` 的区别：
    那个是线程版（``threading.Event``，给 ffmpeg 用），这里必须用 ``asyncio.Event``。
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._flights: dict[str, _Flight] = {}

    @property
    def pending(self) -> int:
        """在飞的 flight 数（结束后必须回到 0）。"""
        with self._lock:
            return len(self._flights)

    async def run(self, key: str, work: Callable[[], Awaitable[Any]]) -> Any:
        """执行 ``work``（同 key 只一个真跑）；返回 leader 或 leader 的结果。"""
        if not key:  # fail-open：算不出可比标识就不去重
            return await work()
        loop = asyncio.get_running_loop()
        with self._lock:
            flight = self._flights.get(key)
            if flight is not None and flight.loop is not loop:
                flight = None  # 换 loop：旧条目作废（它的等待者也在旧 loop 上）
            leader = flight is None
            if leader:
                flight = _Flight(loop, asyncio.Event())
                self._flights[key] = flight
        if not leader:
            await flight.event.wait()
            if flight.error is not None:
                raise flight.error  # 失败向等待者传播（调用方据此不写缓存）
            if flight.value is not _MISSING:
                return flight.value
            return await self.run(key, work)  # leader 被取消：自己接管
        try:
            value = await work()
        except asyncio.CancelledError:
            self._settle(key, flight)  # 无结果：唤醒等待者让它自己接管
            raise
        except BaseException as exc:  # noqa: BLE001 - 失败必须原样传给等待者
            self._settle(key, flight, error=exc)
            raise
        self._settle(key, flight, value=value)
        return value

    def _settle(
        self, key: str, flight: _Flight, *, value: Any = _MISSING, error=None
    ) -> None:
        """登记结果并唤醒等待者（只摘自己那条 flight）。"""
        flight.value, flight.error = value, error
        with self._lock:
            if self._flights.get(key) is flight:
                del self._flights[key]
        flight.event.set()


def cache_key(
    *,
    video: str,
    prompt: str,
    proto: str,
    url: str,
    model: str,
    max_tokens: int,
    version: int | None = None,
) -> str:
    """六个维度 + ``KEY_VERSION`` 的 sha256（十六进制）。

    Args:
        video: 模型实际看到的画面的标识（``VideoInjection.content_key``）：
            直通是 ``as-is:<源内容哈希>``，压缩/截断是
            ``<action>:<源内容哈希 + 压缩参数>``（**不是**产物哈希，见模块说明）。
        prompt: ``build_vision_prompt()`` 的**最终输出**（已折叠用户问题与模板）。
        proto: ``"chat"`` 或 ``"responses"``（``_resolve_endpoint`` 的结果）。
        url: 最终请求 URL（换端点回答会变）。
        model: ``openai_model``。
        max_tokens: ``openai_max_tokens``（上限不同，输出可能被截断）。
        version: 仅供测试注入；默认取模块级 :data:`KEY_VERSION`。

    Returns:
        64 位小写十六进制摘要。
    """
    payload = {
        "v": KEY_VERSION if version is None else int(version),
        "video": str(video),
        "prompt": str(prompt),
        "proto": str(proto),
        "url": str(url),
        "model": str(model),
        "max_tokens": int(max_tokens),
    }
    blob = json.dumps(payload, sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


class VisionCache:
    """``OrderedDict`` 实现的 TTL + LRU 缓存（纯内存、不落盘）。

    * **命中不续期**：否则热门条目永不过期，内存里会沉淀一份"永远正确"的旧答案；
    * **只缓存非空文本**：空回答进缓存等于把一次空响应固定成 TTL 内的持续空结果；
    * **一把 ``threading.Lock``**：钩子在事件循环里单线程调用，但测试会并发，
      而且锁的代价远低于一次视觉请求。

    ``flights`` 是**并发**去重（P2-3）：缓存挡不住"同时发生"的重复投递，
    :class:`VisionFlights` 让同 key 的后来者等 leader 的结果。它随缓存对象一起
    创建 —— ``vision_cache_enabled=false`` 时连对象都没有，行为回到 0.2.1。
    """

    def __init__(
        self,
        *,
        ttl_sec: float = 600.0,
        max_entries: int = 64,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self.ttl_sec = max(0.0, float(ttl_sec))
        self.max_entries = max(1, int(max_entries))
        self._clock = clock
        self._lock = threading.Lock()
        self._entries: OrderedDict[str, tuple[str, float]] = OrderedDict()
        self._stats = VisionCacheStats()
        self.flights = VisionFlights()

    # ------------------------------------------------------------ 只读状态

    @property
    def stats(self) -> VisionCacheStats:
        """累计计数（进程内，永不清零；``clear()`` 只清条目）。"""
        return self._stats

    def __len__(self) -> int:
        with self._lock:
            return len(self._entries)

    # ---------------------------------------------------------------- 读写

    def lookup(self, key: str) -> CacheHit | None:
        """命中返回 :class:`CacheHit`（含剩余 TTL）；未命中/过期返回 ``None``。"""
        if not key:
            with self._lock:
                self._stats.misses += 1
            return None
        now = self._clock()
        with self._lock:
            entry = self._entries.get(key)
            if entry is None:
                self._stats.misses += 1
                return None
            analysis, stored_at = entry
            age = now - stored_at
            if age >= self.ttl_sec:
                del self._entries[key]
                self._stats.expirations += 1
                self._stats.misses += 1
                return None
            self._entries.move_to_end(key)
            self._stats.hits += 1
            return CacheHit(analysis=analysis, ttl_left=self.ttl_sec - age)

    def get(self, key: str) -> str | None:
        """只要分析原文时的便捷入口。"""
        hit = self.lookup(key)
        return None if hit is None else hit.analysis

    def put(self, key: str, analysis: str) -> None:
        """写入一条；空文本/空 key 直接忽略（调用方不必自己判断）。"""
        text = analysis if isinstance(analysis, str) else ""
        if not key or not text.strip():
            return
        now = self._clock()
        with self._lock:
            self._entries[key] = (text, now)
            self._entries.move_to_end(key)
            self._stats.stores += 1
            while len(self._entries) > self.max_entries:
                self._entries.popitem(last=False)
                self._stats.evictions += 1

    def clear(self) -> None:
        """清空条目（计数保留，便于观测"清过几次之后仍然命中的是哪些"）。"""
        with self._lock:
            self._entries.clear()


__all__ = [
    "KEY_VERSION",
    "CacheHit",
    "VisionCache",
    "VisionCacheStats",
    "VisionFlights",
    "cache_key",
]
