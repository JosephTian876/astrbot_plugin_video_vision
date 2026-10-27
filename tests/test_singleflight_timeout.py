"""SingleFlight：follower 的等待**超时**不许让同一个 key 跑第二份（P2-1）。

契约（``core/limits.SingleFlight`` 类文档 / ``README`` 的"同文件并发只压一次"）：

* 同一 key 的并发调用**只让一个真跑**；
* 只有 **leader 自己**能在完成时摘掉登记；
* follower 的 ``wait_timeout`` 只是"我等了多久"，不是"我接管了"。

AUDIT-VV-R9 §3.3 的实测：``wait_timeout=0.2``、leader 跑 0.8s 时
``core/limits.py`` 的超时路径 ``_discard(key, flight)`` 删掉的正是 **leader 还在
跑的那条登记**，于是 follower 重入 ``_enter`` 时 leader 已不存在 —— 它自己成了
第二个 leader：``concurrent_peak=2, work_starts=2``。

本文件把这四条钉死：
1. follower 超时后同 key 的**真实并发峰值仍是 1**（``work_starts == 1``）；
2. 超时后的 follower **没有**删掉 leader 的登记（后来的调用仍加入原 leader）；
3. leader 的确定性失败在 follower 超时之后仍然原样传播（同一个异常对象）；
4. 任何分支结束后 ``pending == 0``（登记表不泄漏）。

等待全部用 ``threading.Event`` 显式同步（包括"follower 的等待预算真的到期了"
这件事 —— 靠 ``on_overdue`` 回调，不靠 ``sleep`` 猜时间）。
"""

from __future__ import annotations

import threading

from vv_plugin.core.limits import SingleFlight


class _MediaToolError(RuntimeError):
    """带机器可读分类的失败（与 ``core/media.MediaToolError`` 同形）。"""

    def __init__(self, message: str, kind: str) -> None:
        super().__init__(message)
        self.kind = kind


class _Meter:
    """记录"真正在跑的 work 份数"（并发峰值与启动次数）。"""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self.concurrent = 0
        self.peak = 0
        self.starts: list[str] = []

    def enter(self) -> None:
        with self._lock:
            self.concurrent += 1
            self.peak = max(self.peak, self.concurrent)
            self.starts.append(threading.current_thread().name)

    def leave(self) -> None:
        with self._lock:
            self.concurrent -= 1

    def snapshot(self) -> tuple[list[str], int]:
        with self._lock:
            return list(self.starts), self.peak


#: follower 的等待上限：比 leader 的存活时间短一个数量级（必超时）
FOLLOWER_TIMEOUT = 0.1


def test_a_timed_out_follower_never_starts_a_second_work():
    """leader 卡着不返回时，follower 超时后**不许**自己另起一份（并发仍为 1）。"""
    flight = SingleFlight()
    meter = _Meter()
    store: dict[str, str] = {}
    leader_started = threading.Event()
    follower_joined = threading.Event()
    follower_overdue = threading.Event()
    release = threading.Event()
    results: list[object] = []

    def work() -> str:
        meter.enter()
        try:
            leader_started.set()
            assert release.wait(10), "用例没有放行 leader"
            store["k"] = "artifact"
            return "artifact"
        finally:
            meter.leave()

    def run(on_join=None, on_overdue=None) -> None:
        try:
            results.append(
                flight.run(
                    "k",
                    work,
                    probe=lambda: store.get("k"),
                    wait_timeout=FOLLOWER_TIMEOUT,
                    on_join=on_join,
                    on_overdue=on_overdue,
                )
            )
        except BaseException as exc:  # pragma: no cover - 失败时把原因带出来
            results.append(exc)

    leader = threading.Thread(target=run, name="leader")
    leader.start()
    assert leader_started.wait(5), "leader 没有开始"
    follower = threading.Thread(
        target=lambda: run(follower_joined.set, follower_overdue.set),
        name="follower",
    )
    follower.start()
    assert follower_joined.wait(5), "follower 没有加入"
    assert follower_overdue.wait(5), "follower 的等待预算没有到期"
    starts_during, peak_during = meter.snapshot()
    pending_during = flight.pending
    release.set()
    leader.join(10)
    follower.join(10)

    assert starts_during == ["leader"], (
        f"follower 超时后又起了一份 work: {starts_during}"
        "（P2-1：它把 leader 的登记删了）"
    )
    assert peak_during == 1, f"同 key 并发峰值 {peak_during} > 1"
    assert pending_during == 1, f"登记表里应当只有 leader 那条，实际 {pending_during}"
    assert meter.starts == ["leader"], meter.starts
    assert meter.peak == 1, f"全程并发峰值 {meter.peak} > 1"
    assert results == ["artifact", "artifact"], results
    assert flight.pending == 0


def test_a_follower_timeout_keeps_the_leader_registered_for_latecomers():
    """follower 超时**只放弃等待**，不许把 leader 的登记删掉（后来者仍加入 leader）。"""
    flight = SingleFlight()
    meter = _Meter()
    store: dict[str, str] = {}
    leader_started = threading.Event()
    follower_overdue = threading.Event()
    late_joined = threading.Event()
    release = threading.Event()
    results: list[object] = []

    def work() -> str:
        meter.enter()
        try:
            leader_started.set()
            assert release.wait(10), "用例没有放行 leader"
            store["k"] = "artifact"
            return "artifact"
        finally:
            meter.leave()

    def run(timeout: float, on_join=None, on_overdue=None) -> None:
        try:
            results.append(
                flight.run(
                    "k",
                    work,
                    probe=lambda: store.get("k"),
                    wait_timeout=timeout,
                    on_join=on_join,
                    on_overdue=on_overdue,
                )
            )
        except BaseException as exc:  # pragma: no cover
            results.append(exc)

    leader = threading.Thread(target=run, args=(10.0,), name="leader")
    leader.start()
    assert leader_started.wait(5), "leader 没有开始"
    follower = threading.Thread(
        target=lambda: run(FOLLOWER_TIMEOUT, None, follower_overdue.set),
        name="follower",
    )
    follower.start()
    # 先确认 follower 已经超时过（它已经把"自己的等待"放弃了一轮）
    assert follower_overdue.wait(5), "follower 的等待预算没有到期"
    # 迟到者：如果 leader 的登记还在，它必须加入 leader（并等 leader 的产物）
    late = threading.Thread(
        target=lambda: run(10.0, late_joined.set), name="late"
    )
    late.start()
    assert late_joined.wait(5), "迟到者没有加入任何 flight"
    starts_before_release, peak_before_release = meter.snapshot()
    release.set()
    for thread in (leader, follower, late):
        thread.join(10)

    assert starts_before_release == ["leader"], (
        f"迟到者另起了一份 work: {starts_before_release}"
        "（leader 的登记被 follower 超时删掉了）"
    )
    assert peak_before_release == 1
    assert meter.starts == ["leader"], meter.starts
    assert meter.peak == 1
    assert sorted(results) == ["artifact"] * 3, results
    assert flight.pending == 0


def test_a_timed_out_follower_still_receives_the_leader_deterministic_failure():
    """leader 的确定性失败在 follower 超时之后仍然原样传播（同一个异常对象）。"""
    flight = SingleFlight()
    meter = _Meter()
    leader_started = threading.Event()
    follower_overdue = threading.Event()
    release = threading.Event()
    errors: list[BaseException] = []

    def work() -> str:
        meter.enter()
        try:
            leader_started.set()
            assert release.wait(10), "用例没有放行 leader"
            raise _MediaToolError("视频已损坏", kind="corrupt")
        finally:
            meter.leave()

    def run(on_overdue=None) -> None:
        try:
            flight.run(
                "k",
                work,
                probe=lambda: None,
                wait_timeout=FOLLOWER_TIMEOUT,
                on_overdue=on_overdue,
            )
        except BaseException as exc:
            errors.append(exc)

    leader = threading.Thread(target=run, name="leader")
    leader.start()
    assert leader_started.wait(5)
    follower = threading.Thread(
        target=lambda: run(follower_overdue.set), name="follower"
    )
    follower.start()
    assert follower_overdue.wait(5), "follower 的等待预算没有到期"
    release.set()
    leader.join(10)
    follower.join(10)

    assert meter.starts == ["leader"], meter.starts
    assert meter.peak == 1
    assert len(errors) == 2, errors
    assert all(getattr(exc, "kind", "") == "corrupt" for exc in errors), errors
    assert errors[0] is errors[1], "两个调用方必须拿到**同一个**异常对象"
    assert flight.pending == 0
