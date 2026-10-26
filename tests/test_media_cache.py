"""压缩缓存：原子发布、绝不返回幽灵路径、大产物不缓存、并发可用。

对应修复项：A1（P1 缓存淘汰吃掉刚写入的产物 + 锁 + 原子发布）。
"""

from __future__ import annotations

import os
import threading
from pathlib import Path

import pytest

from vv_plugin.core.media import CompressCache


def sparse(path: Path, size_bytes: int) -> Path:
    """造一个指定"逻辑体积"的稀疏文件（秒建，不占真实磁盘）。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as handle:
        handle.truncate(size_bytes)
    return path


def test_put_returns_existing_path(tmp_path):
    cache = CompressCache(tmp_path / "cache", max_mb=16)
    produced = sparse(tmp_path / "produced.mp4", 1024 * 1024)

    result = cache.put("key1", produced)

    assert Path(result).is_file()
    assert Path(result) == cache._path_for("key1")
    assert cache.get("key1") == result


def test_get_miss_returns_none(tmp_path):
    cache = CompressCache(tmp_path / "cache", max_mb=16)
    assert cache.get("nope") is None


def test_get_ignores_zero_byte_entry(tmp_path):
    cache = CompressCache(tmp_path / "cache", max_mb=16)
    target = cache._path_for("empty")
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(b"")
    assert cache.get("empty") is None


def test_artifact_larger_than_budget_is_not_cached(tmp_path):
    """回归（P1）：0.2.0 会把刚写进去的大产物删掉，然后返回那个路径。"""
    cache = CompressCache(tmp_path / "cache", max_mb=16)
    produced = sparse(tmp_path / "big.mp4", 20 * 1024 * 1024)

    result = cache.put("big", produced)

    assert Path(result) == produced, "超过缓存预算的产物应原样返回"
    assert Path(result).is_file()
    assert list((tmp_path / "cache").glob("*.mp4")) == []


def test_put_below_budget_is_cached(tmp_path):
    """边界：小于预算的产物必须正常进缓存（确认不是"一律不缓存"）。"""
    cache = CompressCache(tmp_path / "cache", max_mb=16)
    produced = sparse(tmp_path / "ok.mp4", 15 * 1024 * 1024)
    result = cache.put("ok", produced)
    assert Path(result) == cache._path_for("ok")
    assert Path(result).is_file()


def test_put_protects_its_own_entry_during_eviction(tmp_path):
    """新写入的条目在淘汰中必须存活，旧条目被清掉。"""
    cache = CompressCache(tmp_path / "cache", max_mb=16)
    first = cache.put("old", sparse(tmp_path / "old.mp4", 15 * 1024 * 1024))
    assert Path(first).is_file()

    second = cache.put("new", sparse(tmp_path / "new.mp4", 15 * 1024 * 1024))

    assert Path(second).is_file(), "刚写入的产物被淘汰吃掉了"
    assert Path(second) == cache._path_for("new")
    total = sum(p.stat().st_size for p in (tmp_path / "cache").glob("*.mp4"))
    assert total <= cache.max_bytes


def test_zero_byte_artifact_is_returned_as_is(tmp_path):
    cache = CompressCache(tmp_path / "cache", max_mb=16)
    produced = tmp_path / "zero.mp4"
    produced.parent.mkdir(parents=True, exist_ok=True)
    produced.write_bytes(b"")
    assert cache.put("zero", produced) == str(produced)


def test_missing_artifact_path_is_returned_without_raising(tmp_path):
    cache = CompressCache(tmp_path / "cache", max_mb=16)
    ghost = tmp_path / "ghost.mp4"
    assert cache.put("ghost", ghost) == str(ghost)


def test_concurrent_put_keeps_cache_usable(tmp_path):
    """4 线程并发 put：全部返回存在的路径，缓存仍可命中。"""
    cache = CompressCache(tmp_path / "cache", max_mb=64)
    results: dict[int, str] = {}
    errors: list[BaseException] = []
    barrier = threading.Barrier(4)

    def worker(index: int) -> None:
        try:
            produced = sparse(tmp_path / f"p{index}.mp4", 2 * 1024 * 1024)
            barrier.wait(timeout=10)
            results[index] = cache.put(f"k{index}", produced)
        except BaseException as exc:  # noqa: BLE001 - 测试要看到所有异常
            errors.append(exc)

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(4)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=30)

    assert errors == []
    assert len(results) == 4
    assert all(Path(item).is_file() for item in results.values())
    assert len(list((tmp_path / "cache").glob("*.mp4"))) == 4
    assert cache.get("k0") is not None


def test_atomic_publish_never_exposes_a_partial_file(tmp_path):
    """读线程只能看到"完整的旧文件"或"完整的新文件"。"""
    cache = CompressCache(tmp_path / "cache", max_mb=64)
    stop = threading.Event()
    torn: list[tuple[int, int]] = []

    def reader() -> None:
        while not stop.is_set():
            hit = cache.get("same")
            if not hit:
                continue
            data = Path(hit).read_bytes()
            if len(set(data)) != 1:  # 内容应是同一种填充字节
                torn.append((len(data), len(set(data))))

    thread = threading.Thread(target=reader, daemon=True)
    thread.start()
    try:
        for index in range(20):
            fill = b"A" if index % 2 == 0 else b"B"
            source = tmp_path / f"src{index}.mp4"
            source.write_bytes(fill * (512 * 1024))
            cache.put("same", source)
    finally:
        stop.set()
        thread.join(timeout=10)

    assert torn == []


def test_publish_failure_returns_a_live_path(tmp_path, monkeypatch):
    """发布阶段抛错时，不能返回不存在的路径（产物要还原）。"""
    cache = CompressCache(tmp_path / "cache", max_mb=16)
    produced = sparse(tmp_path / "keep.mp4", 1024 * 1024)

    def boom(*args, **kwargs):
        raise OSError(28, "No space left on device")

    monkeypatch.setattr(os, "replace", boom)
    result = cache.put("key", produced)

    assert Path(result).is_file()
    assert Path(result) == produced


def test_eviction_stat_failure_does_not_delete_anything(tmp_path, monkeypatch):
    cache = CompressCache(tmp_path / "cache", max_mb=16)
    cache.put("a", sparse(tmp_path / "a.mp4", 4 * 1024 * 1024))

    class _BrokenPath(type(Path())):
        def stat(self, *args, **kwargs):
            raise OSError("boom")

    # 让 glob 返回的条目 stat 失败：代码应"宁可不淘汰"
    monkeypatch.setattr(
        type(cache.root), "glob", lambda self, pattern: [_BrokenPath(cache.root, "x")]
    )
    cache._evict()  # 不应抛异常
    assert cache.get("a") is not None


@pytest.mark.parametrize("size_mb", [16.0, 16.001])
def test_budget_boundary(tmp_path, size_mb):
    """恰好等于上限时保留，多 1 字节就不进缓存。"""
    cache = CompressCache(tmp_path / "cache", max_mb=16)
    size = int(size_mb * 1024 * 1024)
    produced = sparse(tmp_path / f"s{size}.mp4", size)
    result = cache.put(f"k{size}", produced)
    if size <= cache.max_bytes:
        assert Path(result) == cache._path_for(f"k{size}")
    else:
        assert Path(result) == produced
    assert Path(result).is_file()
