"""把视频占位符转换成 Gemini provider 能识别的视频内容块。

这是插件的核心决策层：**不依赖 AstrBot**。

处理流程::

    路径存在性 -> 探测体积/时长 -> 是否需要压缩或截断
        |                                  |
        |  不需要                          |  需要
        v                                  v
      直接编码                       oversize_action 分支
                                       compress -> ffmpeg(带缓存) -> 编码
                                       truncate -> ffmpeg 截断 -> 编码
                                       reject   -> 放弃并给出原因

任何一步失败都返回 ``ok=False`` 的结果，**绝不抛异常打断对话**。
"""

from __future__ import annotations

import logging
import os
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .config import VideoConfig
from .media import (
    OUTPUT_SUFFIX,
    CompressCache,
    MediaToolError,
    compress_video,
    content_hash,
    file_size_mb,
    probe_duration,
    read_as_data_uri,
    resolve_ffmpeg_tools,
)
from .placeholder import VideoPlaceholder

_log = logging.getLogger(__name__)


@dataclass(frozen=True)
class VideoInjection:
    """一个准备好注入的内容块及其元信息。"""

    block: dict[str, Any]
    source_path: str
    used_path: str
    size_mb: float
    duration_sec: float
    action: str  # as-is | compressed | truncated | rejected
    note: str = ""

    @property
    def data_uri(self) -> str:
        """供上层构造媒体 Part 的 data URI。"""
        return self.block["audio_url"]["url"]


@dataclass(frozen=True)
class PrepareOutcome:
    """准备结果。``ok=False`` 时 ``reason`` 说明原因（用于日志与 /videotest）。"""

    ok: bool
    injection: VideoInjection | None = None
    reason: str = ""


def _make_block(used_path: str) -> dict[str, Any]:
    """构造内容块。

    用 ``audio_url`` 承载视频是刻意为之：Gemini provider 的 ``process_audio_url``
    从 data URI 里取 MIME 再交给 ``Part.from_bytes``，因此 ``data:video/*`` 会被
    正确构建成**视频 Part**。详见 hooks/llm_request.py 的说明。

    MIME 必须**跟随实际文件**，不能硬编码：直通时是源格式（webm/mov/mkv…），
    压缩后才是 ffmpeg 产出的 mp4。谎报 MIME 在 Gemini 上碰巧能跑（它会嗅探真实
    容器），但换一个信任声明 MIME 的端点就会失败。
    """
    data_uri = read_as_data_uri(used_path)
    return {"type": "audio_url", "audio_url": {"url": data_uri}}


def prepare_video(
    placeholder: VideoPlaceholder,
    *,
    config: VideoConfig,
    cache_root: str | os.PathLike[str] | None = None,
    logger: Any | None = None,
    debug: bool = False,
) -> PrepareOutcome:
    """按配置准备一个视频，返回可注入的内容块或失败原因。

    ``debug=True`` 时以 INFO 级别输出逐步细节（不受 AstrBot 全局日志级别影响），
    对应插件配置里的 ``debug_log``。
    """
    log = logger or _log

    def dbg(message: str, *args: Any) -> None:
        if debug:
            log.info("[VideoVision][debug] " + message, *args)

    source = placeholder.path

    if not source or not os.path.exists(source):
        return PrepareOutcome(False, reason=f"文件不存在: {source}")
    if not os.path.isfile(source):
        return PrepareOutcome(False, reason=f"不是普通文件: {source}")

    ffmpeg, ffprobe = resolve_ffmpeg_tools(config.ffmpeg_path)
    size_mb = file_size_mb(source)
    duration = probe_duration(source, ffprobe)
    dbg(
        "探测 %s | 体积=%.2fMB 时长=%.2fs | ffmpeg=%s ffprobe=%s",
        placeholder.name,
        size_mb,
        duration,
        ffmpeg or "缺失",
        ffprobe or "缺失",
    )

    oversize = size_mb > config.max_size_mb
    overlong = (
        config.max_duration_sec > 0
        and duration > 0
        and duration > config.max_duration_sec
    )
    dbg(
        "判定 超体积=%s 超时长=%s | 闸门=%.1fMB 时长上限=%.1fs 策略=%s",
        oversize,
        overlong,
        config.max_size_mb,
        config.max_duration_sec,
        config.oversize_action,
    )

    # ---- 无需处理，直接送 ----
    if not oversize and not overlong:
        dbg("无需处理，直接编码源文件")
        try:
            block = _make_block(source)
        except Exception as exc:  # noqa: BLE001
            return PrepareOutcome(False, reason=f"编码失败: {exc}")
        dbg("编码完成 | data URI %.2fMB", len(block["audio_url"]["url"]) / 1024 / 1024)
        return PrepareOutcome(
            True,
            VideoInjection(
                block=block,
                source_path=source,
                used_path=source,
                size_mb=size_mb,
                duration_sec=duration,
                action="as-is",
            ),
        )

    why = []
    if oversize:
        why.append(f"体积 {size_mb:.1f}MB > 闸门 {config.max_size_mb:.1f}MB")
    if overlong:
        why.append(f"时长 {duration:.1f}s > 上限 {config.max_duration_sec:.1f}s")
    trigger = "；".join(why)

    # ---- 拒绝 ----
    if config.oversize_action == "reject":
        return PrepareOutcome(False, reason=f"按配置拒绝（{trigger}）")

    # ---- 需要 ffmpeg ----
    if not ffmpeg:
        hint = (
            f"配置的 ffmpeg_path 无效: {config.ffmpeg_path}"
            if config.ffmpeg_path
            else "请安装 ffmpeg 或在插件配置里指定 ffmpeg_path"
        )
        return PrepareOutcome(
            False, reason=f"需要压缩但未找到 ffmpeg（{trigger}）。{hint}"
        )
    if duration <= 0:
        return PrepareOutcome(
            False, reason=f"无法读取时长，拒绝盲压缩（{trigger}）"
        )

    truncate_to = config.max_duration_sec if overlong else 0.0
    action = "truncated" if (overlong and not oversize) else "compressed"

    # ---- 缓存查找 ----
    cache: CompressCache | None = None
    cache_key = ""
    if config.cache_enabled and cache_root:
        try:
            signature = (
                f"{content_hash(source)}|{config.compress_max_mb}"
                f"|{config.compress_crf}|{config.compress_preset}|{truncate_to}"
            )
            cache_key = content_hash_from_text(signature)
            cache = CompressCache(cache_root, config.cache_max_mb)
            hit = cache.get(cache_key)
            if hit:
                cached_size = file_size_mb(hit)
                log.debug("[VideoVision] 压缩缓存命中: %s", cache_key)
                dbg("缓存命中 %s | %.2fMB", cache_key, cached_size)
                try:
                    block = _make_block(hit)
                except Exception as exc:  # noqa: BLE001
                    return PrepareOutcome(False, reason=f"缓存文件编码失败: {exc}")
                return PrepareOutcome(
                    True,
                    VideoInjection(
                        block=block,
                        source_path=source,
                        used_path=hit,
                        size_mb=cached_size,
                        duration_sec=min(duration, truncate_to or duration),
                        action=action,
                        note=f"缓存命中；{trigger}",
                    ),
                )
        except Exception as exc:  # noqa: BLE001
            log.debug("[VideoVision] 缓存不可用，继续实时压缩: %s", exc)
            cache = None

    # ---- 实时压缩 ----
    temp_dir = Path(tempfile.mkdtemp(prefix="vv_compress_"))
    produced = temp_dir / f"out{OUTPUT_SUFFIX}"
    dbg(
        "开始压缩 | 目标上限=%.1fMB CRF=%s preset=%s 截断到=%.1fs 超时=%.0fs",
        config.compress_max_mb,
        config.compress_crf,
        config.compress_preset,
        truncate_to,
        config.max_compress_seconds,
    )
    compress_started = time.monotonic()
    try:
        compress_video(
            source,
            produced,
            max_mb=config.compress_max_mb,
            crf=config.compress_crf,
            preset=config.compress_preset,
            max_duration_sec=truncate_to,
            ffmpeg=ffmpeg,
            ffprobe=ffprobe,
            timeout=config.max_compress_seconds,
        )
    except MediaToolError as exc:
        _cleanup(temp_dir)
        return PrepareOutcome(False, reason=f"压缩失败: {exc}")
    except Exception as exc:  # noqa: BLE001
        _cleanup(temp_dir)
        return PrepareOutcome(False, reason=f"压缩异常: {exc}")

    final_path = str(produced)
    if cache is not None and cache_key:
        final_path = cache.put(cache_key, produced)

    new_size = file_size_mb(final_path)
    new_duration = probe_duration(final_path, ffprobe) or (
        min(duration, truncate_to) if truncate_to else duration
    )
    dbg(
        "压缩完成 | 耗时=%.2fs %.2fMB -> %.2fMB | 产物时长=%.2fs",
        time.monotonic() - compress_started,
        size_mb,
        new_size,
        new_duration,
    )

    note = f"{trigger} -> {new_size:.1f}MB"
    if new_size > config.max_size_mb:
        note += "（仍超闸门，已尽力）"
        log.warning(
            "[VideoVision] 压缩后仍超闸门: %.1fMB > %.1fMB",
            new_size,
            config.max_size_mb,
        )

    try:
        block = _make_block(final_path)
    except Exception as exc:  # noqa: BLE001
        return PrepareOutcome(False, reason=f"压缩产物编码失败: {exc}")
    finally:
        # 产物此刻已编码进内存（data URI），临时文件不再需要。
        # 若已移入缓存，temp_dir 已空，rmtree 无副作用；若未启用缓存，
        # 这一步正是防止临时文件泄漏的关键。
        _cleanup(temp_dir)

    return PrepareOutcome(
        True,
        VideoInjection(
            block=block,
            source_path=source,
            used_path=final_path,
            size_mb=new_size,
            duration_sec=new_duration,
            action=action,
            note=note,
        ),
    )


def content_hash_from_text(text: str) -> str:
    """对字符串取短哈希（缓存 key 用）。"""
    import hashlib

    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:32]


def _cleanup(path: str | os.PathLike[str]) -> None:
    import shutil

    try:
        shutil.rmtree(path, ignore_errors=True)
    except Exception:
        pass
