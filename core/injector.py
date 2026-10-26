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

import os
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from astrbot.api import logger as _log

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
    will_truncate,
)
from .placeholder import VIDEO_SUFFIXES, VideoPlaceholder


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
    """准备结果。``ok=False`` 时 ``reason`` 说明原因（用于日志与 /videotest）。

    ``kind`` 是失败分类，取值与 ``MediaToolError.KINDS`` 一致，另有
    ``missing``（路径不存在）/ ``rejected``（配置拒绝）/ ``size`` / ``internal``。
    它存在的意义是让上层不必靠中文字符串猜测失败原因 —— 0.2.0 里
    ``_short_reason`` 就是靠 ``"ffmpeg" in reason`` 判断，于是超时、退出码、
    未找到、产物为空全部被报成「ffmpeg is not available」。
    """

    ok: bool
    injection: VideoInjection | None = None
    reason: str = ""
    kind: str = ""


def _looks_like_video(path: str | os.PathLike[str]) -> bool:
    """扩展名是否属于常见视频容器。"""
    return Path(path).suffix.lower() in VIDEO_SUFFIXES


def _estimate_truncate_seconds(
    duration: float, size_mb: float, max_size_mb: float
) -> float:
    """按体积预算估算「截到多长才不超预算」。

    体积近似随时长线性增长，于是
    ``truncate_to ≈ duration * max_size_mb / size_mb``；再留 10% 余量（线性
    假设对 CRF 编码偏乐观，宁可短一点），且至少 1 秒、不超过源时长。

    Args:
        duration: 源时长（秒）。
        size_mb: 源体积（MB）。
        max_size_mb: 体积闸门（MB）。

    Returns:
        建议的截断时长（秒）；参数不可用时返回 0.0（表示不截断）。
    """
    if duration <= 0 or size_mb <= 0 or max_size_mb <= 0:
        return 0.0
    estimate = duration * (max_size_mb / size_mb) * 0.9
    return max(1.0, min(estimate, duration))


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

    本函数**绝不抛异常**（模块 docstring 的契约）。0.2.0 里 ``tempfile.mkdtemp``
    在 ``try`` 之外，磁盘满时 ``OSError`` 直接穿出 —— 上游 ``hooks._handle`` 的
    兜底把**整条消息的所有视频**都跳过。现在整个实现体都在 ``try`` 内。

    Returns:
        PrepareOutcome。``ok=False`` 时 ``reason`` 可读、``kind`` 可机读。
    """
    log = logger or _log
    try:
        return _prepare_video(
            placeholder,
            config=config,
            cache_root=cache_root,
            log=log,
            debug=debug,
        )
    except Exception as exc:  # noqa: BLE001 - 契约要求：任何失败都只返回结果
        log.warning(
            "[VideoVision] 准备视频时出现未预期异常: %s: %s",
            type(exc).__name__,
            exc,
            exc_info=True,
        )
        return PrepareOutcome(
            False,
            reason=f"未预期异常: {type(exc).__name__}: {exc}",
            kind="internal",
        )


def _prepare_video(
    placeholder: VideoPlaceholder,
    *,
    config: VideoConfig,
    cache_root: str | os.PathLike[str] | None,
    log: Any,
    debug: bool,
) -> PrepareOutcome:
    """``prepare_video`` 的实现体（允许抛异常，由外层兜住）。"""

    def dbg(message: str, *args: Any) -> None:
        if debug:
            log.info("[VideoVision][debug] " + message, *args)

    source = placeholder.path

    if not source or not os.path.exists(source):
        return PrepareOutcome(False, reason=f"文件不存在: {source}", kind="missing")
    if not os.path.isfile(source):
        return PrepareOutcome(
            False, reason=f"不是普通文件: {source}", kind="missing"
        )

    ffmpeg, ffprobe = resolve_ffmpeg_tools(config.ffmpeg_path)
    size_mb = file_size_mb(source)
    # 0 字节 / 读不到体积：0.2.0 会当成"未超限"直接编码注入，等于喂给模型空内容
    if size_mb <= 0:
        return PrepareOutcome(
            False,
            reason=f"文件是 0 字节（或无法读取体积），拒绝发送: {source}",
            kind="empty",
        )
    duration = probe_duration(source, ffprobe)
    # 有时长探测能力却读不出时长、且扩展名是视频/容器 -> 文件损坏或不是媒体。
    # 没有 ffprobe 时无法判断，保持旧行为（照样送，不误伤）。
    if ffprobe and duration <= 0 and _looks_like_video(source):
        return PrepareOutcome(
            False,
            reason=(
                "无法解析视频（时长读取失败，文件可能已损坏或不是有效视频）: "
                f"{source}"
            ),
            kind="corrupt",
        )
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
        except MediaToolError as exc:
            # 例如编码后超过 data URI 硬上限：必须变成可读的 ok=False，
            # 不能让异常冒到钩子层（那样整条消息的视频都不处理）
            return PrepareOutcome(False, reason=f"编码失败: {exc}", kind=exc.kind)
        except Exception as exc:  # noqa: BLE001
            return PrepareOutcome(False, reason=f"编码失败: {exc}", kind="internal")
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
        return PrepareOutcome(
            False, reason=f"按配置拒绝（{trigger}）", kind="rejected"
        )

    # ---- 需要 ffmpeg ----
    if not ffmpeg:
        hint = (
            f"配置的 ffmpeg_path 无效: {config.ffmpeg_path}"
            if config.ffmpeg_path
            else "请安装 ffmpeg 或在插件配置里指定 ffmpeg_path"
        )
        return PrepareOutcome(
            False,
            reason=f"需要压缩但未找到 ffmpeg（{trigger}）。{hint}",
            kind="not_found",
        )
    if duration <= 0:
        return PrepareOutcome(
            False,
            reason=f"无法读取时长，拒绝盲压缩（{trigger}）",
            kind="duration",
        )

    # ---- 截断决策（truncate 现在是**真的**按体积预算截断）----
    # 0.2.0 的 truncate 与 compress 行为完全相同（只在"仅超时长"时多传一个
    # -t），选项形同虚设。这里按体积预算估算一个时长：
    #   truncate_to ≈ duration * max_size_mb / size_mb（留 10% 余量，最小 1s）
    # 用户显式设的 max_duration_sec 仍然优先（取两者更小的）。
    truncate_to = config.max_duration_sec if overlong else 0.0
    if config.oversize_action == "truncate" and oversize:
        estimated = _estimate_truncate_seconds(
            duration, size_mb, config.max_size_mb
        )
        if estimated > 0:
            truncate_to = min(estimated, truncate_to) if truncate_to else estimated
            dbg(
                "truncate 按体积预算估算: %.1fMB -> %.1fs（源 %.1fs / %.1fMB）",
                config.max_size_mb,
                truncate_to,
                duration,
                size_mb,
            )
    # action 只由「是否真的传了 -t」决定，避免报告与行为不一致
    applied_truncate = will_truncate(duration, truncate_to)
    action = "truncated" if applied_truncate else "compressed"

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
                except MediaToolError as exc:
                    return PrepareOutcome(
                        False, reason=f"缓存文件编码失败: {exc}", kind=exc.kind
                    )
                except Exception as exc:  # noqa: BLE001
                    return PrepareOutcome(
                        False, reason=f"缓存文件编码失败: {exc}", kind="internal"
                    )
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
    # 注意：mkdtemp 也可能抛 OSError（磁盘满 / 权限），它必须留在 try 内，
    # 由外层 prepare_video 兜住 —— 0.2.0 把它放在 try 之外，直接穿出契约。
    temp_dir: str | None = None
    try:
        temp_dir = tempfile.mkdtemp(prefix="vv_compress_")
        produced = Path(temp_dir) / f"out{OUTPUT_SUFFIX}"
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
            return PrepareOutcome(
                False, reason=f"压缩失败: {exc}", kind=exc.kind
            )
        except Exception as exc:  # noqa: BLE001
            return PrepareOutcome(
                False, reason=f"压缩异常: {exc}", kind="internal"
            )

        final_path = str(produced)
        if cache is not None and cache_key:
            final_path = cache.put(cache_key, produced)

        new_size = file_size_mb(final_path)
        expected_duration = min(duration, truncate_to) if truncate_to else duration
        new_duration = probe_duration(final_path, ffprobe) or expected_duration
        dbg(
            "压缩完成 | 耗时=%.2fs %.2fMB -> %.2fMB | 产物时长=%.2fs",
            time.monotonic() - compress_started,
            size_mb,
            new_size,
            new_duration,
        )

        note = f"{trigger} -> {new_size:.1f}MB"
        if applied_truncate:
            note += f"（按体积预算截断到 {truncate_to:.1f}s）"
        if expected_duration and new_duration < expected_duration * 0.8:
            # -fs 触顶会让 ffmpeg 提前封口：时长比预期更短，是硬上限的代价
            note += "（受 -fs 体积硬上限约束，产物比预期更短）"
        if new_size > config.max_size_mb:
            note += "（仍超闸门，已尽力）"
            log.warning(
                "[VideoVision] 压缩后仍超闸门: %.1fMB > %.1fMB",
                new_size,
                config.max_size_mb,
            )

        try:
            block = _make_block(final_path)
        except MediaToolError as exc:
            return PrepareOutcome(
                False, reason=f"压缩产物编码失败: {exc}", kind=exc.kind
            )
        except Exception as exc:  # noqa: BLE001
            return PrepareOutcome(
                False, reason=f"压缩产物编码失败: {exc}", kind="internal"
            )

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
    finally:
        # 产物此刻已编码进内存（data URI）或已移入缓存，临时文件不再需要。
        # 放在 finally 里，任何提前 return / 异常路径都不会泄漏临时目录。
        if temp_dir is not None:
            _cleanup(temp_dir)


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
