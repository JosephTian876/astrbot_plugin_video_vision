"""视频文件的探测、编码与压缩工具。

**不依赖 AstrBot**，便于日后上移到核心。

所有外部命令（ffmpeg / ffprobe）都带超时与容错，任何失败都返回 None 或抛
``MediaToolError``，由上层决定降级策略。
"""

from __future__ import annotations

import base64
import hashlib
import os
import shutil
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path

#: 常见视频容器 -> MIME
VIDEO_MIME_BY_SUFFIX: dict[str, str] = {
    ".mp4": "video/mp4",
    ".m4v": "video/mp4",
    ".mov": "video/quicktime",
    ".webm": "video/webm",
    ".mkv": "video/x-matroska",
    ".avi": "video/x-msvideo",
    ".flv": "video/x-flv",
    ".wmv": "video/x-ms-wmv",
    ".mpeg": "video/mpeg",
    ".mpg": "video/mpeg",
    ".3gp": "video/3gpp",
    ".ts": "video/mp2t",
    ".ogv": "video/ogg",
}

#: 压缩输出的统一容器
OUTPUT_SUFFIX = ".mp4"
OUTPUT_MIME = "video/mp4"


class MediaToolError(RuntimeError):
    """ffmpeg / ffprobe 调用失败。"""


@dataclass(frozen=True)
class VideoInfo:
    """一次探测的结果。"""

    path: str
    size_mb: float
    duration_sec: float
    mime: str


def guess_video_mime(path: str | os.PathLike[str]) -> str:
    """按扩展名猜 MIME；未知则回落到 video/mp4。"""
    suffix = Path(path).suffix.lower()
    return VIDEO_MIME_BY_SUFFIX.get(suffix, OUTPUT_MIME)


def file_size_mb(path: str | os.PathLike[str]) -> float:
    try:
        return os.path.getsize(path) / 1024 / 1024
    except OSError:
        return 0.0


def resolve_ffmpeg_tools(configured: str = "") -> tuple[str | None, str | None]:
    """定位 ffmpeg / ffprobe。

    ``configured`` 可以是 ffmpeg 可执行文件路径，也可以是所在目录。
    留空则从 PATH 查找。
    """
    configured = (configured or "").strip()
    if configured:
        candidate = Path(configured)
        if candidate.is_dir():
            ffmpeg = candidate / ("ffmpeg.exe" if os.name == "nt" else "ffmpeg")
            ffprobe = candidate / ("ffprobe.exe" if os.name == "nt" else "ffprobe")
            return (
                str(ffmpeg) if ffmpeg.exists() else None,
                str(ffprobe) if ffprobe.exists() else None,
            )
        if candidate.is_file():
            sibling = candidate.with_name(
                "ffprobe.exe" if os.name == "nt" else "ffprobe"
            )
            return (
                str(candidate),
                str(sibling) if sibling.exists() else shutil.which("ffprobe"),
            )
        return None, None
    return shutil.which("ffmpeg"), shutil.which("ffprobe")


def probe_duration(path: str | os.PathLike[str], ffprobe: str | None) -> float:
    """读取时长（秒）。失败返回 0.0（视为未知，不阻断流程）。"""
    if not ffprobe:
        return 0.0
    try:
        completed = subprocess.run(
            [
                ffprobe,
                "-v", "error",
                "-show_entries", "format=duration",
                "-of", "default=nw=1:nk=1",
                str(path),
            ],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=30,
            check=True,
        )
        return float((completed.stdout or "").strip())
    except Exception:
        return 0.0


def probe_video(path: str | os.PathLike[str], ffprobe: str | None) -> VideoInfo:
    """探测体积 / 时长 / MIME。"""
    return VideoInfo(
        path=str(path),
        size_mb=file_size_mb(path),
        duration_sec=probe_duration(path, ffprobe),
        mime=guess_video_mime(path),
    )


def content_hash(path: str | os.PathLike[str], *, chunk: int = 1 << 20) -> str:
    """按内容哈希（用于压缩缓存 key）。"""
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        while True:
            block = handle.read(chunk)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()[:32]


def read_as_data_uri(
    path: str | os.PathLike[str], mime: str | None = None
) -> str:
    """读取文件并编码为 data URI。"""
    resolved_mime = mime or guess_video_mime(path)
    with open(path, "rb") as handle:
        payload = handle.read()
    encoded = base64.b64encode(payload).decode("ascii")
    return f"data:{resolved_mime};base64,{encoded}"


def _run_ffmpeg(args: list[str], timeout: float) -> None:
    try:
        completed = subprocess.run(
            args,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=max(1.0, timeout),
        )
    except subprocess.TimeoutExpired as exc:
        raise MediaToolError(f"ffmpeg 超时（>{timeout:.0f}s）") from exc
    except Exception as exc:  # noqa: BLE001
        raise MediaToolError(f"ffmpeg 调用失败: {exc}") from exc
    if completed.returncode != 0:
        tail = (completed.stderr or "").strip().splitlines()[-4:]
        raise MediaToolError(
            f"ffmpeg 退出码 {completed.returncode}: {' | '.join(tail)[:300]}"
        )


def compress_video(
    source: str | os.PathLike[str],
    output: str | os.PathLike[str],
    *,
    max_mb: float,
    crf: int = 30,
    preset: str = "veryfast",
    max_duration_sec: float = 0.0,
    ffmpeg: str | None,
    ffprobe: str | None,
    timeout: float = 45.0,
) -> str:
    """压缩（可选截断）视频，体积受 ``max_mb`` 约束。

    采用 **constrained CRF**：``-crf`` 决定画质与体积（随内容复杂度自适应，
    实测同参数下静态画面压得极小、高熵画面压得较大），``-maxrate``/``-bufsize``
    按 ``max_mb`` 反推出码率**上限**作为保险丝，保证不会压出超预算的文件。

    之所以不用 ABR（``-b:v``）精确瞄准体积：体积随内容自适应更符合"送进模型看懂"
    这个目标 —— 简单画面压得更小、上传更快，复杂画面自动保留更多细节。

    Returns:
        输出文件路径。
    """
    if not ffmpeg:
        raise MediaToolError("未找到 ffmpeg，无法压缩")

    duration = probe_duration(source, ffprobe)
    if duration <= 0:
        raise MediaToolError("无法读取视频时长，拒绝盲压缩")

    effective_duration = duration
    if max_duration_sec and max_duration_sec > 0:
        effective_duration = min(duration, max_duration_sec)

    # 体积上限 -> 码率上限（kbps）。留 8% 余量给容器与音频开销。
    budget_kbits = max(1.0, max_mb) * 8192 * 0.92
    total_kbps = budget_kbits / max(0.5, effective_duration)
    # 音频固定 96k，其余给视频，且不低于 120k 以免画质崩坏
    video_kbps = max(120.0, total_kbps - 96.0)

    args = [ffmpeg, "-y", "-hide_banner", "-loglevel", "error", "-i", str(source)]
    if max_duration_sec and max_duration_sec > 0 and duration > max_duration_sec:
        args += ["-t", f"{max_duration_sec:.3f}"]
    args += [
        "-c:v", "libx264",
        "-preset", preset,
        "-crf", str(int(crf)),
        "-maxrate", f"{int(video_kbps)}k",
        "-bufsize", f"{int(video_kbps * 2)}k",
        "-pix_fmt", "yuv420p",
        "-c:a", "aac",
        "-b:a", "96k",
        "-movflags", "+faststart",
        str(output),
    ]
    _run_ffmpeg(args, timeout)

    if not os.path.exists(output) or os.path.getsize(output) == 0:
        raise MediaToolError("ffmpeg 未产出有效文件")
    return str(output)


class CompressCache:
    """按源文件内容哈希缓存压缩结果，避免重试时重复压缩。"""

    def __init__(self, root: str | os.PathLike[str], max_mb: int = 512) -> None:
        self.root = Path(root)
        self.max_bytes = max(16, int(max_mb)) * 1024 * 1024

    def _path_for(self, key: str) -> Path:
        return self.root / f"{key}{OUTPUT_SUFFIX}"

    def get(self, key: str) -> str | None:
        candidate = self._path_for(key)
        if candidate.exists() and candidate.stat().st_size > 0:
            try:
                os.utime(candidate, None)  # 刷新 LRU 时间戳
            except OSError:
                pass
            return str(candidate)
        return None

    def put(self, key: str, produced: str | os.PathLike[str]) -> str:
        """把压缩产物移入缓存并返回缓存路径。失败则返回原路径。"""
        try:
            self.root.mkdir(parents=True, exist_ok=True)
            target = self._path_for(key)
            if Path(produced) != target:
                shutil.move(str(produced), str(target))
            self._evict()
            return str(target)
        except Exception:
            return str(produced)

    def _evict(self) -> None:
        try:
            entries = sorted(
                (p for p in self.root.glob(f"*{OUTPUT_SUFFIX}") if p.is_file()),
                key=lambda p: p.stat().st_mtime,
            )
        except OSError:
            return
        total = sum(p.stat().st_size for p in entries)
        index = 0
        while total > self.max_bytes and index < len(entries):
            victim = entries[index]
            index += 1
            try:
                size = victim.stat().st_size
                victim.unlink()
                total -= size
            except OSError:
                continue

    def stamp(self) -> str:
        return time.strftime("%Y%m%d%H%M%S")
