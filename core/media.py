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
import signal
import subprocess
import threading
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any

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

#: data URI 的硬上限（**编码后**的 base64 字符数）。
#: base64 每 3 字节膨胀到 4 字节，18MB 对应约 13.5MB 原始体积 —— 恰好是默认
#: 体积闸门 13MB 的量级。再大就会让请求体与内存峰值失控（实测 19.53MB 视频
#: 的 data URI 峰值内存 269.6MB = 3.67 倍）。超限直接失败，绝不静默发出去。
MAX_DATA_URI_BYTES = 18 * 1024 * 1024

#: 引用视频（ref 形态）下载的**原始字节**硬上限 = data URI 硬上限换算回原始体积
#: （18MB * 3/4 = 13.5MB，与 ``core/decide.py`` 的 ``URI_LIMIT_RAW_MB`` 同一个
#: 换算）。超过它的视频即使下载成功，也必然在 ``read_as_data_uri`` 的编码上限
#: 处失败；边下边拦，别先写满磁盘再报错。
MAX_REF_DOWNLOAD_BYTES = MAX_DATA_URI_BYTES * 3 // 4


class MediaToolError(RuntimeError):
    """ffmpeg / ffprobe 调用失败。

    ``kind`` 是给上层用的**机器可读分类**，决定 ``hooks.llm_request._short_reason``
    输出哪一句文案（0.2.0 只按 "ffmpeg" 这个词做子串匹配，于是超时、退出码非 0、
    真未找到、产物为空全部被说成「ffmpeg is not available」）。

    取值：``timeout`` / ``exit_code`` / ``not_found`` / ``corrupt`` / ``empty`` /
    ``too_large`` / ``duration`` / ``other``。
    """

    KINDS: frozenset[str] = frozenset(
        {
            "timeout",
            "exit_code",
            "not_found",
            "corrupt",
            "empty",
            "too_large",
            "duration",
            "other",
        }
    )

    def __init__(self, message: str, kind: str = "other") -> None:
        super().__init__(message)
        self.kind = kind if kind in self.KINDS else "other"


def suffix_of(path: str | os.PathLike[str] | None) -> str:
    """小写扩展名（含点）；没有扩展名 / 传空时是空串。

    全仓**唯一**的「取小写后缀」实现。0.2.1 有四份：``main._suffix_of``、
    ``placeholder.VideoPlaceholder.looks_like_video``、``guess_video_mime`` 里的
    ``Path(path).suffix``，以及 ``main`` 找 File 组件时内联的 ``os.path.splitext``
    —— 同一件事四种写法，改一处就会漏（ARCH-REVIEW R5）。

    注意它吃的是**路径**：``suffix_of("mp4")`` 是 ``""``（没有点），
    ``suffix_of(".mp4")`` 也是 ``""``（``splitext`` 把前导点当文件名）。
    要判断「裸扩展名或 MIME 是不是视频容器」，用
    :func:`core.decide._looks_like_video`（它专门处理这两种形态）。

    Args:
        path: 文件路径（``str`` / ``PathLike``）；``None`` 视为空。

    Returns:
        形如 ``.mp4`` 的小写后缀；无后缀时是 ``""``。
    """
    text = "" if path is None else os.fspath(path)
    return os.path.splitext(text)[1].lower()


def guess_video_mime(path: str | os.PathLike[str]) -> str:
    """按扩展名猜 MIME；未知则回落到 video/mp4。"""
    return VIDEO_MIME_BY_SUFFIX.get(suffix_of(path), OUTPUT_MIME)


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
    """读取文件并编码为 data URI。

    Args:
        path: 本地文件路径。
        mime: 声明用的 MIME；留空则按扩展名猜。

    Returns:
        ``data:<mime>;base64,<payload>`` 形式的字符串。

    Raises:
        MediaToolError: 文件不可读、为 0 字节，或编码后超过
            :data:`MAX_DATA_URI_BYTES`（``kind="too_large"``）。
    """
    resolved_mime = mime or guess_video_mime(path)
    try:
        size = os.path.getsize(path)
    except OSError as exc:
        raise MediaToolError(f"无法读取文件: {exc}", kind="other") from exc
    if size <= 0:
        raise MediaToolError(f"文件是 0 字节，无法编码: {path}", kind="empty")
    # 先按体积预测拦一次，避免为了一个必然失败的结果白读几百 MB
    predicted = 4 * ((size + 2) // 3)
    limit_mb = MAX_DATA_URI_BYTES / 1024 / 1024
    if predicted > MAX_DATA_URI_BYTES:
        raise MediaToolError(
            f"编码后约 {predicted / 1024 / 1024:.1f}MB，超过硬上限 {limit_mb:.0f}MB"
            "（请调小 max_size_mb / compress_max_mb，或缩短视频）",
            kind="too_large",
        )
    with open(path, "rb") as handle:
        payload = handle.read()
    encoded = base64.b64encode(payload).decode("ascii")
    if len(encoded) > MAX_DATA_URI_BYTES:
        raise MediaToolError(
            f"编码后 {len(encoded) / 1024 / 1024:.1f}MB，超过硬上限 {limit_mb:.0f}MB"
            "（请调小 max_size_mb / compress_max_mb，或缩短视频）",
            kind="too_large",
        )
    return f"data:{resolved_mime};base64,{encoded}"


#: 引用视频下载的跳数上限（urllib 的默认值是 10）。签名 CDN URL 实测常见 1 跳，
#: 给到 5 足够应付中转站；再多只会把"坏链接"拖得更久。
MAX_REF_REDIRECTS = 5

#: 边下边计的读块大小。**绝不** ``read()`` 整段：一段 500MB 的视频会先吃满内存
#: 再撞上限，而这一层存在的意义正是"超限即停"。
_REF_CHUNK_BYTES = 64 * 1024

#: 部分 CDN 拒绝空 User-Agent。刻意不带版本号：这行 UA 不该随 ``metadata.yaml``
#: 的版本一起漂移。
_REF_USER_AGENT = "astrbot-plugin-video-vision/ref-download"


class _LimitedRedirectHandler(urllib.request.HTTPRedirectHandler):
    """把 urllib 的跳数上限从默认 10 收紧到 :data:`MAX_REF_REDIRECTS`。

    ``urllib`` 的两个计数本来就是类属性：``max_repeats`` 防同一个地址来回跳、
    ``max_redirections`` 防长链。改类属性比复写 ``redirect_request`` 少抄一份
    ``urljoin`` 逻辑 —— 那份逻辑一旦与标准库分叉，异常消息里就可能带上 URL。
    """

    max_repeats = MAX_REF_REDIRECTS
    max_redirections = MAX_REF_REDIRECTS


def _is_http_ref(ref: str) -> bool:
    """``ref`` 是否是 http(s) 地址（大小写不敏感）。"""
    return ref.lower().startswith(("http://", "https://"))


def _ref_suffix(url: str) -> str:
    """从 URL 的 path 里取视频后缀；未知 / 没有后缀时回落到 ``.mp4``。

    后缀决定 ``prepare_video`` 里的 MIME 声明与 corrupt 判决，所以只接受
    :data:`VIDEO_MIME_BY_SUFFIX` 里的已知容器：签名 URL 常以
    ``/download?id=...`` 结尾，那里的 ``.mp4`` 出现在 query 而不是 path 里。
    """
    suffix = suffix_of(urllib.parse.urlsplit(url).path)
    return suffix if suffix in VIDEO_MIME_BY_SUFFIX else OUTPUT_SUFFIX


def _declared_length(response: object) -> int | None:
    """``Content-Length``（**不可信**，只用来在写入前提前拦一刀）。

    服务端可能不写、也可能写错，所以真正的拦截靠下面边下边计的字节数。
    """
    headers = getattr(response, "headers", None)
    raw = headers.get("Content-Length") if hasattr(headers, "get") else None
    try:
        value = int(raw) if raw is not None else 0
    except (TypeError, ValueError):
        return None
    return value if value > 0 else None


def _discard(path: Path) -> None:
    """删掉下到一半的产物：任何失败路径都不留垃圾文件。"""
    try:
        path.unlink()
    except OSError:
        pass


def _open_ref(request: urllib.request.Request, timeout: float) -> Any:
    """打开引用下载请求（跳数受限）。

    独立成函数是为了让测试能替换**唯一**的网络入口：``build_opener`` 返回的
    ``OpenerDirector`` 绕过了 ``urllib.request.urlopen``，monkeypatch 后者是
    拦不住的。生产路径始终走这个函数。
    """
    opener = urllib.request.build_opener(_LimitedRedirectHandler)
    return opener.open(request, timeout=timeout)


def _close_quietly(exc: BaseException) -> None:
    """``HTTPError`` 本身就是一个**未关闭的响应**：不关它等于漏一个 socket。

    失败是常态（403 / 404 / 跳数超限），长跑插件必须自己回收；其它异常没有
    ``close``，直接跳过。
    """
    close = getattr(exc, "close", None)
    if callable(close):
        try:
            close()
        except Exception:  # noqa: BLE001 - 回收失败也不能盖住真正的原因
            pass


def _describe_download_failure(exc: BaseException) -> tuple[str, str]:
    """把下载异常翻译成 ``(message, kind)``，**message 里绝不含 URL**。

    签名 query 里的 token 既不该进日志，也不该进注入给模型的文本。所以这里
    只回异常类型 / HTTP 状态码，不拼 ``ref``，也不转发 ``str(exc)``
    （``ValueError``、``InvalidURL`` 这类异常的原文里会带地址）。
    """
    reason = getattr(exc, "reason", None)
    if isinstance(exc, TimeoutError) or isinstance(reason, TimeoutError):
        return "引用视频下载超时", "timeout"
    if isinstance(exc, urllib.error.HTTPError):
        return f"引用视频下载失败（HTTP {exc.code}）", "other"
    if isinstance(exc, urllib.error.URLError):
        return f"引用视频下载失败（{type(reason or exc).__name__}）", "other"
    return f"引用视频下载失败（{type(exc).__name__}）", "other"


def download_ref_to(
    ref: str, directory: str | os.PathLike[str], timeout: float = 30.0
) -> str:
    """把引用形态（ref）的视频物化成可读的本地路径。

    ``core/placeholder.py`` 的 ref 占位符只给一个引用值 —— 核心的
    ``video.path or video.url or video.file``（``astr_main_agent.py`` 的 quoted
    分支）。它可能是三种东西，本函数各有一条明确出口：

    * ``http(s)://`` —— 下载到 ``directory``（调用方给的**本次请求独占**目录）；
    * 本机路径 / 裸文件名 —— **存在**就原样返回，不做任何网络 I/O；
    * 其它（不存在且不是 http(s)）—— 明确失败，**绝不**把它当 URL 去试探
      （把 ``a1b2.mp4`` 当相对路径去读，错的是那条路径，不是"下载失败"）。

    **只用标准库**：``core/`` 受 ``tests/test_docs.py`` 的 AST 锁约束，不许
    ``import astrbot``；``httpx`` 只是测试依赖（``openai_vision`` 也是惰性、
    可缺）。``urllib`` 零运行时依赖。

    拦截：跳数上限 :data:`MAX_REF_REDIRECTS`；边下边计字节，超过
    :data:`MAX_REF_DOWNLOAD_BYTES` 立刻中断并删掉半成品；``timeout`` 是单次
    socket 操作上限。

    **异常消息绝不含 URL**：只要走到失败分支，消息里只有异常类型 / HTTP 状态码
    （:func:`_describe_download_failure`）或一句固定的中文说明。

    Args:
        ref: 引用值（可能是 URL，也可能是本机路径）。
        directory: 下载目录（调用方负责清理；不存在时会被创建）。
        timeout: 单次 socket 操作上限（秒）。

    Returns:
        本地路径：URL 时是 ``directory`` 下的新文件，本机路径时原样返回。

    Raises:
        MediaToolError: 引用为空、不是 http(s) 且本机不存在、超时、体积超限、
            HTTP 错误或其它下载失败。``kind`` 取
            ``MediaToolError.KINDS`` 里的 ``timeout`` / ``too_large`` / ``empty``
            / ``other``。
    """
    text = (ref or "").strip()
    if not text:
        raise MediaToolError("引用视频地址为空，无法物化", kind="other")
    if not _is_http_ref(text):
        if os.path.isfile(text):
            return text
        raise MediaToolError(
            "引用视频不是 http(s) 地址，且本机不存在该文件", kind="other"
        )

    # 落盘名字用引用的摘要：同一 URL 重试写同一个文件（不会堆垃圾），
    # 且文件名里不出现 URL 的任何片段（日志 / 目录列表里都不会泄漏签名）。
    digest = hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]
    target = Path(directory) / f"ref-{digest}{_ref_suffix(text)}"
    try:
        os.makedirs(directory, exist_ok=True)
    except OSError as exc:
        raise MediaToolError(
            f"引用视频下载目录不可用（{type(exc).__name__}）", kind="other"
        ) from exc

    request = urllib.request.Request(text, headers={"User-Agent": _REF_USER_AGENT})
    limit_mb = MAX_REF_DOWNLOAD_BYTES / 1024 / 1024
    written = 0
    try:
        with _open_ref(request, timeout) as response:
            declared = _declared_length(response)
            if declared is not None and declared > MAX_REF_DOWNLOAD_BYTES:
                raise MediaToolError(
                    f"引用视频超过下载上限 {limit_mb:.1f}MB", kind="too_large"
                )
            with open(target, "wb") as sink:
                while True:
                    chunk = response.read(_REF_CHUNK_BYTES)
                    if not chunk:
                        break
                    written += len(chunk)
                    if written > MAX_REF_DOWNLOAD_BYTES:
                        raise MediaToolError(
                            f"引用视频超过下载上限 {limit_mb:.1f}MB",
                            kind="too_large",
                        )
                    sink.write(chunk)
    except MediaToolError:
        _discard(target)
        raise
    except Exception as exc:  # noqa: BLE001 - 全部翻译成"不带 URL"的 MediaToolError
        _discard(target)
        _close_quietly(exc)
        message, kind = _describe_download_failure(exc)
        raise MediaToolError(message, kind=kind) from exc
    if written <= 0:
        _discard(target)
        raise MediaToolError("引用视频下载到 0 字节", kind="empty")
    return str(target)


#: 超时后等待子进程回收的上限（秒）。孙进程可能仍持有管道，不能无限等。
_KILL_GRACE_SEC = 2.0

#: ffmpeg stderr 里代表「输入损坏 / 不是媒体」的特征串
_CORRUPT_MARKERS = (
    "invalid data found",
    "moov atom not found",
    "could not find codec parameters",
    "error while decoding",
    "is not a valid",
    "no such file or directory",
)


def _looks_corrupt(stderr: str | None) -> bool:
    """stderr 是否指向「文件损坏 / 不是可解码媒体」。"""
    low = (stderr or "").lower()
    return any(marker in low for marker in _CORRUPT_MARKERS)


def _kill_process_tree(proc: subprocess.Popen) -> None:
    """终止子进程**及其后代**。

    Windows: ``proc.kill()`` 只杀直接子进程；指向 ``.cmd`` 包装器时，真正的
    ffmpeg 是 cmd.exe 的孙进程，会继续持有管道。``taskkill /T`` 才是唯一可靠的
    整棵树终止手段，``proc.kill()`` 留作兜底。

    POSIX: ``proc.kill()`` 同样只杀直接子进程。包装脚本（``ffmpeg.sh`` 之类）的
    孙进程会活下来并继续持有管道，于是"超时"仍然要等到孙进程结束才返回 ——
    正是 Windows 上修过的那个 bug。所以 ``_run_ffmpeg`` 用
    ``start_new_session=True`` 让子进程自成进程组，这里对**整个进程组**
    发 ``SIGKILL``（``os.killpg(os.getpgid(pid), SIGKILL)``）。

    .. warning::
       本项目的开发/验证机是 Windows，**没有 WSL 也没有 docker**，所以
       POSIX 这条分支**无法在本机实测** —— 它是"按 POSIX 语义实现 + 源码守卫
       测试"（``tests/test_injector.py::test_kill_process_tree_covers_posix``），
       不是"跑通过"的代码。CI 的 ubuntu lane 会执行 ``start_new_session`` 与
       ``killpg`` 的真实调用（超时用例在 Linux 上不 skip 的部分），但那个结果
       只能由 CI 给出，本地不能替它背书。
    """
    if os.name == "nt":
        try:
            subprocess.run(
                ["taskkill", "/F", "/T", "/PID", str(proc.pid)],
                capture_output=True,
                timeout=5,
            )
        except Exception:  # noqa: BLE001 - 兜底手段失败也不能影响主流程
            pass
    else:
        try:
            os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
        except (ProcessLookupError, PermissionError, OSError):
            # 进程已退出 / 没有权限 / 不是组长：都退回下面的 proc.kill()
            pass
    try:
        proc.kill()
    except Exception:  # noqa: BLE001
        pass


def _run_ffmpeg(args: list[str], timeout: float) -> None:
    """跑一次 ffmpeg，``timeout`` 是**墙钟**上限。

    不能用 ``subprocess.run(timeout=...)``：它超时后只 kill 直接子进程，然后在
    回收管道时**无界阻塞**。实测把 ``ffmpeg_path`` 指向一个 ``.cmd`` 包装器时，
    2 秒的上限实际跑了 25.94s（本轮复测 11.23s，取决于孙进程寿命）——
    ``max_compress_seconds`` 因此形同虚设。

    这里改为 ``Popen`` + ``communicate(timeout)``，超时后终止整棵进程树，再做
    **有界**的二次 ``communicate``（最多 :data:`_KILL_GRACE_SEC` 秒），保证
    "超时" 真的按时返回。

    POSIX 上 ``start_new_session=True`` 让子进程自成进程组，:func:`_kill_process_tree`
    才能用 ``killpg`` 把包装脚本的孙进程一起杀掉（Windows 忽略该参数，
    那边靠 ``taskkill /T``）。

    Args:
        args: 完整命令行（argv[0] 为 ffmpeg 可执行文件）。
        timeout: 墙钟上限（秒），最小 1 秒。

    Raises:
        MediaToolError: 超时（``kind="timeout"``）、无法启动
            （``kind="not_found"``）或退出码非 0（``kind="exit_code"`` /
            ``kind="corrupt"``）。
    """
    limit = max(1.0, timeout)
    try:
        proc = subprocess.Popen(
            args,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8",
            errors="replace",
            # POSIX：自成进程组，超时时整组 SIGKILL（见 _kill_process_tree）。
            # Windows 上该参数被忽略（默认 False），杀树仍靠 taskkill /T。
            start_new_session=(os.name != "nt"),
        )
    except FileNotFoundError as exc:
        raise MediaToolError(f"ffmpeg 无法启动: {exc}", kind="not_found") from exc
    except Exception as exc:
        raise MediaToolError(f"ffmpeg 调用失败: {exc}", kind="other") from exc

    try:
        _, stderr = proc.communicate(timeout=limit)
    except subprocess.TimeoutExpired as exc:
        _kill_process_tree(proc)
        try:
            # 有界二次回收：孙进程若仍持有管道，等不到就放弃（不再阻塞）
            proc.communicate(timeout=_KILL_GRACE_SEC)
        except (subprocess.TimeoutExpired, OSError, ValueError):
            pass
        raise MediaToolError(f"ffmpeg 超时（>{limit:.0f}s）", kind="timeout") from exc

    if proc.returncode != 0:
        tail = (stderr or "").strip().splitlines()[-4:]
        kind = "corrupt" if _looks_corrupt(stderr) else "exit_code"
        raise MediaToolError(
            f"ffmpeg 退出码 {proc.returncode}: {' | '.join(tail)[:300]}", kind=kind
        )


def will_truncate(duration: float, max_duration_sec: float) -> bool:
    """本次压缩是否会**真的**传 ``-t``（即真的截断）。

    抽出来是为了让 ``compress_video`` 与 ``injector`` 用同一个判定：
    ``-t`` 只在「上限 > 0 且源时长确实超过上限」时才生效，否则传了也是 no-op。
    0.2.0 的 ``action`` 标签由 ``overlong`` 单独决定，于是"既超体积又超时长"时
    明明截断了却报告 ``compressed``。
    """
    if not max_duration_sec or max_duration_sec <= 0:
        return False
    return duration > max_duration_sec


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
    实测同参数下静态画面压得极小、高熵画面压得较大）。

    **体积硬上限靠 ``-fs``**：ffmpeg 在写满指定字节数后停止复用，产物最多溢出
    不到一个 GOP（实测 0.5MB 预算得 537KB = 1.03×，0.2MB 这种极小预算约 1.28×）。
    0.2.0 只用 ``-maxrate``/``-bufsize`` 做"保险丝"，但那只是码率上限 ——
    实际会有 1.3~3.2 倍的溢出（音频固定 96k、视频码率下限 120k 在极小预算下
    直接反超预算；同一场景实测 1.49×），所以 ``-fs`` 才是真正的兜底。
    ``-bufsize`` 取 ``1×maxrate``（原本 2× 会让瞬时峰值翻倍）。

    之所以不用 ABR（``-b:v``）精确瞄准体积：体积随内容自适应更符合"送进模型看懂"
    这个目标 —— 简单画面压得更小、上传更快，复杂画面自动保留更多细节。

    Args:
        source: 源文件路径。
        output: 输出文件路径。
        max_mb: 产物体积硬上限（MB）。
        crf: x264 CRF。
        preset: x264 速度预设。
        max_duration_sec: 截断到前 N 秒；0 表示不限（仅在源时长超过它时生效）。
        ffmpeg / ffprobe: 可执行文件路径。
        timeout: 墙钟上限（秒）。

    Returns:
        输出文件路径。

    Raises:
        MediaToolError: 缺 ffmpeg、读不出时长、超时、退出码非 0、产物为空。
    """
    if not ffmpeg:
        raise MediaToolError("未找到 ffmpeg，无法压缩", kind="not_found")

    duration = probe_duration(source, ffprobe)
    if duration <= 0:
        raise MediaToolError(
            "无法读取视频时长，拒绝盲压缩（文件可能已损坏）", kind="duration"
        )

    effective_duration = duration
    if max_duration_sec and max_duration_sec > 0:
        effective_duration = min(duration, max_duration_sec)

    # 体积上限 -> 码率上限（kbps）。留 8% 余量给容器与音频开销。
    budget_kbits = max(1.0, max_mb) * 8192 * 0.92
    total_kbps = budget_kbits / max(0.5, effective_duration)
    # 音频固定 96k，其余给视频，且不低于 120k 以免画质崩坏
    video_kbps = max(120.0, total_kbps - 96.0)
    # -fs 的单位是字节；这是真正的硬上限，与上面的码率估算互为补充。
    # 注意这里**不能**套用上面的 max(1.0, ...) 下限：那是码率估算的保守处理，
    # 用在 -fs 上会让 0.2MB 的预算被悄悄放宽到 1MB。
    hard_limit_bytes = max(1024, int(float(max_mb) * 1024 * 1024))

    args = [ffmpeg, "-y", "-hide_banner", "-loglevel", "error", "-i", str(source)]
    if will_truncate(duration, max_duration_sec):
        args += ["-t", f"{max_duration_sec:.3f}"]
    args += [
        "-c:v", "libx264",
        "-preset", preset,
        "-crf", str(int(crf)),
        "-maxrate", f"{int(video_kbps)}k",
        "-bufsize", f"{int(video_kbps)}k",
        "-pix_fmt", "yuv420p",
        "-c:a", "aac",
        "-b:a", "96k",
        "-movflags", "+faststart",
        "-fs", str(hard_limit_bytes),
        str(output),
    ]
    _run_ffmpeg(args, timeout)

    if not os.path.exists(output) or os.path.getsize(output) == 0:
        raise MediaToolError("ffmpeg 未产出有效文件", kind="empty")
    return str(output)


class CompressCache:
    """按源文件内容哈希缓存压缩结果，避免重试时重复压缩。

    **线程安全**：``get`` / ``put`` / ``_evict`` 共用一把锁，且淘汰只发生在写
    锁内，因此并发 ``put`` 不会互相把对方的条目算错。

    **发布是原子的**：``put`` 先把产物写成目标目录里的临时文件，``os.replace``
    原子改名到目标（同分区 rename，不会出现"截断+重写"的可见窗口），再淘汰
    其它条目（本次写入受 ``protect`` 保护），返回前校验目标确实存在。

    为什么这么改（0.2.0 的 P1）：旧 ``put`` 先 ``shutil.move`` 再**无条件**
    ``_evict()``。单个产物一旦大于 ``cache_max_mb``，淘汰会把刚写进去的产物
    连同所有旧条目一起删掉，然后仍然 ``return str(target)`` —— 上层拿到一个
    不存在的路径，``_make_block`` 抛 ``FileNotFoundError``，视频被静默丢弃
    （实测 ``ok=False 压缩产物编码失败: [Errno 2] ...``，缓存目录被清空）。
    现在这种产物**根本不进缓存**，直接返回临时目录里的路径。
    """

    def __init__(self, root: str | os.PathLike[str], max_mb: int = 512) -> None:
        self.root = Path(root)
        self.max_bytes = max(16, int(max_mb)) * 1024 * 1024
        self._lock = threading.Lock()
        self._tmp_seq = 0

    def _path_for(self, key: str) -> Path:
        return self.root / f"{key}{OUTPUT_SUFFIX}"

    def get(self, key: str) -> str | None:
        """命中则返回路径（并刷新 LRU 时间戳），否则返回 None。"""
        with self._lock:
            candidate = self._path_for(key)
            try:
                if candidate.stat().st_size <= 0:
                    return None
                os.utime(candidate, None)  # 刷新 LRU 时间戳
            except OSError:
                return None
            return str(candidate)

    def put(self, key: str, produced: str | os.PathLike[str]) -> str:
        """把压缩产物移入缓存并返回**确实存在**的路径。

        产物本身大于整个缓存预算（或为空、不可读）时**不缓存**，原样返回产物
        路径 —— 进了缓存也只会被自己挤掉，还会连带清空其它条目。

        Args:
            key: 内容哈希 key。
            produced: 压缩产物路径（通常位于调用方的临时目录）。

        Returns:
            缓存路径或产物原路径；返回的路径正常情况下指向一个存在的文件。
        """
        produced_path = Path(produced)
        try:
            size = produced_path.stat().st_size
        except OSError:
            # 产物在进入缓存前就没了：报给调用方，让它按失败处理
            return str(produced)
        if size <= 0 or size > self.max_bytes:
            return str(produced)

        with self._lock:
            try:
                self.root.mkdir(parents=True, exist_ok=True)
            except OSError:
                return str(produced_path)
            target = self._path_for(key)
            if produced_path == target:
                self._evict(protect=target)
                return self._first_existing(target, produced_path)
            self._tmp_seq += 1
            tmp = target.with_name(
                f".{target.stem}.{os.getpid()}.{self._tmp_seq}.tmp"
            )
            moved = False
            try:
                shutil.move(str(produced_path), str(tmp))
                moved = True
                os.replace(str(tmp), str(target))  # 原子发布
            except Exception:  # noqa: BLE001 - 缓存失败不能影响主流程
                # 尽量把产物还回原位，保证返回的路径有效
                if moved and not produced_path.exists() and tmp.exists():
                    try:
                        shutil.move(str(tmp), str(produced_path))
                    except Exception:  # noqa: BLE001
                        pass
                return self._first_existing(produced_path, target)
            self._evict(protect=target)
            return self._first_existing(target, produced_path)

    @staticmethod
    def _first_existing(*candidates: Path) -> str:
        """返回第一个存在且非空的候选路径；都不在用传入的第一个。"""
        for candidate in candidates:
            try:
                if candidate.stat().st_size > 0:
                    return str(candidate)
            except OSError:
                continue
        return str(candidates[0])

    def _evict(self, protect: Path | None = None) -> None:
        """按 mtime 淘汰最旧条目，直到总体积落回预算。

        Args:
            protect: 本次刚写入的条目，**永不淘汰**。即使它独自超过预算也保留
                （调用方已保证只有 ``size <= max_bytes`` 的产物才会进来）。

        调用方必须持有 ``self._lock``。
        """
        try:
            entries = sorted(
                (p for p in self.root.glob(f"*{OUTPUT_SUFFIX}") if p.is_file()),
                key=lambda p: p.stat().st_mtime,
            )
            total = sum(p.stat().st_size for p in entries)
        except OSError:
            # 统计失败就本轮不淘汰：宁可不淘汰，也不能删错
            return
        for victim in entries:
            if total <= self.max_bytes:
                break
            if protect is not None and victim == protect:
                continue
            try:
                size = victim.stat().st_size
                victim.unlink()
                total -= size
            except OSError:
                continue
