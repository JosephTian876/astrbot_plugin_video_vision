"""引用视频（ref 形态）的物化：下载、去重、临时目录生命周期与失败降级。

背景：引用消息里的视频在本地取不到时，核心只写一行

    [Video Attachment in quoted message: name a1b2.mp4, ref https://.../a1b2.mp4]

0.3.0 完全静默跳过（``_VIDEO_RE`` 只认 ``, path ``）。本 PR 在注入前把它物化成
可读文件：http(s) 下载、本机路径直读、其余明确失败。这里锁六件事（一行一条）：

1. 下载真的落在调用方给的目录里，后缀取自 URL 的 **path**（未知回落 ``.mp4``）；
2. 本机已有的路径**零网络 I/O**；
3. 不存在且不是 http(s) 的引用**明确失败**、不抛异常；
4. 超限在写入前中断，不留半成品；
5. 失败注入文本与短原因都不含 URL（签名 query 不能泄漏）；
6. 同一轮里同一 URL 只下载一次。

HTTP 全部走本机 ``http.server`` 假端点（仓库既有做法）；下载层另外把
``core.media._open_ref``（唯一的网络入口）换成可控的假响应，用于"超限 /
Content-Length 预拦"这类真实服务器不好构造的分支。**不联网**。
"""

from __future__ import annotations

import asyncio
import os
import shutil
import sys
import tempfile
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Iterator

import pytest

from vv_plugin.core import media, placeholder
from vv_plugin.core.config import VideoConfig
from vv_plugin.core.media import (
    MAX_REF_DOWNLOAD_BYTES,
    VIDEO_MIME_BY_SUFFIX,
    MediaToolError,
    download_ref_to,
)
from vv_plugin.hooks import llm_request as hooks

# ============================================================== 本机假端点


class _ServerState:
    """假端点共享状态（每个测试前清空）。"""

    def __init__(self) -> None:
        #: path（含 query）-> bytes | (status, body)
        self.routes: dict[str, Any] = {}
        self.requests: list[str] = []


STATE = _ServerState()


class _Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def do_GET(self) -> None:  # BaseHTTPRequestHandler 的约定名
        STATE.requests.append(self.path)
        route = STATE.routes.get(self.path)
        if route is None:
            self._send(404, b"no route")
            return
        if isinstance(route, tuple):
            status, payload = route
            extra = {"Location": self.path} if 300 <= int(status) < 400 else None
            self._send(int(status), bytes(payload), extra)
            return
        self._send(200, bytes(route))

    def _send(
        self, status: int, data: bytes, extra: dict[str, str] | None = None
    ) -> None:
        self.send_response(status)
        self.send_header("Content-Length", str(len(data)))
        for key, value in (extra or {}).items():
            self.send_header(key, value)
        self.send_header("Connection", "close")
        self.close_connection = True
        self.end_headers()
        try:
            self.wfile.write(data)
        except OSError:  # 客户端提前断开（超限用例）
            pass

    def log_message(self, *args: Any, **kwargs: Any) -> None:  # 静音
        pass


class _QuietServer(ThreadingHTTPServer):
    """客户端提前断开（超限 / 跳数超限）是**预期**行为，不要在输出里喷栈。"""

    def handle_error(self, request: Any, client_address: Any) -> None:
        exc = sys.exc_info()[1]
        if isinstance(exc, (ConnectionError, TimeoutError)):
            return
        super().handle_error(request, client_address)


@pytest.fixture(scope="module")
def server_base() -> Iterator[str]:
    server = _QuietServer(("127.0.0.1", 0), _Handler)
    server.daemon_threads = True
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_address[1]}"
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=10)


@pytest.fixture(autouse=True)
def _clear_routes() -> Iterator[None]:
    STATE.routes.clear()
    STATE.requests.clear()
    yield
    STATE.routes.clear()
    STATE.requests.clear()


# ================================================================= 小工具


class _TextPart:
    def __init__(self, text: str) -> None:
        self.text = text


class _ProviderContext:
    def __init__(self, provider: Any) -> None:
        self._provider = provider

    async def get_using_provider_async(self, umo: Any = None) -> Any:
        return self._provider


def _ref_line(ref: str, name: str = "a1b2.mp4") -> str:
    return f"[Video Attachment in quoted message: name {name}, ref {ref}]"


def _plugin() -> Any:
    """一个能通过 gemini_native 三道闸门的插件替身（不 import 真 AstrBot）。"""
    provider = SimpleNamespace(
        provider_config={
            "type": "googlegenai_chat_completion",
            "id": "gemini-2.5-flash",
            "modalities": ["text", "image", "audio"],
        }
    )
    return SimpleNamespace(
        context=_ProviderContext(provider),
        config=VideoConfig(provider_mode="gemini_native"),
        cache_root="",
    )


def _request(text: str) -> Any:
    return SimpleNamespace(extra_user_content_parts=[_TextPart(text)])


def _run_hook(req: Any, plugin: Any = None) -> None:
    asyncio.run(
        hooks.handle_llm_request(
            plugin or _plugin(), SimpleNamespace(unified_msg_origin="umo"), req
        )
    )


def _record_mkdtemp(monkeypatch: pytest.MonkeyPatch) -> list[str]:
    """记录 hooks 建过的临时目录（用后即删，测试要能证明它被删了）。"""
    created: list[str] = []
    real_mkdtemp = tempfile.mkdtemp

    def recording(*args: Any, **kwargs: Any) -> str:
        path = real_mkdtemp(*args, **kwargs)
        created.append(path)
        return path

    monkeypatch.setattr(hooks.tempfile, "mkdtemp", recording)
    return created


class _FakeResponse:
    """只实现 ``download_ref_to`` 用到的两个成员：``headers`` 与 ``read``。"""

    def __init__(self, chunks: list[bytes], headers: dict[str, str] | None = None) -> None:
        self._chunks = list(chunks)
        self.headers = headers or {}
        self.read_calls = 0

    def read(self, _size: int = -1) -> bytes:
        self.read_calls += 1
        return self._chunks.pop(0) if self._chunks else b""

    def __enter__(self) -> _FakeResponse:
        return self

    def __exit__(self, *exc: object) -> bool:
        return False


def _patch_open_ref(
    monkeypatch: pytest.MonkeyPatch, response: Any
) -> list[Any]:
    """把唯一的网络入口换成假响应，并返回调用记录。"""
    calls: list[Any] = []

    def fake_open_ref(request: Any, timeout: float) -> Any:
        calls.append((request, timeout))
        return response

    monkeypatch.setattr(media, "_open_ref", fake_open_ref)
    return calls


# ===================================================== 1. 下载落点 / 后缀


@pytest.mark.parametrize(
    ("route", "expected_suffix"),
    [
        ("/clip.mp4", ".mp4"),
        ("/clip.WEBM", ".webm"),  # 后缀大小写归一
        ("/download", ".mp4"),  # 没有后缀 -> 回落 mp4
        ("/clip.mp4?sig=a", ".mp4"),  # query 里的后缀不算，path 里的才算
        ("/clip.txt", ".mp4"),  # 不是已知视频容器 -> 回落 mp4
    ],
)
def test_download_writes_into_the_given_directory(
    route: str, expected_suffix: str, server_base: str, tmp_path: Path
) -> None:
    payload = b"0123456789"
    STATE.routes[route] = payload

    target = download_ref_to(f"{server_base}{route}", tmp_path, timeout=5)

    path = Path(target)
    assert path.parent == tmp_path
    assert path.suffix == expected_suffix
    assert path.read_bytes() == payload
    # 文件名只放引用摘要：URL 路径、query、签名都不许出现在目录列表里
    assert path.name.startswith("ref-")
    assert "clip" not in path.name and "sig" not in path.name


def test_the_download_suffix_set_matches_the_placeholder_suffix_set() -> None:
    """``_ref_suffix`` 用 ``VIDEO_MIME_BY_SUFFIX``、``looks_like_video`` 用
    ``VIDEO_SUFFIXES``：两份集合必须逐元素相同，否则"下载后回落 .mp4"的判定
    会与"这个附件算不算视频"分叉。"""
    assert set(VIDEO_MIME_BY_SUFFIX) == set(placeholder.VIDEO_SUFFIXES)


def test_the_download_cap_equals_the_raw_data_uri_limit() -> None:
    """13.5MB 不是随手写的数：它必须与判决侧的 ``URI_LIMIT_RAW_MB`` 同源。

    超过这个体积的视频下载下来也必然在 ``read_as_data_uri`` 处失败，两处
    各写一个数字迟早分叉（一边拦住、另一边已经写满磁盘）。
    """
    from vv_plugin.core.decide import URI_LIMIT_RAW_MB

    assert MAX_REF_DOWNLOAD_BYTES == int(URI_LIMIT_RAW_MB * 1024 * 1024)


# ================================================ 2. 本机路径：零网络 I/O


def test_an_existing_local_path_is_used_without_network_io(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    local = tmp_path / "clip.mp4"
    local.write_bytes(b"local bytes")
    open_calls = _patch_open_ref(monkeypatch, _FakeResponse([b"never"]))
    # 同时把 urllib 的入口也盯住：任何"顺手 urlopen"的写法都必须失败
    urlopen_calls: list[Any] = []
    monkeypatch.setattr(
        media.urllib.request,
        "urlopen",
        lambda *a, **k: urlopen_calls.append(a),
    )

    assert download_ref_to(str(local), tmp_path) == str(local)
    assert open_calls == [] and urlopen_calls == []


# ======================================== 3. 不存在且非 URL：明确失败、不抛


def test_a_missing_local_ref_fails_clearly_without_touching_the_network(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    open_calls = _patch_open_ref(monkeypatch, _FakeResponse([b"never"]))

    with pytest.raises(MediaToolError) as info:
        download_ref_to("a1b2.mp4", tmp_path)

    message = str(info.value)
    assert "http(s)" in message  # 明确说出"这不是可下载地址"
    assert "a1b2.mp4" not in message  # 不回显引用值（与 URL 同一条口径）
    assert open_calls == []
    assert list(tmp_path.iterdir()) == []


def test_a_missing_local_ref_becomes_a_short_reason_not_an_exception() -> None:
    parts = [_TextPart(_ref_line("a1b2.mp4"))]

    directory, failures = hooks._materialize_refs(parts, config=VideoConfig())
    try:
        assert failures == [
            "the quoted video reference is not a downloadable address"
        ]
        assert "a1b2.mp4" not in failures[0]
        # 失败不注入：占位文本保持原样（与既有"失败保留路径"口径一致）
        assert parts[0].text == _ref_line("a1b2.mp4")
    finally:
        shutil.rmtree(directory, ignore_errors=True)


# ============================================== 4. 超限在落盘前中断


def test_an_oversized_stream_aborts_and_keeps_no_partial_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    half = MAX_REF_DOWNLOAD_BYTES // 2
    response = _FakeResponse([b"x" * half, b"x" * (MAX_REF_DOWNLOAD_BYTES - half + 1)])
    _patch_open_ref(monkeypatch, response)

    with pytest.raises(MediaToolError) as info:
        download_ref_to("https://cdn.example/big.mp4", tmp_path)

    assert info.value.kind == "too_large"
    assert "https" not in str(info.value)
    assert list(tmp_path.iterdir()) == [], "超限必须删掉半成品"


def test_a_declared_content_length_over_the_cap_aborts_before_reading(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    response = _FakeResponse(
        [b"should never be read"],
        headers={"Content-Length": str(MAX_REF_DOWNLOAD_BYTES + 1)},
    )
    _patch_open_ref(monkeypatch, response)

    with pytest.raises(MediaToolError) as info:
        download_ref_to("https://cdn.example/big.mp4", tmp_path)

    assert info.value.kind == "too_large"
    assert response.read_calls == 0, "Content-Length 已超限，不应再读一个字节"
    assert list(tmp_path.iterdir()) == []


def test_redirect_chains_are_bounded(server_base: str, tmp_path: Path) -> None:
    """跳数上限是硬约束：指向自己的 302 必须在有限次内失败，不能转圈。"""
    STATE.routes["/loop"] = (302, b"")

    with pytest.raises(MediaToolError) as info:
        download_ref_to(f"{server_base}/loop", tmp_path, timeout=5)

    assert info.value.kind in {"other", "timeout"}
    assert len(STATE.requests) <= media.MAX_REF_REDIRECTS + 1
    assert list(tmp_path.iterdir()) == []


# ========================================== 5. 任何失败文本都不含 URL


@pytest.mark.parametrize(
    ("reason", "kind", "expected"),
    [
        (
            f"{hooks.REF_MATERIALIZE_PREFIX}: 引用视频下载超时",
            "timeout",
            "the quoted video reference timed out while downloading",
        ),
        (
            f"{hooks.REF_MATERIALIZE_PREFIX}: 引用视频超过下载上限 13.5MB",
            "too_large",
            "the quoted video reference is too large to download",
        ),
        (
            f"{hooks.REF_MATERIALIZE_PREFIX}: 引用视频不是 http(s) 地址，"
            "且本机不存在该文件",
            "other",
            "the quoted video reference is not a downloadable address",
        ),
        (
            f"{hooks.REF_MATERIALIZE_PREFIX}: 引用视频下载失败（ConnectionRefusedError）",
            "other",
            "the quoted video reference could not be downloaded",
        ),
    ],
)
def test_ref_failures_map_to_fixed_url_free_phrases(
    reason: str, kind: str, expected: str
) -> None:
    phrase = hooks._short_reason(reason, kind)

    assert phrase == expected
    # 与 ``tests/test_decide.py`` 对短语的字形要求同构：ASCII、无冒号、无反斜杠
    assert phrase.isascii() and ":" not in phrase and "\\" not in phrase
    assert "http" not in phrase


def test_the_injected_notice_never_contains_the_url(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """极端情形：下载层把 URL 拼进了异常消息，注入侧也必须洗掉。"""
    url = "https://cdn.example/v/a1b2.mp4?sign=SECRET"

    def failing(ref: str, directory: str, timeout: float = 30.0) -> str:
        raise MediaToolError(f"boom {ref}", kind="other")

    monkeypatch.setattr(hooks, "download_ref_to", failing)
    req = _request(_ref_line(url))

    _run_hook(req)

    texts = [hooks._text_of(part) or "" for part in req.extra_user_content_parts]
    notices = [text for text in texts if "was skipped" in text]
    assert notices == [
        "[Video attachment was skipped: "
        "the quoted video reference could not be downloaded]"
    ]
    assert "SECRET" not in notices[0] and "cdn.example" not in notices[0]


# ================================= 6. 同一轮里同一 URL 只下载一次


def test_the_same_reference_is_downloaded_once_per_request(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    url = "https://cdn.example/v/a1b2.mp4?sign=abc"
    calls: list[str] = []

    def fake_download(ref: str, directory: str, timeout: float = 30.0) -> str:
        calls.append(ref)
        path = Path(directory) / f"ref-{len(calls)}.mp4"
        path.write_bytes(b"x")
        return str(path)

    monkeypatch.setattr(hooks, "download_ref_to", fake_download)
    parts = [_TextPart(_ref_line(url)), _TextPart(_ref_line(url))]

    directory, failures = hooks._materialize_refs(parts, config=VideoConfig())
    try:
        assert calls == [url], "同一个 URL 一轮只能下一次"
        assert failures == []
        assert Path(directory).is_dir()
        first = placeholder.parse_placeholder_text(parts[0].text)
        assert first is not None and first.kind == "video"
        assert first.path.startswith(directory)
        assert url not in parts[0].text
        # 重复的那个保持原样：不改写就不会把"请求结束后即被删除的临时路径"
        # 留在请求里（``build_video_blocks`` 会把它当作重复视频跳过）
        assert parts[1].text == _ref_line(url)
    finally:
        shutil.rmtree(directory, ignore_errors=True)


def test_two_different_references_are_both_downloaded(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[str] = []

    def fake_download(ref: str, directory: str, timeout: float = 30.0) -> str:
        calls.append(ref)
        path = Path(directory) / f"ref-{len(calls)}.mp4"
        path.write_bytes(b"x")
        return str(path)

    monkeypatch.setattr(hooks, "download_ref_to", fake_download)
    parts = [
        _TextPart(_ref_line("https://cdn.example/v/a.mp4")),
        _TextPart(_ref_line("https://cdn.example/v/b.mp4")),
    ]

    directory, failures = hooks._materialize_refs(parts, config=VideoConfig())
    try:
        assert len(calls) == len(set(calls)) == 2
        assert failures == []
    finally:
        shutil.rmtree(directory, ignore_errors=True)


@pytest.mark.parametrize(
    "options", [{"handle_quoted_video": False}, {"max_video_count": 0}]
)
def test_nothing_is_downloaded_when_the_feature_is_off(
    options: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    def forbidden(*args: Any, **kwargs: Any) -> str:
        raise AssertionError("功能关闭时不该有任何下载")

    monkeypatch.setattr(hooks, "download_ref_to", forbidden)
    parts = [_TextPart(_ref_line("https://cdn.example/v/a.mp4"))]

    assert hooks._materialize_refs(parts, config=VideoConfig(**options)) == ("", [])
    assert parts[0].text == _ref_line("https://cdn.example/v/a.mp4")


@pytest.mark.parametrize(
    "options", [{"handle_quoted_video": False}, {"max_video_count": 0}]
)
def test_the_hook_returns_before_downloading_when_disabled(
    options: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    """预检与物化读同一处判定：功能关掉时连 ``_materialize_refs`` 都不进。"""

    def forbidden(*args: Any, **kwargs: Any) -> str:
        raise AssertionError("功能关闭时不该有任何下载")

    monkeypatch.setattr(hooks, "download_ref_to", forbidden)
    plugin = SimpleNamespace(
        context=_ProviderContext(None),
        config=VideoConfig(provider_mode="gemini_native", **options),
        cache_root="",
    )
    req = _request(_ref_line("https://cdn.example/v/a.mp4"))

    _run_hook(req, plugin)

    assert len(req.extra_user_content_parts) == 1
    assert req.extra_user_content_parts[0].text == _ref_line(
        "https://cdn.example/v/a.mp4"
    )


# ============================== 临时目录生命周期（建 / 用 / 删）


def test_the_directory_exists_while_preparing_and_is_gone_after(
    server_base: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    created = _record_mkdtemp(monkeypatch)
    STATE.routes["/tiny.mp4"] = b"fake video bytes"
    seen: dict[str, Any] = {}

    async def fake_build_video_blocks(parts: list[Any], **_kwargs: Any) -> Any:
        text = hooks._text_of(parts[0]) or ""
        parsed = placeholder.parse_placeholder_text(text)
        assert parsed is not None, text
        assert parsed.kind == "video", "物化后必须是 path 形态"
        seen["path"] = parsed.path
        seen["exists_during"] = os.path.isfile(parsed.path)
        return [], set(), []

    monkeypatch.setattr(hooks, "build_video_blocks", fake_build_video_blocks)
    req = _request(_ref_line(f"{server_base}/tiny.mp4"))

    _run_hook(req)

    assert seen["exists_during"] is True, "准备阶段文件必须真的在磁盘上"
    assert created, "应当建了一个 vv_ref_ 临时目录"
    assert not os.path.exists(seen["path"]), "准备结束后临时文件必须删掉"
    assert not os.path.exists(created[0]), "临时目录本身也必须删掉"


def test_a_quoted_ref_video_is_really_injected(tiny_video: str, server_base: str) -> None:
    """端到端：下载 → prepare_video → 内容块 → 占位文本被改写。"""
    STATE.routes["/tiny.mp4"] = Path(tiny_video).read_bytes()
    req = _request(_ref_line(f"{server_base}/tiny.mp4", "tiny.mp4"))

    _run_hook(req)

    parts = req.extra_user_content_parts
    assert len(parts) == 2, [type(item).__name__ for item in parts]
    assert "data:video/mp4;base64," in parts[1].url
    assert parts[1].model_dump_for_context()["_no_save"] is True
    # 成功注入后占位文本被改写成"已提供"说明：URL 与本机路径都不再出现
    assert "cdn" not in parts[0].text and "127.0.0.1" not in parts[0].text
    assert "tiny.mp4" in parts[0].text
    assert STATE.requests == ["/tiny.mp4"]
