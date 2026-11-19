"""0.4.1 可靠性、安全性与兼容性修复回归测试。

全离线、不依赖真实公网与外部凭据；涉及真实转码的单测复用 ``conftest``
动态生成的 ``tiny_video`` 夹具。覆盖：

1. 纯附件 ``<attachment>`` 占位词识别与问答提示词分流；
2. OpenAI 兼容服务配置校验、端点保留 query / 裸域名补全、多模态与思考段解析、
   ``504 Time-out`` 分类、错误摘要 API Key 脱敏、空白模型名兜底与状态接口联动；
3. ``selected_provider`` 单轮临时切模优先解析与回退；
4. 引用视频配额前置、压缩模式 100MB 放宽、SSRF 防护与重复 ``ref``/``path`` 脱敏；
5. ``AsyncSingleFlight`` 预占槽并发合并与总体等待超时保护；
6. ``.ts`` 从视频扩展名剔除、``prepare_video`` 消除重复探测、``_DONE_ATTR`` 后置。
"""

from __future__ import annotations

import asyncio
import ipaddress
import socket
import tempfile
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from vv_plugin.core import diagnostics, media, openai_vision, placeholder
from vv_plugin.core.config import (
    DEFAULT_OPENAI_MODEL,
    VideoConfig,
    is_valid_openai_api_key,
    is_valid_openai_base_url,
)
from vv_plugin.core.injector import prepare_video
from vv_plugin.core.limits import AsyncSingleFlight, SingleFlight, SlotLimiter
from vv_plugin.hooks import llm_request as hooks
from vv_plugin.hooks.status import VideoVisionStatusAPI, mode_is_usable


def _cfg(**overrides: Any) -> VideoConfig:
    base = dict(
        enable=True,
        provider_mode="openai_compat",
        openai_base_url="https://vision.example.com/v1",
        openai_api_key="sk-test-token-1234",
        openai_model=DEFAULT_OPENAI_MODEL,
        max_size_mb=13.0,
        compress_max_mb=8.0,
        max_duration_sec=120.0,
        oversize_action="compress",
        handle_quoted_video=True,
        max_video_count=2,
        ffmpeg_singleflight=True,
        cache_enabled=True,
    )
    base.update(overrides)
    return VideoConfig(**base)


def _part(text: str) -> SimpleNamespace:
    return SimpleNamespace(type="text", text=text)


def _req(parts: list[Any], *, prompt: str = "") -> SimpleNamespace:
    return SimpleNamespace(
        prompt=prompt,
        contexts=[],
        extra_user_content_parts=parts,
        video_urls=[],
    )


def _event(
    *,
    message_str: str = "",
    selected_provider: str | None = None,
    umo: str = "test:group:1",
) -> SimpleNamespace:
    extra = {} if selected_provider is None else {"selected_provider": selected_provider}
    return SimpleNamespace(
        message_str=message_str,
        unified_msg_origin=umo,
        get_message_str=lambda: message_str,
        get_extra=lambda key, default=None: extra.get(key, default),
    )


# ============================================== 1. <attachment> 占位词过滤


class TestAttachmentPromptFiltering:
    def test_ignores_bare_attachment_placeholder_in_req_prompt(self) -> None:
        req = _req([], prompt="<attachment>")
        event = _event(message_str="")
        assert hooks._extract_question(event, req) == ""

    def test_ignores_bare_attachment_placeholder_in_event_text(self) -> None:
        req = _req([], prompt="备用问题")
        event = _event(message_str="  <attachment>  ")
        assert hooks._extract_question(event, req) == "备用问题"

        req_both = _req([], prompt=" <attachment>\n ")
        assert hooks._extract_question(_event(message_str="  <attachment>  "), req_both) == ""

    def test_strips_standalone_attachment_line_while_keeping_question(self) -> None:
        event = _event(
            message_str=(
                "[Video Attachment: name demo.mp4, path /tmp/demo.mp4]\n"
                "<attachment>\n"
                "这段视频里黑板写了什么字？"
            )
        )
        assert hooks._extract_question(event, _req([])) == "这段视频里黑板写了什么字？"

    def test_preserves_question_that_mentions_attachment_tag_inline(self) -> None:
        event = _event(message_str="请问这个<attachment>片段讲了什么？")
        assert hooks._extract_question(event, _req([])) == "请问这个<attachment>片段讲了什么？"


# ====================================== 2. 视觉服务配置 / 端点 / 响应 / 脱敏


class TestOpenAIVisionHardening:
    @pytest.mark.parametrize(
        ("url", "expected"),
        [
            ("https://api.example.com", True),
            ("https://api.example.com/v1/chat/completions", True),
            ("http://127.0.0.1:8080/responses?api-version=2025-01-01", True),
            ("", False),
            ("   ", False),
            ("api.example.com/v1", False),
            ("ftp://api.example.com/v1", False),
            ("file:///etc/passwd", False),
            ("https://", False),
        ],
    )
    def test_validates_openai_base_url(self, url: str, expected: bool) -> None:
        assert is_valid_openai_base_url(url) is expected

    @pytest.mark.parametrize(
        ("key", "expected"),
        [
            ("sk-valid-key-9876", True),
            ("bearer-or-opaque-secret", True),
            ("", False),
            ("   ", False),
            ("https://proxy.example.com/v1", False),
            ("HTTP://127.0.0.1:8080", False),
        ],
    )
    def test_validates_openai_api_key(self, key: str, expected: bool) -> None:
        assert is_valid_openai_api_key(key) is expected

    def test_invalid_credentials_trigger_downgrade_and_diagnostics_without_leak(
        self,
    ) -> None:
        bad_key = _cfg(openai_api_key="https://secret.example.com/hook")
        assert bad_key.openai_service_ready is False
        assert bad_key.effective_provider_mode == "gemini_native"
        missing = diagnostics.missing_credentials(bad_key)
        assert any("不可为 URL" in item for item in missing)
        assert "secret.example.com" not in " ".join(missing)

        bad_url = _cfg(openai_base_url="not-a-url")
        assert bad_url.openai_service_ready is False
        assert any("http(s)" in item for item in diagnostics.missing_credentials(bad_url))

        # 状态接口在无效凭据时降级到 gemini_native（仍同步返回四字段）
        plugin = SimpleNamespace(config=bad_key)
        status = VideoVisionStatusAPI(plugin).companion_v2_status()
        assert status["schema"] == "companion.provider-status.v1"
        assert status["runtime_ready"] is True
        assert mode_is_usable(bad_key) is True

    def test_blank_openai_model_falls_back_to_default(self) -> None:
        cfg, _ = VideoConfig.from_mapping_detailed({"openai_model": "   \t  "})
        assert cfg.openai_model == DEFAULT_OPENAI_MODEL

    @pytest.mark.parametrize(
        ("raw_url", "expected_proto", "expected_url"),
        [
            (
                "https://proxy.example.com",
                "chat",
                "https://proxy.example.com/v1/chat/completions",
            ),
            (
                "https://proxy.example.com/",
                "chat",
                "https://proxy.example.com/v1/chat/completions",
            ),
            (
                "https://proxy.example.com/v1",
                "chat",
                "https://proxy.example.com/v1/chat/completions",
            ),
            (
                "https://proxy.example.com/custom/v2",
                "chat",
                "https://proxy.example.com/custom/v2/chat/completions",
            ),
            (
                "https://proxy.example.com/custom/v2/chat/completions",
                "chat",
                "https://proxy.example.com/custom/v2/chat/completions",
            ),
            (
                "https://azure.example.com/openai/deployments/v?api-version=2025-01-01",
                "chat",
                "https://azure.example.com/openai/deployments/v/chat/completions?api-version=2025-01-01",
            ),
            (
                "https://proxy.example.com/v1/responses?api-version=2025-01-01",
                "responses",
                "https://proxy.example.com/v1/responses?api-version=2025-01-01",
            ),
        ],
    )
    def test_resolve_endpoint_handles_bare_domain_and_preserves_query(
        self, raw_url: str, expected_proto: str, expected_url: str
    ) -> None:
        assert openai_vision._resolve_endpoint(raw_url) == (
            expected_proto,
            expected_url,
        )

    def test_parse_response_handles_multimodal_content_list_and_filters_thinking(
        self,
    ) -> None:
        payload = {
            "choices": [
                {
                    "message": {
                        "content": [
                            None,
                            {"type": "thinking", "text": "内部推理不应返回"},
                            {"type": "thought", "thought": "secret reasoning"},
                            {"type": "reasoning", "text": "skip me"},
                            {"type": "text", "text": "第一段描述"},
                            {"type": "output_text", "text": "第二段描述"},
                        ]
                    }
                }
            ],
            "usage": {"prompt_tokens": 120, "completion_tokens": "45"},
        }
        text, p_tok, c_tok = openai_vision._parse_response("chat", payload)
        assert text == "第一段描述\n第二段描述"
        assert "内部推理" not in text
        assert "secret reasoning" not in text
        assert (p_tok, c_tok) == (120, 45)

        empty_payload = {"choices": [{"message": {"content": None}}], "usage": None}
        assert openai_vision._parse_response("chat", empty_payload) == ("", 0, 0)

        with pytest.raises(ValueError, match="不是对象"):
            openai_vision._parse_response("chat", ["unexpected"])

    def test_504_timeout_classification_and_secret_redaction(self) -> None:
        assert "timed out" in hooks._short_reason(
            "视觉服务返回 HTTP 504: 504 Gateway Time-out"
        )
        msg = openai_vision._classify_error(
            RuntimeError("504 Gateway Time-out at https://x/v?key=SECRETKEY99"),
            api_key="sk-real-secret-token",
        )
        assert "超时" in msg

        secret = "sk-my-very-secret-key-8899"
        leaked = f"Authorization: Bearer {secret} &api_key=abc123 failed"
        redacted = openai_vision._redact_secret(leaked, secret)
        assert secret not in redacted
        assert "abc123" not in redacted
        assert "***" in redacted


# ==================================== 3. selected_provider 单轮切模优先解析


class TestSelectedProviderResolution:
    def test_prefers_selected_provider_from_event_extra_sync_and_async(self) -> None:
        default_prov = SimpleNamespace(name="default")
        chosen_prov = SimpleNamespace(name="chosen")

        async def get_using_provider_async(umo: str = "") -> Any:
            del umo
            return default_prov

        context = SimpleNamespace(
            get_provider_by_id=lambda pid: chosen_prov if pid == "prov-override" else None,
            get_using_provider_async=get_using_provider_async,
        )
        plugin = SimpleNamespace(context=context)

        prov, err = asyncio.run(
            hooks.resolve_provider(plugin, _event(selected_provider="prov-override"))
        )
        assert prov is chosen_prov
        assert err == ""

        # async get_provider_by_id 同样兼容
        async def async_by_id(pid: str) -> Any:
            return chosen_prov if pid == "prov-async" else None

        context_async = SimpleNamespace(
            get_provider_by_id=async_by_id,
            get_using_provider_async=get_using_provider_async,
        )
        prov2, err2 = asyncio.run(
            hooks.resolve_provider(
                SimpleNamespace(context=context_async),
                _event(selected_provider="prov-async"),
            )
        )
        assert prov2 is chosen_prov
        assert err2 == ""

    def test_falls_back_to_using_provider_when_selected_provider_absent_or_missing(
        self,
    ) -> None:
        default_prov = SimpleNamespace(name="default")

        async def get_using_provider_async(umo: str = "") -> Any:
            del umo
            return default_prov

        context = SimpleNamespace(
            get_provider_by_id=lambda _pid: None,
            get_using_provider_async=get_using_provider_async,
        )
        plugin = SimpleNamespace(context=context)
        prov, err = asyncio.run(
            hooks.resolve_provider(plugin, _event(selected_provider="unknown-id"))
        )
        assert prov is default_prov
        assert err == ""


# ====================== 4. 引用视频：配额前置、100MB 放宽、SSRF 与重复脱敏


class TestRefSecurityQuotaAndDedup:
    def test_quota_skips_over_limit_refs_with_zero_network_calls(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        local_video = tmp_path / "first.mp4"
        local_video.write_bytes(b"\x00\x00\x00\x18ftypmp42")
        parts = [
            _part(f"[Video Attachment: name first.mp4, path {local_video}]"),
            _part(
                "[Video Attachment in quoted message: name second.mp4, "
                "ref https://cdn.example/second.mp4?sign=sec123]"
            ),
        ]
        calls: list[str] = []

        def fake_download(ref: str, directory: str, **_kwargs: Any) -> str:
            calls.append(ref)
            raise AssertionError("超配额 ref 不应触发下载")

        monkeypatch.setattr(hooks, "download_ref_to", fake_download)
        directory, reasons = hooks._materialize_refs(parts, config=_cfg(max_video_count=1))
        assert directory == ""
        assert reasons == []
        assert calls == []

    def test_ref_download_limit_relaxes_to_100mb_only_when_ffmpeg_ready(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(
            hooks, "resolve_ffmpeg_tools", lambda _path="": ("/bin/ffmpeg", "/bin/ffprobe")
        )
        assert (
            hooks._ref_download_limit(_cfg(oversize_action="compress"))
            == media.MAX_REF_COMPRESS_DOWNLOAD_BYTES
        )
        assert (
            hooks._ref_download_limit(_cfg(oversize_action="truncate"))
            == media.MAX_REF_COMPRESS_DOWNLOAD_BYTES
        )
        assert (
            hooks._ref_download_limit(_cfg(oversize_action="reject"))
            == media.MAX_REF_DOWNLOAD_BYTES
        )

        monkeypatch.setattr(hooks, "resolve_ffmpeg_tools", lambda _path="": (None, None))
        assert (
            hooks._ref_download_limit(_cfg(oversize_action="compress"))
            == media.MAX_REF_DOWNLOAD_BYTES
        )

        # 验证 _call_download_ref 确实向支持 max_bytes 的实现传递放宽上限
        seen_limits: list[int | None] = []

        def capture_download(
            ref: str, directory: str, *, timeout: float = 30.0, max_bytes: int | None = None
        ) -> str:
            del ref, directory, timeout
            seen_limits.append(max_bytes)
            return "/tmp/ok.mp4"

        monkeypatch.setattr(hooks, "download_ref_to", capture_download)
        hooks._call_download_ref(
            "https://cdn.example/big.mp4", "/tmp", media.MAX_REF_COMPRESS_DOWNLOAD_BYTES
        )
        assert seen_limits == [media.MAX_REF_COMPRESS_DOWNLOAD_BYTES]

    def test_ssrf_blocks_dangerous_urls_and_unauthorized_local_paths(
        self, tmp_path: Path
    ) -> None:
        decimal_loopback = int(ipaddress.IPv4Address("127.0.0.1"))
        blocked_urls = [
            "file:///etc/passwd",
            "http://127.0.0.1/secret.mp4",
            "http://localhost/secret.mp4",
            "http://localhost.localdomain/secret.mp4",
            "http://metadata.google.internal/secret.mp4",
            "http://169.254.169.254/latest/secret.mp4",
            "http://10.0.0.8/secret.mp4",
            "http://192.168.1.50/secret.mp4",
            "http://[::1]/secret.mp4",
            "http://[::ffff:127.0.0.1]/secret.mp4",
            "http://198.18.0.1/literal-fake-ip.mp4",
            f"http://{decimal_loopback}/secret.mp4",
            "http://0x7f.0.0.1/secret.mp4",
            "https://user:pass@cdn.example/secret.mp4",
        ]
        for url in blocked_urls:
            with pytest.raises(media.MediaToolError):
                media.download_ref_to(url, tmp_path)

        # 本机路径边界：非视频后缀与目录外无后缀文件必须拦截，临时目录无后缀 & 授权视频后缀放行
        json_file = tmp_path / "secret.json"
        json_file.write_text("{}", encoding="utf-8")
        with pytest.raises(media.MediaToolError):
            media.download_ref_to(str(json_file), tmp_path)

        non_temp_dir = Path.cwd()
        assert non_temp_dir != Path(tempfile.gettempdir()).resolve()
        outside_blob = tmp_path / "noext_in_tmp_test"
        outside_blob.write_bytes(b"data")
        # 在 pytest tmp_path（若不在 system tempdir）下按 _is_authorized_local_ref 检查
        if Path(tempfile.gettempdir()).resolve() in tmp_path.resolve().parents:
            assert media.download_ref_to(str(outside_blob), tmp_path) == str(outside_blob)
        else:
            with pytest.raises(media.MediaToolError):
                media.download_ref_to(str(outside_blob), Path.cwd())

        # 下载目录自身里的无后缀临时文件始终被放行
        in_dl_dir = tmp_path / "downloaded_blob"
        in_dl_dir.write_bytes(b"video-bytes")
        assert media.download_ref_to(str(in_dl_dir), tmp_path) == str(in_dl_dir)

    def test_ssrf_dns_allows_fake_ip_and_blocks_private_dns(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # 域名解析到 Clash 198.18.0.0/15 假 IP 时放行（_validate_http_ref_url 不抛异常）
        monkeypatch.setattr(
            socket,
            "getaddrinfo",
            lambda *_a, **_kw: [(socket.AF_INET, socket.SOCK_STREAM, 6, "", ("198.18.0.42", 443))],
        )
        media._validate_http_ref_url("https://cdn.example/video.mp4", resolve_dns=True)

        # 域名解析到私网 10.0.0.9 时立即拒绝
        monkeypatch.setattr(
            socket,
            "getaddrinfo",
            lambda *_a, **_kw: [(socket.AF_INET, socket.SOCK_STREAM, 6, "", ("10.0.0.9", 443))],
        )
        with pytest.raises(media.MediaToolError, match="受限内网 IP"):
            media._validate_http_ref_url("https://rebind.example/video.mp4", resolve_dns=True)

    def test_ssrf_blocks_redirect_to_private_or_loopback_ip(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # HTTP 302 跳转到受限地址时在跳转拦截器里中止
        class _RedirectHandler(BaseHTTPRequestHandler):
            def do_GET(self) -> None:
                self.send_response(302)
                self.send_header("Location", "http://169.254.169.254/secret.mp4")
                self.end_headers()

            def log_message(self, format: str, *args: Any) -> None:
                del format, args

        server = ThreadingHTTPServer(("127.0.0.1", 0), _RedirectHandler)
        server_port = server.server_address[1]
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            monkeypatch.setattr(
                media,
                "_is_allowed_test_ref_host",
                lambda host, port: host == "127.0.0.1" and port == server_port,
            )
            with pytest.raises(media.MediaToolError, match="受限内网 IP"):
                media.download_ref_to(
                    f"http://127.0.0.1:{server_port}/go.mp4", tmp_path, timeout=5
                )
        finally:
            server.shutdown()
            server.server_close()
            thread.join(timeout=10)

    def test_duplicate_ref_placeholders_strip_signed_urls_after_handle(
        self, tiny_video: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        raw_bytes = Path(tiny_video).read_bytes()
        secret_url = "https://cdn.example/quoted.mp4?sign=TOPSECRET_SIG_TOKEN"
        download_calls: list[str] = []

        def fake_download(ref: str, directory: str, **_kwargs: Any) -> str:
            download_calls.append(ref)
            local = Path(directory) / "quoted.mp4"
            local.write_bytes(raw_bytes)
            return str(local)

        monkeypatch.setattr(hooks, "download_ref_to", fake_download)
        line = (
            "[Video Attachment in quoted message: name quoted.mp4, "
            f"ref {secret_url}]"
        )
        req = _req([_part(line), _part(line)])
        cfg = _cfg(provider_mode="gemini_native", hide_local_path=True, require_gemini_provider=False)
        prov = SimpleNamespace(meta=lambda: SimpleNamespace(type="google_genai"))

        async def get_using_provider_async(umo: str = "") -> Any:
            del umo
            return prov

        plugin = SimpleNamespace(
            config=cfg,
            cache_root=str(tmp_path),
            limiter=None,
            vision_cache=None,
            context=SimpleNamespace(get_using_provider_async=get_using_provider_async),
        )
        asyncio.run(hooks._handle(plugin, _event(), req))

        assert download_calls == [secret_url]
        texts = [p.text for p in req.extra_user_content_parts if hasattr(p, "text")]
        assert all("TOPSECRET_SIG_TOKEN" not in t for t in texts)
        assert all("https://cdn.example" not in t for t in texts)
        assert sum("已随本消息直接提供" in t for t in texts) == 2

    def test_duplicate_path_placeholders_both_neutralize_local_paths(
        self, tiny_video: str, tmp_path: Path
    ) -> None:
        line = f"[Video Attachment: name tiny.mp4, path {tiny_video}]"
        req = _req([_part(line), _part(line)])
        cfg = _cfg(
            provider_mode="gemini_native",
            hide_local_path=True,
            require_gemini_provider=False,
        )
        prov = SimpleNamespace(meta=lambda: SimpleNamespace(type="google_genai"))

        async def get_using_provider_async(umo: str = "") -> Any:
            del umo
            return prov

        plugin = SimpleNamespace(
            config=cfg,
            cache_root=str(tmp_path),
            limiter=None,
            vision_cache=None,
            context=SimpleNamespace(get_using_provider_async=get_using_provider_async),
        )
        asyncio.run(hooks._handle(plugin, _event(), req))

        texts = [p.text for p in req.extra_user_content_parts if hasattr(p, "text")]
        assert all(tiny_video not in t for t in texts)
        assert sum("已随本消息直接提供" in t for t in texts) == 2


# ============================ 5. 预占槽 AsyncSingleFlight 与总体超时保护


class TestPreflightAndOverallTimeout:
    def test_async_singleflight_coalesces_before_slot_without_blocking_other_video(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        vid_a = tmp_path / "a.mp4"
        vid_a.write_bytes(b"video-content-alpha")
        vid_b = tmp_path / "b.mp4"
        vid_b.write_bytes(b"video-content-bravo")

        limiter = SlotLimiter(limit=1, queue_timeout=2.0)
        cfg = _cfg(ffmpeg_singleflight=True, cache_enabled=True)

        a_started = threading.Event()
        release_a = threading.Event()
        calls: list[str] = []

        def fake_prepare(placeholder_obj: Any, **_kwargs: Any) -> hooks.PrepareOutcome:
            calls.append(placeholder_obj.name)
            if placeholder_obj.name == "a.mp4":
                a_started.set()
                release_a.wait(timeout=5.0)
            inj = hooks.VideoInjection(
                block={
                    "type": "audio_url",
                    "audio_url": {"url": "data:video/mp4;base64,QUFB"},
                },
                size_mb=0.01,
                duration_sec=1.0,
                action="as-is",
                source_path=placeholder_obj.path,
                used_path=placeholder_obj.path,
            )
            return hooks.PrepareOutcome(True, inj)

        monkeypatch.setattr(hooks, "prepare_video", fake_prepare)

        async def scenario() -> tuple[Any, Any, Any]:
            t_a1 = asyncio.create_task(
                hooks.build_video_blocks(
                    [_part(f"[Video Attachment: name a.mp4, path {vid_a}]")],
                    config=cfg,
                    cache_root=str(tmp_path),
                    limiter=limiter,
                )
            )
            # 等待 leader a1 进入 worker 槽
            while not a_started.is_set():
                await asyncio.sleep(0.01)

            # follower a2 命中 AsyncSingleFlight（事件循环侧等待），绝不占用 SlotLimiter 槽位
            t_a2 = asyncio.create_task(
                hooks.build_video_blocks(
                    [_part(f"[Video Attachment: name a.mp4, path {vid_a}]")],
                    config=cfg,
                    cache_root=str(tmp_path),
                    limiter=limiter,
                )
            )
            await asyncio.sleep(0.03)
            assert not t_a2.done()
            assert limiter.waiters == 0
            assert limiter.inflight == 1

            # 释放 a1 后，a2 共享产物、b 顺利拿到唯一槽位执行
            release_a.set()
            t_b = asyncio.create_task(
                hooks.build_video_blocks(
                    [_part(f"[Video Attachment: name b.mp4, path {vid_b}]")],
                    config=cfg,
                    cache_root=str(tmp_path),
                    limiter=limiter,
                )
            )
            return await asyncio.gather(t_a1, t_a2, t_b)

        try:
            (prep_a1, _, fail_a1), (prep_a2, _, fail_a2), (prep_b, _, fail_b) = (
                asyncio.run(scenario())
            )
        finally:
            release_a.set()
            limiter.shutdown(wait=False)

        assert fail_a1 == [] and fail_a2 == [] and fail_b == []
        assert len(prep_a1) == 1 and len(prep_a2) == 1 and len(prep_b) == 1
        assert calls.count("a.mp4") == 1
        assert calls.count("b.mp4") == 1
        assert limiter.stats.waits == 0

    def test_singleflight_overall_timeout_guards_against_infinite_hang(self) -> None:
        sync_flight: SingleFlight[str] = SingleFlight()
        started = threading.Event()
        gate = threading.Event()

        def stuck_sync() -> str:
            started.set()
            gate.wait(timeout=2.0)
            return "late"

        leader = threading.Thread(
            target=lambda: sync_flight.run("key-sync", stuck_sync, overall_timeout=2.0),
            daemon=True,
        )
        leader.start()
        assert started.wait(timeout=2.0)
        try:
            with pytest.raises(TimeoutError):
                sync_flight.run("key-sync", stuck_sync, overall_timeout=0.05)
        finally:
            gate.set()
            leader.join(timeout=2.0)
        assert sync_flight.pending == 0

        # 异步 SingleFlight 跟随者超时同样触发 TimeoutError，且 leader 完成后状态清零
        async def async_scenario() -> str:
            async_flight: AsyncSingleFlight[str] = AsyncSingleFlight()
            async_started = asyncio.Event()
            block = asyncio.Event()

            async def stuck_async() -> str:
                async_started.set()
                await block.wait()
                return "late"

            t_leader = asyncio.create_task(
                async_flight.run("key-async", stuck_async, wait_timeout=2.0)
            )
            await async_started.wait()
            with pytest.raises(TimeoutError):
                await async_flight.run("key-async", stuck_async, wait_timeout=0.05)
            block.set()
            await t_leader
            assert async_flight.pending == 0

            async def fast_async() -> str:
                return "recovered"

            val, _ = await async_flight.run("key-async", fast_async, wait_timeout=1.0)
            return val

        assert asyncio.run(async_scenario()) == "recovered"


# ====================== 6. .ts 剔除、去重探测与 _DONE_ATTR 异常处理


class TestTsExclusionSingleProbeAndDoneAttr:
    def test_ts_suffix_is_excluded_from_video_detection(self, tmp_path: Path) -> None:
        assert ".ts" not in placeholder.VIDEO_SUFFIXES
        assert ".ts" not in media.VIDEO_MIME_BY_SUFFIX

        ts_file = tmp_path / "App.ts"
        ts_file.write_text("export const x: number = 1;\n", encoding="utf-8")
        parts = [_part(f"[File Attachment: name App.ts, path {ts_file}]")]

        prepared, consumed, failures = asyncio.run(
            hooks.build_video_blocks(
                parts, config=_cfg(treat_file_as_video=True), cache_root=str(tmp_path)
            )
        )
        assert prepared == []
        assert consumed == set()
        assert failures == []

    def test_prepare_video_probes_duration_only_once_across_compression(
        self, tiny_video: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        injector_probes: list[str] = []
        media_probes: list[str] = []
        compress_durations: list[float | None] = []
        real_probe = media.probe_duration
        real_compress = media.compress_video

        def spy_injector_probe(path: str, *args: Any, **kwargs: Any) -> float:
            injector_probes.append(path)
            return real_probe(path, *args, **kwargs)

        def spy_media_probe(path: str, *args: Any, **kwargs: Any) -> float:
            media_probes.append(path)
            return real_probe(path, *args, **kwargs)

        def spy_compress(*args: Any, **kwargs: Any) -> str:
            compress_durations.append(kwargs.get("duration"))
            return real_compress(*args, **kwargs)

        monkeypatch.setattr("vv_plugin.core.injector.probe_duration", spy_injector_probe)
        monkeypatch.setattr("vv_plugin.core.media.probe_duration", spy_media_probe)
        monkeypatch.setattr("vv_plugin.core.injector.compress_video", spy_compress)

        # 用 0.001MB 门限强制走压缩路径（tiny_video 约 ~0.04MB），验证源文件只探测 1 次
        cfg = _cfg(
            max_size_mb=0.001,
            compress_max_mb=0.0005,
            oversize_action="compress",
            cache_enabled=False,
        )
        ph = placeholder.VideoPlaceholder(name="tiny.mp4", path=tiny_video)
        outcome = prepare_video(ph, config=cfg, cache_root=str(tmp_path))
        assert outcome.ok or outcome.kind != "other"
        assert injector_probes.count(tiny_video) == 1
        assert media_probes == []
        assert compress_durations and compress_durations[0] is not None
        assert compress_durations[0] > 0.0

    def test_done_attr_is_not_set_on_midflight_exception(
        self, tiny_video: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        cfg = _cfg(provider_mode="openai_compat")

        async def exploding_analyze(*_args: Any, **_kwargs: Any) -> list[Any]:
            raise RuntimeError("transient mid-flight failure")

        monkeypatch.setattr(hooks, "_analyze_prepared_videos", exploding_analyze)

        async def get_using_provider_async(umo: str = "") -> Any:
            del umo
            return SimpleNamespace(meta=lambda: SimpleNamespace(type="openai"))

        plugin = SimpleNamespace(
            config=cfg,
            cache_root=str(tmp_path),
            limiter=None,
            vision_cache=None,
            context=SimpleNamespace(get_using_provider_async=get_using_provider_async),
        )
        req = _req([_part(f"[Video Attachment: name tiny.mp4, path {tiny_video}]")])

        with pytest.raises(RuntimeError, match="transient mid-flight failure"):
            asyncio.run(hooks._handle(plugin, _event(), req))

        assert getattr(req, hooks._DONE_ATTR, False) is False

        # 下一次恢复正常调用时，由于未误设 _DONE_ATTR，重试可正常完成并打标
        async def ok_analyze(prepared: list[Any], *_a: Any, **_kw: Any) -> list[Any]:
            return [
                openai_vision.VisionResult(
                    ok=True, name=item.name, analysis="恢复后的视频理解"
                )
                for item in prepared
            ]

        monkeypatch.setattr(hooks, "_analyze_prepared_videos", ok_analyze)
        asyncio.run(hooks._handle(plugin, _event(), req))
        assert getattr(req, hooks._DONE_ATTR, False) is True
        assert any("恢复后的视频理解" in getattr(p, "text", "") for p in req.extra_user_content_parts)
