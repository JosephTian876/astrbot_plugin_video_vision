"""视觉服务层：两种协议、鉴权、错误分支、空 choices、超时、响应截断。

对应修复项：B10（模板缺 {question}）、B12（响应截断）、B17（choices: []）。
全部 HTTP 都打本机 ``http.server`` 假端点，**不联网**。
"""

from __future__ import annotations

import asyncio
import json
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from types import SimpleNamespace

import pytest

from vv_plugin.core import openai_vision
from vv_plugin.core.config import VideoConfig
from vv_plugin.core.openai_vision import (
    analyze_video,
    analyze_videos,
    build_vision_prompt,
)

# 0.2.1 新增常量：用模块属性取，保证本文件在 0.2.0 上也能被收集
MAX_ANALYSIS_CHARS = getattr(openai_vision, "MAX_ANALYSIS_CHARS", 16 * 1024)

DATA_URI = "data:video/mp4;base64,AAAA"
CHAT_PATH = "/chat/completions"


class _State:
    """假端点共享状态（每个测试前清空）。"""

    def __init__(self) -> None:
        self.routes: dict[str, object] = {}
        self.requests: list[dict] = []


STATE = _State()


class _Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def do_POST(self) -> None:  # noqa: N802 - BaseHTTPRequestHandler 约定
        length = int(self.headers.get("Content-Length") or 0)
        raw = self.rfile.read(length)
        try:
            parsed = json.loads(raw.decode("utf-8")) if raw else {}
        except Exception:  # noqa: BLE001
            parsed = {"_raw": raw.decode("utf-8", "replace")}
        STATE.requests.append(
            {
                "path": self.path,
                "headers": dict(self.headers),
                "json": parsed,
                "raw": raw,
            }
        )
        route = STATE.routes.get(self.path)
        if route is None:
            self._send(404, b"no route", "text/plain")
            return
        if route == "drip":
            self._drip()
            return
        status, payload, content_type = route  # type: ignore[misc]
        data = payload if isinstance(payload, bytes) else str(payload).encode("utf-8")
        self._send(status, data, content_type)

    def _send(self, status: int, data: bytes, content_type: str) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Connection", "close")
        self.end_headers()
        try:
            self.wfile.write(data)
        except OSError:  # 客户端已取消
            pass

    def _drip(self) -> None:
        """每 0.5s 吐 1 字节：httpx 的分阶段超时永远碰不到，只能靠整体上限。"""
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", "64")
        self.end_headers()
        try:
            for _ in range(64):
                self.wfile.write(b" ")
                self.wfile.flush()
                time.sleep(0.5)
        except OSError:
            pass

    def log_message(self, *args, **kwargs) -> None:  # 静音
        pass


@pytest.fixture(scope="session")
def _server_base() -> str:
    server = ThreadingHTTPServer(("127.0.0.1", 0), _Handler)
    server.daemon_threads = True
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_address[1]}"
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=10)


@pytest.fixture
def vision(_server_base):
    STATE.routes.clear()
    STATE.requests.clear()
    yield SimpleNamespace(
        base=_server_base, routes=STATE.routes, requests=STATE.requests
    )
    STATE.routes.clear()
    STATE.requests.clear()


def make_config(base_url: str, **kwargs) -> VideoConfig:
    options = {
        "openai_base_url": base_url,
        "openai_api_key": "sk-unit-test",
        "openai_model": "vision-model",
        "openai_timeout_sec": 10.0,
    }
    options.update(kwargs)
    return VideoConfig(**options)


def chat_body(text: str = "画面里有一只猫", **usage) -> str:
    return json.dumps(
        {
            "choices": [{"message": {"role": "assistant", "content": text}}],
            "usage": {
                "prompt_tokens": usage.get("prompt_tokens", 262),
                "completion_tokens": usage.get("completion_tokens", 17),
            },
        }
    )


def run(coro):
    return asyncio.run(coro)


# --------------------------------------------------------------- 两种协议


def test_chat_completions_protocol(vision):
    vision.routes[CHAT_PATH] = (200, chat_body(), "application/json")
    config = make_config(vision.base + CHAT_PATH)

    result = run(analyze_video(DATA_URI, "a.mp4", "描述一下", config=config))

    assert result.ok is True
    assert result.analysis == "画面里有一只猫"
    assert result.prompt_tokens == 262
    assert result.completion_tokens == 17
    request = vision.requests[0]
    assert request["path"] == CHAT_PATH
    assert request["headers"]["Authorization"] == "Bearer sk-unit-test"
    assert request["headers"]["Content-Type"] == "application/json"
    content = request["json"]["messages"][0]["content"]
    assert content[0] == {"type": "text", "text": "描述一下"}
    assert content[1]["type"] == "image_url"
    assert content[1]["image_url"]["url"] == DATA_URI


def test_responses_protocol(vision):
    vision.routes["/responses"] = (
        200,
        json.dumps(
            {
                "output_text": "这是一段屏幕录制",
                "usage": {"input_tokens": 402, "output_tokens": 21},
            }
        ),
        "application/json",
    )
    config = make_config(vision.base + "/responses")

    result = run(analyze_video(DATA_URI, "b.mp4", "讲了什么", config=config))

    assert result.ok is True
    assert result.analysis == "这是一段屏幕录制"
    assert result.prompt_tokens == 402
    request = vision.requests[0]
    assert request["path"] == "/responses"
    content = request["json"]["input"][0]["content"]
    assert content[0]["type"] == "input_text"
    assert content[1] == {"type": "input_image", "image_url": DATA_URI}


def test_responses_protocol_nested_output(vision):
    vision.routes["/responses"] = (
        200,
        json.dumps(
            {
                "output": [
                    {
                        "content": [
                            {"type": "output_text", "text": "第一段"},
                            {"type": "output_text", "text": "第二段"},
                        ]
                    }
                ]
            }
        ),
        "application/json",
    )
    result = run(
        analyze_video(
            DATA_URI, "c.mp4", "x", config=make_config(vision.base + "/responses")
        )
    )
    assert result.ok is True
    assert result.analysis == "第一段\n第二段"


@pytest.mark.parametrize(
    ("base_suffix", "expected_path", "body", "ok"),
    [
        ("/v1", "/v1/chat/completions", chat_body("ok"), True),
        ("/v1/", "/v1/chat/completions", chat_body("ok"), True),
        ("/v1/chat/completions", "/v1/chat/completions", chat_body("ok"), True),
        ("/proxy/responses", "/proxy/responses", '{"output_text": "ok"}', True),
    ],
)
def test_base_url_variants(vision, base_suffix, expected_path, body, ok):
    vision.routes[expected_path] = (200, body, "application/json")
    config = make_config(vision.base + base_suffix)

    result = run(analyze_video(DATA_URI, "d.mp4", "x", config=config))

    assert result.ok is ok
    assert vision.requests[0]["path"] == expected_path


# --------------------------------------------------------------- 错误分支


def test_http_error_is_reported(vision):
    vision.routes[CHAT_PATH] = (500, "upstream exploded", "text/plain")
    result = run(
        analyze_video(
            DATA_URI, "e.mp4", "x", config=make_config(vision.base + CHAT_PATH)
        )
    )
    assert result.ok is False
    assert "HTTP 500" in result.reason
    assert "upstream exploded" in result.reason


def test_non_json_body_is_reported_as_parse_failure(vision):
    vision.routes[CHAT_PATH] = (200, "<html>not json</html>", "text/html")
    result = run(
        analyze_video(
            DATA_URI, "f.mp4", "x", config=make_config(vision.base + CHAT_PATH)
        )
    )
    assert result.ok is False
    assert "解析失败" in result.reason


def test_empty_choices_reports_empty_content(vision):
    """回归：0.2.0 走 IndexError -> "响应解析失败"（把服务端空回答伪装成 bug）。"""
    vision.routes[CHAT_PATH] = (200, json.dumps({"choices": []}), "application/json")
    result = run(
        analyze_video(
            DATA_URI, "g.mp4", "x", config=make_config(vision.base + CHAT_PATH)
        )
    )
    assert result.ok is False
    assert result.reason == "视觉服务返回了空内容"


def test_missing_choices_field_reports_empty_content(vision):
    vision.routes[CHAT_PATH] = (200, "{}", "application/json")
    result = run(
        analyze_video(
            DATA_URI, "h.mp4", "x", config=make_config(vision.base + CHAT_PATH)
        )
    )
    assert result.ok is False
    assert result.reason == "视觉服务返回了空内容"


def test_drip_endpoint_hits_the_overall_timeout(vision):
    # base 不以 /chat/completions 结尾 -> 实际请求 /drip/chat/completions
    vision.routes["/drip/chat/completions"] = "drip"
    config = make_config(vision.base + "/drip", openai_timeout_sec=1.0)

    started = time.monotonic()
    result = run(analyze_video(DATA_URI, "i.mp4", "x", config=config))
    elapsed = time.monotonic() - started

    assert result.ok is False
    assert "超时" in result.reason
    assert elapsed < 6.0


def test_connection_refused_is_classified():
    """连一个必然关闭的端口：只验证不会抛异常、reason 可读。"""
    config = make_config("http://127.0.0.1:1/v1", openai_timeout_sec=2.0)
    result = run(analyze_video(DATA_URI, "j.mp4", "x", config=config))
    assert result.ok is False
    assert result.reason.startswith("视觉服务")


# --------------------------------------------------------------- 响应截断


def test_huge_response_is_truncated(vision, stub_logger):
    """回归：0.2.0 会把 1MB 正文原样拼进 prompt。"""
    vision.routes[CHAT_PATH] = (
        200,
        chat_body("X" * (MAX_ANALYSIS_CHARS * 4)),
        "application/json",
    )
    result = run(
        analyze_video(
            DATA_URI,
            "k.mp4",
            "x",
            config=make_config(vision.base + CHAT_PATH),
            logger=stub_logger,
        )
    )

    assert result.ok is True
    assert len(result.analysis) <= MAX_ANALYSIS_CHARS + 32
    assert result.analysis.startswith("X")
    assert any("过长" in item for item in stub_logger.messages("warning"))


def test_short_response_is_not_truncated(vision, stub_logger):
    vision.routes[CHAT_PATH] = (200, chat_body("短"), "application/json")
    result = run(
        analyze_video(
            DATA_URI,
            "l.mp4",
            "x",
            config=make_config(vision.base + CHAT_PATH),
            logger=stub_logger,
        )
    )
    assert result.analysis == "短"
    assert stub_logger.messages("warning") == []


def test_analyze_videos_keeps_order_and_isolates_failures(vision):
    vision.routes[CHAT_PATH] = (200, chat_body("ok"), "application/json")
    config = make_config(vision.base + CHAT_PATH)

    results = run(
        analyze_videos([("[a]", DATA_URI), ("[b]", DATA_URI)], "", config=config)
    )

    assert [item.name for item in results] == ["[a]", "[b]"]
    assert all(item.ok for item in results)
    assert vision.requests[0]["json"]["messages"][0]["content"][0]["type"] == "text"


# --------------------------------------------------------------- 提示词模板


def test_qa_template_with_placeholder_is_used():
    config = VideoConfig(vision_qa_prompt="问题：{question}")
    assert build_vision_prompt(config, "讲了什么") == "问题：讲了什么"


def test_qa_template_without_placeholder_keeps_the_question(stub_logger):
    """回归：0.2.0 会静默丢弃用户问题，只把模板发出去。"""
    openai_vision._warned_templates.clear()
    config = VideoConfig(vision_qa_prompt="没有占位符的模板")

    prompt = build_vision_prompt(config, "讲了什么", logger=stub_logger)

    assert "没有占位符的模板" in prompt
    assert "讲了什么" in prompt
    assert any("没有 {question}" in item for item in stub_logger.messages("warning"))

    # 只告警一次
    build_vision_prompt(config, "又问一次", logger=stub_logger)
    warnings = [
        item for item in stub_logger.messages("warning") if "没有 {question}" in item
    ]
    assert len(warnings) == 1


def test_no_question_falls_back_to_describe_prompt():
    config = VideoConfig()
    assert build_vision_prompt(config, "") == config.vision_describe_prompt
    assert build_vision_prompt(config, "   ") == config.vision_describe_prompt


def test_vision_with_question_disabled_uses_describe_prompt():
    config = VideoConfig(vision_with_question=False, vision_qa_prompt="Q:{question}")
    assert build_vision_prompt(config, "问题") == config.vision_describe_prompt
