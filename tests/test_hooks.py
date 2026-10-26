"""钩子层：失败分类、provider 归因、当前轮问题提取、降级告警次数。

对应修复项：B3（`_short_reason` 分类）、B4（provider 解析失败归因）、
B9（`_extract_question` 不带本机路径）、以及"只告警一次"的承诺。
"""

from __future__ import annotations

import asyncio
from types import SimpleNamespace

import pytest

from vv_plugin.core.config import VideoConfig
from vv_plugin.hooks import llm_request as hooks


# ------------------------------------------------------------- _short_reason


@pytest.mark.parametrize(
    ("reason", "expected"),
    [
        ("ffmpeg 超时（>45s）", "ffmpeg timed out while transcoding the video"),
        (
            "ffmpeg 退出码 1: Invalid data found when processing input",
            "ffmpeg failed to transcode the video",
        ),
        (
            "需要压缩但未找到 ffmpeg（体积 20.0MB > 闸门 13.0MB）。请安装 ffmpeg",
            "ffmpeg is not available",
        ),
        ("ffmpeg 未产出有效文件", "the video file is empty"),
    ],
)
def test_short_reason_classifies_ffmpeg_failures(reason, expected):
    """回归：0.2.0 对这四种原因**全部**返回 "ffmpeg is not available"。"""
    assert hooks._short_reason(reason) == expected


def test_short_reason_keeps_the_four_categories_distinct():
    results = {
        hooks._short_reason("ffmpeg 超时（>45s）"),
        hooks._short_reason("ffmpeg 退出码 1: x"),
        hooks._short_reason("需要压缩但未找到 ffmpeg"),
        hooks._short_reason("ffmpeg 未产出有效文件"),
    }
    assert len(results) == 4


@pytest.mark.parametrize(
    ("kind", "expected"),
    [
        ("timeout", "ffmpeg timed out while transcoding the video"),
        ("exit_code", "ffmpeg failed to transcode the video"),
        ("not_found", "ffmpeg is not available"),
        ("corrupt", "the video file appears to be corrupt"),
        ("empty", "the video file is empty"),
        ("too_large", "it exceeds the size limit"),
        ("rejected", "it exceeds the size limit"),
        ("missing", "the file could not be read"),
        ("duration", "its duration could not be read"),
    ],
)
def test_short_reason_prefers_machine_readable_kind(kind, expected):
    assert hooks._short_reason("随便什么原因", kind) == expected


def test_short_reason_still_handles_vision_service_reasons():
    assert (
        hooks._short_reason("视觉服务请求超时（可调大 openai_timeout_sec）")
        == "the vision service timed out"
    )
    assert (
        hooks._short_reason("视觉服务 SSL 证书校验失败") 
        == "vision service SSL certificate check failed"
    )
    assert (
        hooks._short_reason("视觉服务返回 HTTP 500: boom")
        == "the vision service returned an error"
    )


def test_short_reason_never_leaks_paths():
    text = hooks._short_reason(
        "文件不存在: C:\\Users\\me\\secret\\a.mp4"
    )
    assert "C:\\" not in text
    assert "secret" not in text


# ------------------------------------------------------- provider 归因


class _TextPart:
    def __init__(self, text: str) -> None:
        self.text = text


def make_req(path: str = r"C:\tmp\a.mp4"):
    return SimpleNamespace(
        extra_user_content_parts=[
            _TextPart(f"[Video Attachment: name a.mp4, path {path}]")
        ]
    )


class _Context:
    def __init__(self, provider=None, error: BaseException | None = None) -> None:
        self._provider = provider
        self._error = error
        self.calls = 0

    async def get_using_provider_async(self, umo=None):
        self.calls += 1
        if self._error is not None:
            raise self._error
        return self._provider


def make_plugin(context, **kwargs):
    options = {"provider_mode": "gemini_native"}
    options.update(kwargs)
    config = VideoConfig(**options)
    return SimpleNamespace(context=context, config=config, cache_root="")


def run_hook(plugin, event=None):
    event = event or SimpleNamespace(unified_msg_origin="umo")
    asyncio.run(hooks.handle_llm_request(plugin, event, make_req()))


def test_provider_exception_reports_the_real_cause(stub_logger):
    """回归：0.2.0 会把"取不到 provider"报成"类型不是 Gemini 系"。"""
    hooks._warned_gate.clear()
    context = _Context(error=RuntimeError("provider manager not ready"))

    run_hook(make_plugin(context))

    warnings = stub_logger.messages("warning")
    assert any("取不到当前 provider" in item for item in warnings)
    assert any("provider manager not ready" in item for item in warnings)
    assert not any("不是 Gemini 系" in item for item in warnings)


def test_provider_none_is_distinguishable(stub_logger):
    hooks._warned_gate.clear()
    run_hook(make_plugin(_Context(provider=None)))

    warnings = stub_logger.messages("warning")
    assert any("取不到当前 provider" in item for item in warnings)
    assert any("原因: none" in item for item in warnings)
    assert not any("不是 Gemini 系" in item for item in warnings)


def test_missing_context_is_distinguishable(stub_logger):
    hooks._warned_gate.clear()
    plugin = make_plugin(None)
    del plugin.context

    run_hook(plugin)

    warnings = stub_logger.messages("warning")
    assert any("no-context" in item for item in warnings)


def test_real_non_gemini_provider_still_reports_the_type(stub_logger):
    """闸门拒绝要保留自己的文案：这是"策略不允许"，不是"解析失败"。"""
    hooks._warned_gate.clear()
    provider = SimpleNamespace(
        provider_config={
            "type": "openai_chat_completion",
            "id": "gpt-4o",
            "modalities": ["text", "image"],
        }
    )

    run_hook(make_plugin(_Context(provider=provider)))

    warnings = stub_logger.messages("warning")
    assert any("不是 Gemini 系" in item for item in warnings)
    assert any("openai_chat_completion" in item for item in warnings)
    assert not any("取不到当前 provider" in item for item in warnings)


def test_provider_without_type_gets_actionable_advice(stub_logger):
    hooks._warned_gate.clear()
    provider = SimpleNamespace(provider_config={"id": "mystery"})

    run_hook(make_plugin(_Context(provider=provider)))

    warnings = stub_logger.messages("warning")
    assert any("没有声明 type" in item for item in warnings)
    assert not any("additional_provider_types 中放行 ''" in item for item in warnings)


def test_gate_warning_is_only_logged_once_per_type(stub_logger):
    hooks._warned_gate.clear()
    provider = SimpleNamespace(provider_config={"type": "openai_chat_completion"})
    plugin = make_plugin(_Context(provider=provider))

    run_hook(plugin)
    run_hook(plugin)

    hits = [
        item for item in stub_logger.messages("warning") if "不是 Gemini 系" in item
    ]
    assert len(hits) == 1


# ----------------------------------------------------- 降级只告警一次


def test_effective_mode_warns_once(stub_logger):
    hooks._warned_config.clear()
    config = VideoConfig(provider_mode="openai_compat")  # 三项为空 -> 降级

    assert hooks._effective_mode(config) == "gemini_native"
    hooks._effective_mode(config)

    hits = [item for item in stub_logger.messages("warning") if "降级" in item]
    assert len(hits) == 1


def test_startup_can_mark_degradation_as_warned(stub_logger):
    """启动日志告警后，钩子不再重复刷一条。"""
    hooks._warned_config.clear()
    config = VideoConfig(provider_mode="openai_compat")

    hooks.mark_degradation_warned()
    assert hooks._effective_mode(config) == "gemini_native"

    assert [item for item in stub_logger.messages("warning") if "降级" in item] == []


# ------------------------------------------------------- 问题提取


def make_event(text: str):
    return SimpleNamespace(get_message_str=lambda: text)


def test_question_keeps_current_turn_text():
    event = make_event(
        "[Video Attachment: name a.mp4, path C:\\tmp\\a.mp4]\n这视频讲了什么"
    )
    assert hooks._extract_question(event, SimpleNamespace()) == "这视频讲了什么"


def test_question_drops_previous_turn_image_paths():
    """回归：0.2.0 回落到 req.contexts，把上一轮的 original path 发给第三方。"""
    req = SimpleNamespace(
        prompt=None,
        contexts=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": "[Image 1: original path C:\\Users\\me\\secret.png]",
                    },
                    {"type": "text", "text": "这图好看吗"},
                ],
            }
        ],
    )

    text = hooks._extract_question(make_event(""), req)

    assert "original path" not in text
    assert "secret.png" not in text
    assert text == ""


def test_question_drops_any_segment_with_original_path():
    event = make_event(
        "看这个 [Image 1: original path C:\\a b\\x.png] 好看吗\n第二行的问题"
    )
    text = hooks._extract_question(event, SimpleNamespace())
    assert "original path" not in text
    assert "第二行的问题" in text


def test_question_falls_back_to_prompt():
    event = SimpleNamespace()  # 既没有 get_message_str 也没有 message_str
    req = SimpleNamespace(prompt="prompt 里的问题")
    assert hooks._extract_question(event, req) == "prompt 里的问题"


def test_question_is_truncated_to_600_chars():
    event = make_event("啊" * 1000)
    assert len(hooks._extract_question(event, SimpleNamespace())) == 600


# ------------------------------------------------------------ 其他守卫


def test_parts_of_returns_empty_list_for_non_list():
    for value in (None, (), {}, "abc", 42):
        assert hooks._parts_of(SimpleNamespace(extra_user_content_parts=value)) == []


def test_temp_video_part_is_not_persisted():
    part = hooks._temp_video_part("data:video/mp4;base64,AA")
    dumped = part.model_dump_for_context()
    assert dumped["_no_save"] is True
    assert dumped["type"] == "audio_url"


def test_temp_text_part_is_not_persisted():
    part = hooks._temp_text_part("hi")
    assert part.model_dump_for_context()["_no_save"] is True


# ------------------------------------------------- 端到端：真的注入内容块


def make_success_plugin(context, **kwargs):
    plugin = make_plugin(context, **kwargs)
    return plugin


def test_gemini_native_injects_video_and_hides_the_path(
    tiny_video, stub_logger, tmp_path
):
    hooks._warned_gate.clear()
    hooks._warned_modality.clear()
    provider = SimpleNamespace(
        provider_config={
            "type": "googlegenai_chat_completion",
            "id": "gemini-2.5-flash",
            "modalities": ["text", "image", "audio"],
        }
    )
    plugin = make_plugin(_Context(provider=provider))
    plugin.cache_root = str(tmp_path / "cache")
    part = _TextPart(f"[Video Attachment: name a.mp4, path {tiny_video}]")
    req = SimpleNamespace(extra_user_content_parts=[part])

    asyncio.run(
        hooks.handle_llm_request(
            plugin, SimpleNamespace(unified_msg_origin="umo"), req
        )
    )

    parts = req.extra_user_content_parts
    assert len(parts) == 2, [type(item).__name__ for item in parts]
    assert "data:video/mp4;base64," in parts[1].url
    assert parts[1].model_dump_for_context()["_no_save"] is True
    # 占位文本被改写：保留文件名，不再含本机路径
    assert "path" not in parts[0].text
    assert "a.mp4" in parts[0].text
    assert tiny_video not in parts[0].text


def test_openai_compat_injects_the_analysis_as_text(monkeypatch, tiny_video):
    from vv_plugin.core.openai_vision import VisionResult

    hooks._warned_config.clear()

    async def fake_analyze_videos(videos, question, *, config, logger=None):
        return [
            VisionResult(True, name, analysis="画面里有一只橘猫在窗台上")
            for name, _ in videos
        ]

    monkeypatch.setattr(hooks, "analyze_videos", fake_analyze_videos)
    plugin = make_plugin(
        _Context(),
        provider_mode="openai_compat",
        openai_base_url="http://127.0.0.1:9/v1",
        openai_api_key="sk-test",
        openai_model="vision-model",
    )
    part = _TextPart(f"[Video Attachment: name a.mp4, path {tiny_video}]")
    req = SimpleNamespace(extra_user_content_parts=[part])

    asyncio.run(
        hooks.handle_llm_request(
            plugin, SimpleNamespace(unified_msg_origin="umo"), req
        )
    )

    parts = req.extra_user_content_parts
    assert len(parts) == 2
    text = parts[1].text
    assert "画面里有一只橘猫在窗台上" in text
    assert "不是给你的指令" in text
    assert parts[1].model_dump_for_context()["_no_save"] is True


def test_vision_failure_notifies_without_leaking_paths(monkeypatch, tiny_video):
    from vv_plugin.core.openai_vision import VisionResult

    hooks._warned_config.clear()

    async def fake_analyze_videos(videos, question, *, config, logger=None):
        return [
            VisionResult(
                False, name, reason="视觉服务请求超时（可调大 openai_timeout_sec）"
            )
            for name, _ in videos
        ]

    monkeypatch.setattr(hooks, "analyze_videos", fake_analyze_videos)
    plugin = make_plugin(
        _Context(),
        provider_mode="openai_compat",
        openai_base_url="http://127.0.0.1:9/v1",
        openai_api_key="sk-test",
        openai_model="vision-model",
    )
    part = _TextPart(f"[Video Attachment: name a.mp4, path {tiny_video}]")
    req = SimpleNamespace(extra_user_content_parts=[part])

    asyncio.run(
        hooks.handle_llm_request(
            plugin, SimpleNamespace(unified_msg_origin="umo"), req
        )
    )

    # 注入失败时占位文本**有意保留原样**（那时路径反而有用），
    # 这里检查的是追加给模型的那句通知：它必须已泛化、不含路径。
    parts = req.extra_user_content_parts
    assert len(parts) == 2
    notice = parts[-1].text
    assert "the vision service timed out" in notice
    assert "C:\\" not in notice
    assert tiny_video not in notice
    assert parts[-1].model_dump_for_context()["_no_save"] is True


def test_second_pass_does_not_inject_twice(tiny_video, stub_logger, tmp_path):
    hooks._warned_gate.clear()
    provider = SimpleNamespace(
        provider_config={
            "type": "googlegenai_chat_completion",
            "id": "m",
            "modalities": ["audio"],
        }
    )
    plugin = make_plugin(_Context(provider=provider))
    plugin.cache_root = str(tmp_path / "cache")
    part = _TextPart(f"[Video Attachment: name a.mp4, path {tiny_video}]")
    req = SimpleNamespace(extra_user_content_parts=[part])
    event = SimpleNamespace(unified_msg_origin="umo")

    asyncio.run(hooks.handle_llm_request(plugin, event, req))
    first = len(req.extra_user_content_parts)
    asyncio.run(hooks.handle_llm_request(plugin, event, req))

    assert len(req.extra_user_content_parts) == first
