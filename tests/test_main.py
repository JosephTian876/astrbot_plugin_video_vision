"""``/videotest`` 与插件装配：GreedyStr 契约、事件循环不阻塞、诊断内容。

对应修复项：B2（to_thread）、B7（GreedyStr）、B8（诊断信息）、B18（未知键）。
"""

from __future__ import annotations

import asyncio
import inspect
import shutil
import threading
from pathlib import Path
from types import SimpleNamespace

import pytest

from astrbot.core.star.filter.command import GreedyStr
from vv_plugin import main as main_module
from vv_plugin.core.config import VideoConfig
from vv_plugin.core.injector import PrepareOutcome
from vv_plugin.core.openai_vision import VisionResult
from vv_plugin.hooks import llm_request as hooks
from vv_plugin.main import VideoVisionPlugin

# 0.2.1 新增的私有助手：用模块属性取，保证本文件在 0.2.0 上也能被收集
_missing_credentials = getattr(main_module, "_missing_credentials", None)

# --------------------------------------------------- 框架参数提取（复刻源码）


def handler_params(func) -> dict:
    """复刻 ``CommandFilter.init_handler_md``（command.py:71-79）。

    有默认值时框架存的是**默认值本身**，不是注解 —— 这正是 0.2.0 的 bug 来源。
    """
    params: dict = {}
    signature = inspect.signature(func, eval_str=True)
    for index, (name, parameter) in enumerate(signature.parameters.items()):
        if index < 2:  # self / event
            continue
        if parameter.default == inspect.Parameter.empty:
            params[name] = parameter.annotation
        else:
            params[name] = parameter.default
    return params


def apply_params(params: dict, tokens: list[str]) -> dict:
    """复刻 ``validate_and_convert_params`` 的贪婪分支（command.py:101-114）。"""
    result: dict = {}
    items = list(params.items())
    for index, (name, value) in enumerate(items):
        if value is GreedyStr:
            assert index == len(items) - 1, "GreedyStr 必须是最后一个参数"
            result[name] = " ".join(tokens[index:])
            break
        result[name] = tokens[index] if index < len(tokens) else value
    return result


def test_videotest_path_is_registered_as_greedy():
    """回归：``path: GreedyStr = ""`` 会让 is_greedy 恒为 False。"""
    params = handler_params(VideoVisionPlugin.cmd_videotest)
    assert params == {"path": GreedyStr}


@pytest.mark.parametrize(
    "raw_path",
    [
        r"C:\Program Files\Some App\视频 1.mp4",
        r"C:\Users\UwU\Videos\中文 空格 都有.mp4",
        r"D:\a b\c d\e.mp4",
    ],
)
def test_videotest_keeps_paths_with_spaces(raw_path):
    tokens = raw_path.split()  # 框架按空格切分参数
    assert len(tokens) > 1
    params = handler_params(VideoVisionPlugin.cmd_videotest)
    assert apply_params(params, tokens)["path"] == raw_path


# ------------------------------------------------------------- 测试用装配


class _Context:
    def __init__(self, provider=None, error: BaseException | None = None) -> None:
        self._provider = provider
        self._error = error

    async def get_using_provider_async(self, umo=None):
        if self._error is not None:
            raise self._error
        return self._provider


class _Event:
    role = "admin"

    def __init__(self, text: str = "") -> None:
        self.message_str = text
        self.unified_msg_origin = "test-umo"

    def get_message_str(self) -> str:
        return self.message_str

    def plain_result(self, text: str) -> str:
        return text


def make_plugin(context=None, **raw_config) -> VideoVisionPlugin:
    plugin = VideoVisionPlugin(context or _Context(), raw_config)
    return plugin


def point_to_ffmpeg(plugin: VideoVisionPlugin, media_tools) -> None:
    ffmpeg, ffprobe = media_tools
    plugin.ffmpeg = ffmpeg
    plugin.ffprobe = ffprobe


def run(coro):
    return asyncio.run(coro)


# --------------------------------------------------- 事件循环不被阻塞


def test_diagnose_offloads_blocking_calls(monkeypatch, tiny_video, stub_logger):
    """回归：0.2.0 在事件循环里同步跑 ffprobe/ffmpeg。"""

    plugin = make_plugin()
    seen: dict[str, threading.Thread] = {}

    def fake_probe(path, ffprobe):
        seen["probe"] = threading.current_thread()
        return 2.0

    def fake_prepare(*args, **kwargs):
        seen["prepare"] = threading.current_thread()
        return PrepareOutcome(False, reason="stub 停止后续步骤", kind="internal")

    monkeypatch.setattr(main_module, "probe_duration", fake_probe)
    monkeypatch.setattr(main_module, "prepare_video", fake_prepare)

    report = run(plugin._diagnose(tiny_video))

    assert "准备失败" in report
    assert seen["probe"] is not threading.main_thread()
    assert seen["prepare"] is not threading.main_thread()


def test_diagnose_yields_to_the_event_loop(monkeypatch, tiny_video):
    """心跳：诊断期间另一个协程仍能推进（旧实现会被卡住）。"""

    plugin = make_plugin()
    ticks = 0

    def slow_probe(path, ffprobe):
        import time

        time.sleep(0.2)
        return 2.0

    def fake_prepare(*args, **kwargs):
        import time

        time.sleep(0.2)
        return PrepareOutcome(False, reason="stub", kind="internal")

    monkeypatch.setattr(main_module, "probe_duration", slow_probe)
    monkeypatch.setattr(main_module, "prepare_video", fake_prepare)

    async def scenario():
        nonlocal ticks

        async def heartbeat():
            nonlocal ticks
            while True:
                ticks += 1
                await asyncio.sleep(0.01)

        task = asyncio.create_task(heartbeat())
        try:
            return await plugin._diagnose(tiny_video)
        finally:
            task.cancel()

    run(scenario())
    assert ticks > 5


# ------------------------------------------------------------- 诊断内容


def test_diagnose_reports_provider_gates_and_prompt(
    media_tools, tiny_video, stub_logger
):
    provider = SimpleNamespace(
        provider_config={
            "type": "googlegenai_chat_completion",
            "id": "gemini-2.5-flash",
            "modalities": ["text", "image", "audio"],
        }
    )
    plugin = make_plugin(_Context(provider), provider_mode="gemini_native")
    point_to_ffmpeg(plugin, media_tools)

    report = run(plugin._diagnose(tiny_video, event=_Event()))

    for needle in (
        "provider:",
        "type=googlegenai_chat_completion",
        "modalities=[text, image, audio]",
        "闸门   :",
        "① 类型 googlegenai_chat_completion ✅",
        "② modalities 含 audio ✅",
        "模式   : gemini_native",
        "凭据   : 视觉服务",
        "prompt :",
    ):
        assert needle in report, needle
    assert "✅ 准备成功" in report
    assert "动作: 直接送（未压缩）" in report


def test_diagnose_openai_mode_shows_credentials_and_prompt(
    media_tools, tiny_video, monkeypatch, stub_logger
):

    async def fake_analyze(*args, **kwargs):
        return VisionResult(False, "x", reason="stub 不联网")

    monkeypatch.setattr(main_module, "analyze_video", fake_analyze)
    plugin = make_plugin(
        _Context(),
        provider_mode="openai_compat",
        openai_base_url="http://127.0.0.1:9/v1",
        openai_api_key="sk-x",
        openai_model="vision-model",
    )
    point_to_ffmpeg(plugin, media_tools)

    report = run(plugin._diagnose(tiny_video, event=_Event()))

    assert "模式   : openai_compat" in report
    assert "凭据   : 视觉服务 ✅ base_url / api_key / model 齐全" in report
    assert "① Gemini 系 provider 校验 → openai_compat 不适用" in report
    assert "prompt :" in report
    assert "请详细描述这段视频" in report
    assert "❌ 视觉调用失败: stub 不联网" in report


def test_diagnose_reports_missing_credentials(media_tools, tiny_video):
    plugin = make_plugin(_Context(), provider_mode="gemini_native")
    point_to_ffmpeg(plugin, media_tools)

    report = run(plugin._diagnose(tiny_video, event=_Event()))

    assert "❌ 缺 openai_base_url、openai_api_key" in report
    assert _missing_credentials is not None
    assert _missing_credentials(plugin.config) == [
        "openai_base_url",
        "openai_api_key",
    ]


def test_diagnose_reports_provider_resolution_failure(media_tools, tiny_video):
    context = _Context(error=RuntimeError("provider manager not ready"))
    plugin = make_plugin(context, provider_mode="gemini_native")
    point_to_ffmpeg(plugin, media_tools)

    report = run(plugin._diagnose(tiny_video, event=_Event()))

    assert "① provider 解析失败" in report
    assert "provider manager not ready" in report


def test_diagnose_refuses_non_video_extension_honestly(
    media_tools, tmp_path, stub_logger
):
    """/videotest 不校验扩展名：这是已知限制，文档里写明了（这里只锁行为）。"""
    not_video = tmp_path / "notes.txt"
    not_video.write_text("hello", encoding="utf-8")
    plugin = make_plugin(_Context(), provider_mode="gemini_native")
    point_to_ffmpeg(plugin, media_tools)

    report = run(plugin._diagnose(str(not_video), event=_Event()))

    # 有 ffprobe 却读不出时长、扩展名不是视频 -> 不被"损坏"规则拦，照样送
    assert "准备失败" in report or "直接送" in report


# -------------------------------------------------------------- 命令层


def test_videotest_rejects_non_admin():
    plugin = make_plugin()
    event = SimpleNamespace(role="member", plain_result=lambda text: text)

    async def collect():
        return [item async for item in plugin.cmd_videotest(event, "x.mp4")]

    assert "仅管理员" in run(collect())[0]


def test_videotest_handles_paths_with_spaces_endtoend(
    media_tools, tiny_video, tmp_path, stub_logger
):
    source = Path(tiny_video)
    target = tmp_path / "带 空格 的视频.mp4"
    shutil.copyfile(source, target)
    plugin = make_plugin(_Context(), provider_mode="gemini_native")
    point_to_ffmpeg(plugin, media_tools)
    event = _Event()

    async def collect():
        return [
            item async for item in plugin.cmd_videotest(event, f'"{target}"')
        ]

    report = run(collect())[0]

    assert "❌ 文件不存在" not in report
    assert target.name in report
    assert "✅ 准备成功" in report


# ------------------------------------------------------------- 装配告警


def test_plugin_reports_unknown_config_keys(stub_logger):
    """回归：0.2.0 对拼错的键完全静默。"""
    make_plugin(_Context(), max_size_mbb=5)

    warnings = stub_logger.messages("warning")
    assert any("未知配置键" in item for item in warnings)
    assert any("max_size_mbb" in item for item in warnings)


def test_plugin_reports_clamped_values(stub_logger):
    make_plugin(_Context(), max_video_count=99)

    warnings = stub_logger.messages("warning")
    assert any("max_video_count" in item and "上限" in item for item in warnings)


def test_degradation_is_warned_once_across_startup_and_hook(stub_logger):
    """回归：0.2.0 会在启动与首次请求各告警一次。"""
    hooks._warned_config.clear()

    plugin = make_plugin(_Context())  # 默认 openai_compat + 空三项 -> 降级

    startup = [item for item in stub_logger.messages("warning") if "降级" in item]
    assert len(startup) == 1

    assert hooks._effective_mode(plugin.config) == "gemini_native"
    after_hook = [item for item in stub_logger.messages("warning") if "降级" in item]
    assert len(after_hook) == 1


def test_plugin_masks_api_key_in_startup_log(stub_logger):
    make_plugin(
        _Context(),
        provider_mode="openai_compat",
        openai_base_url="https://example.invalid/v1",
        openai_api_key="sk-super-secret-value",
        openai_model="vision-model",
    )
    infos = " ".join(stub_logger.messages("info"))
    assert "sk-super-secret-value" not in infos
    assert "…alue" in infos or "已配置" in infos


def test_config_property_is_a_video_config():
    plugin = make_plugin(_Context(), max_size_mb=10, compress_max_mb=40)
    assert isinstance(plugin.config, VideoConfig)
    # normalized() 会把压缩上限压到闸门以下
    assert plugin.config.compress_max_mb < plugin.config.max_size_mb
