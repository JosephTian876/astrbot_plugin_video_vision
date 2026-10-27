"""``/videotest`` 与插件装配：GreedyStr 契约、事件循环不阻塞、诊断内容。

对应修复项：B2（to_thread）、B7（GreedyStr）、B8（诊断信息）、B18（未知键）。
"""

from __future__ import annotations

import asyncio
import inspect
import shutil
import subprocess
import threading
from pathlib import Path
from types import SimpleNamespace

import pytest

from astrbot.core.star.filter.command import GreedyStr
from vv_plugin import main as main_module
from vv_plugin.core import diagnostics
from vv_plugin.core.config import VideoConfig
from vv_plugin.core.injector import PrepareOutcome
from vv_plugin.core.openai_vision import VisionResult
from vv_plugin.hooks import llm_request as hooks
from vv_plugin.main import VideoVisionPlugin

# 凭据齐全性诊断在 ``core/diagnostics.py``（本轮从 ``main`` 搬出）
_missing_credentials = diagnostics.missing_credentials

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
        r"C:\videos\中文 空格 都有.mp4",
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
    # 但必须提前说明"后缀未知、MIME 是猜的"，不能再装作一切正常
    assert "不在已知视频容器内" in report
    assert "(推断)" in report


def test_diagnose_corrupt_judgement_matches_the_outcome(
    media_tools, ffmpeg_dir, tmp_path, stub_logger
):
    """回归：结论行写"未超限，直接送"，紧接着"❌ 准备失败 分类: corrupt"。

    真机证据见 LIVE-TEST §10.2②（broken.mp4：2MB 随机字节）。
    """
    broken = tmp_path / "broken.mp4"
    broken.write_bytes(b"\x00\x01\x02not a video" * 500)
    plugin = make_plugin(
        _Context(), ffmpeg_path=ffmpeg_dir, provider_mode="gemini_native"
    )
    plugin.cache_root = str(tmp_path / "cache")

    report = run(plugin._diagnose(str(broken), event=_Event()))

    assert "结论   : 无法解析（时长读取失败）→ 会以 corrupt 失败" in report
    assert "未超限，直接送" not in report
    # 结论行必须和随后的真实结果一致
    assert "分类: corrupt" in report


def test_diagnose_corrupt_judgement_stays_silent_without_ffprobe(
    tmp_path, stub_logger
):
    """没有 ffprobe 时不误伤：injector 不判 corrupt，结论行也不能提前说 corrupt。"""
    broken = tmp_path / "broken.mp4"
    broken.write_bytes(b"\x00\x01\x02not a video" * 500)
    plugin = make_plugin(
        _Context(),
        ffmpeg_path=str(tmp_path / "missing-dir"),
        provider_mode="gemini_native",
    )
    plugin.cache_root = str(tmp_path / "cache")
    plugin.ffprobe = None

    report = run(plugin._diagnose(str(broken), event=_Event()))

    assert "corrupt" not in report
    assert "未超限，直接送" in report


def test_diagnose_flags_the_guessed_mime_for_unknown_suffixes(
    media_tools, tmp_path, stub_logger
):
    """回归：`/videotest anim.gif` 说"✅ 准备成功 / video/mp4"，真发出去是假 MIME。

    ``guess_video_mime`` 对未知后缀回落 ``OUTPUT_MIME``，而显式路径跳过闸门 ⓪
    （真机证据见 LIVE-TEST §10.2③）。这里只要求诊断把话说清楚，行为不变。
    """
    ffmpeg = media_tools[0]
    gif = tmp_path / "anim.gif"
    subprocess.run(
        [
            ffmpeg, "-y", "-hide_banner", "-loglevel", "error",
            "-f", "lavfi", "-i", "testsrc2=s=32x32:d=1:r=5", str(gif),
        ],
        check=True,
        capture_output=True,
        timeout=120,
    )
    plugin = make_plugin(
        _Context(), ffmpeg_path=str(Path(media_tools[0]).parent),
        provider_mode="gemini_native",
    )
    plugin.cache_root = str(tmp_path / "cache")

    report = run(plugin._diagnose(str(gif), event=_Event()))

    # ① 一行明确警示（就放在结论行上方）
    assert (
        "提醒   : ⚠️ 扩展名 .gif 不在已知视频容器内：显式路径不校验扩展名，"
        "将以 video/mp4 发送，provider 可能拒绝"
    ) in report
    # ② mime 行标注"推断"
    assert "MIME   : video/mp4 (推断)" in report
    # ③ 正常消息流的行为没变：它照样准备成功（闸门 ⓪ 负责筛选，这里不拦）
    assert "✅ 准备成功" in report


def test_diagnose_echoes_config_corrections_with_the_raw_value(
    media_tools, tiny_video, tmp_path, stub_logger
):
    """真实配置那种「15.0 配 13.0 闸门」，`/videotest` 必须打出用户写的 15.0。

    真机证据见 LIVE-TEST §10.2①：用户只能看到「上限=7.8MB」，看不到 15.0 被夹过。
    单趟归一之后是**一条**提示，原值 15.0 与最终值 11.7 在同一句里 ——
    旧实现会打两条（`配置提示:` 说收敛为 13.0、`配置修正:` 说下调为 11.7），
    同一个旋钮三个数字。
    """
    plugin = make_plugin(
        _Context(),
        ffmpeg_path=str(Path(media_tools[0]).parent),
        provider_mode="gemini_native",
        max_size_mb=13.0,
        compress_max_mb=15.0,
    )
    plugin.cache_root = str(tmp_path / "cache")

    report = run(plugin._diagnose(tiny_video, event=_Event()))

    assert "配置   : 闸门=13.0MB 上限=11.7MB" in report
    assert (
        "配置提示: compress_max_mb=15.0 超过体积闸门 max_size_mb=13.0，"
        "已按闸门的 90% 收敛为 11.7"
    ) in report
    # 同一个字段只有一条提示：不能再出现第二条"已自动下调"
    assert "配置修正:" not in report
    assert "已自动下调" not in report


def test_diagnose_does_not_flag_known_video_suffixes(
    media_tools, tiny_video, tmp_path, stub_logger
):
    """已知视频容器不得出现扩展名警示，MIME 行也不带"推断"。"""
    plugin = make_plugin(
        _Context(), ffmpeg_path=str(Path(media_tools[0]).parent),
        provider_mode="gemini_native",
    )
    plugin.cache_root = str(tmp_path / "cache")

    report = run(plugin._diagnose(tiny_video, event=_Event()))

    assert "提醒   :" not in report
    assert "(推断)" not in report
    assert "MIME   : video/mp4\n" in report or report.endswith("MIME   : video/mp4")


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


# --------------------------------------------------- 0.3.0：限流 / 缓存装配


class _RecordingLimiter:
    """记录进出次数的限流器替身（``SlotLimiter`` 的鸭子类型）。"""

    def __init__(self) -> None:
        self.limit = 2
        self.queue_timeout = 30.0
        self.entered = 0
        self.exited = 0

    def slot(self, name: str = ""):
        recorder = self

        class _Slot:
            async def __aenter__(self):
                recorder.entered += 1
                return self

            async def __aexit__(self, *exc):
                recorder.exited += 1
                return False

            async def wait_for_worker(self, worker):
                # 与真 ``_Slot`` 的同名方法同义：等 worker（替身不模拟取消）
                return await worker

        return _Slot()


def test_plugin_builds_the_limiter_from_config(stub_logger):
    """``ffmpeg_max_concurrency`` 决定**有没有**限流器；0 = 完全不建。"""
    off = make_plugin(_Context(), ffmpeg_max_concurrency=0)
    assert off.limiter is None, "0 表示关闭限流（逐字节等价 0.2.1）"

    on = make_plugin(
        _Context(), ffmpeg_max_concurrency=2, ffmpeg_queue_timeout_sec=12.0
    )
    assert on.limiter is not None
    assert on.limiter.limit == 2
    assert on.limiter.queue_timeout == 12.0


def test_diagnose_uses_the_limiter(media_tools, tiny_video, stub_logger):
    """``/videotest`` 也必须排队（否则诊断结果与真实链路不一致）。"""
    plugin = make_plugin(_Context(), provider_mode="gemini_native")
    point_to_ffmpeg(plugin, media_tools)
    plugin.cache_root = str(Path(tiny_video).parent / "cache")
    limiter = _RecordingLimiter()
    plugin.limiter = limiter

    report = run(plugin._diagnose(tiny_video, event=_Event()))

    assert limiter.entered == 1, "一次诊断只能占一个槽"
    assert limiter.exited == 1
    assert "✅ 准备成功" in report


def test_diagnose_keeps_working_without_a_limiter(media_tools, tiny_video, stub_logger):
    """限流关掉时 ``/videotest`` 行为不变（0 = 关闭）。"""
    plugin = make_plugin(_Context(), provider_mode="gemini_native")
    point_to_ffmpeg(plugin, media_tools)
    plugin.cache_root = str(Path(tiny_video).parent / "cache")
    plugin.limiter = None

    report = run(plugin._diagnose(tiny_video, event=_Event()))

    assert "✅ 准备成功" in report


def test_singleflight_with_the_cache_off_warns_once_at_startup(stub_logger):
    """``ffmpeg_singleflight=true`` + ``cache_enabled=false`` 是自相矛盾的配置。

    没有缓存就没有可共享的产物，singleflight 只能被跳过（见
    ``core/injector._prepare_video``）。这时必须**在启动时告警一次**：
    逐视频刷 WARNING 会淹没日志，静默跳过又会让用户以为"开了但没生效"
    （AUDIT-030 §C.1-7）。
    """
    plugin = make_plugin(_Context(), ffmpeg_singleflight=True, cache_enabled=False)
    warnings = [m for m in stub_logger.messages("warning") if "singleflight" in m]
    assert len(warnings) == 1, stub_logger.records
    assert "cache_enabled" in warnings[0]
    # 插件照常装配（只是这条组合不生效），不是错误
    assert plugin.config.ffmpeg_singleflight is True
    assert plugin.limiter is not None
    # 生效状态是显式记录的，不是"猜"出来的
    assert plugin.singleflight_effective is False

    # 默认配置（缓存开）不许刷这条
    stub_logger.clear()
    default_plugin = make_plugin(_Context())
    assert not [m for m in stub_logger.messages("warning") if "singleflight" in m]
    assert default_plugin.singleflight_effective is True

    # 显式关掉 singleflight 之后也不许刷（用户已经明确选择）
    stub_logger.clear()
    off = make_plugin(_Context(), ffmpeg_singleflight=False, cache_enabled=False)
    assert not [m for m in stub_logger.messages("warning") if "singleflight" in m]
    assert off.singleflight_effective is False


# ------------------------------------------------ 0.3.0：视觉缓存装配 / 绕过


def test_vision_cache_with_the_compress_cache_off_warns_once_at_startup(stub_logger):
    """``vision_cache_enabled=true`` + ``cache_enabled=false``：启动说清代价。

    P1-1 修好之后这条**不再**是"缓存不生效"（key 取"源 + 压缩参数"，压缩视频
    照样命中），但"需要压缩的视频每次投递都要重压一遍"是用户看不见的本地开销 ——
    启动时告警一次，逐视频刷 WARNING 会淹没日志（与 singleflight 那条同一个模板）。
    """
    plugin = make_plugin(
        _Context(),
        provider_mode="openai_compat",
        openai_base_url="http://127.0.0.1:9/v1",
        openai_api_key="sk-x",
        openai_model="vision-model",
        vision_cache_enabled=True,
        cache_enabled=False,
    )
    warnings = [m for m in stub_logger.messages("warning") if "vision_cache" in m]
    assert len(warnings) == 1, stub_logger.records
    text = warnings[0]
    assert "cache_enabled" in text
    # 语义必须是**修复后的事实**：缓存仍然命中，代价只是每次都要重压
    assert "仍然生效" in text
    assert "重新压一遍" in text
    # 缓存对象照常装配（只是效率打折），不是错误配置
    assert plugin.vision_cache is not None

    # 默认（压缩缓存开着）不许刷这条
    stub_logger.clear()
    make_plugin(
        _Context(),
        provider_mode="openai_compat",
        openai_base_url="http://127.0.0.1:9/v1",
        openai_api_key="sk-x",
        openai_model="vision-model",
    )
    assert not [m for m in stub_logger.messages("warning") if "vision_cache" in m]

    # 视觉缓存本来就关着：用户已经明确选择，不许刷
    stub_logger.clear()
    make_plugin(
        _Context(),
        provider_mode="openai_compat",
        openai_base_url="http://127.0.0.1:9/v1",
        openai_api_key="sk-x",
        openai_model="vision-model",
        vision_cache_enabled=False,
        cache_enabled=False,
    )
    assert not [m for m in stub_logger.messages("warning") if "vision_cache" in m]

    # gemini_native：缓存本来就不使用（已有 INFO 说明），不该再刷一条 WARNING
    stub_logger.clear()
    make_plugin(
        _Context(),
        provider_mode="gemini_native",
        vision_cache_enabled=True,
        cache_enabled=False,
    )
    assert not [m for m in stub_logger.messages("warning") if "vision_cache" in m]


def test_plugin_builds_the_vision_cache_from_config(stub_logger):
    on = make_plugin(
        _Context(),
        vision_cache_enabled=True,
        vision_cache_ttl_sec=120.0,
        vision_cache_max_entries=8,
    )
    assert on.vision_cache is not None
    assert on.vision_cache.ttl_sec == 120.0
    assert on.vision_cache.max_entries == 8

    off = make_plugin(_Context(), vision_cache_enabled=False)
    assert off.vision_cache is None, "关闭时完全不建缓存对象（逐字节等价 0.2.1）"


def test_diagnose_bypasses_the_vision_cache(monkeypatch, media_tools, tiny_video, stub_logger):
    """``/videotest`` 是诊断：必须真实调用视觉服务，不许命中缓存。"""
    plugin = make_plugin(
        _Context(),
        provider_mode="openai_compat",
        openai_base_url="http://127.0.0.1:9/v1",
        openai_api_key="sk-x",
        openai_model="vision-model",
        vision_cache_enabled=True,
    )
    point_to_ffmpeg(plugin, media_tools)
    plugin.cache_root = str(Path(tiny_video).parent / "cache")

    calls: list[int] = []

    async def fake_analyze_video(*args, **kwargs):
        calls.append(1)
        return VisionResult(True, "tiny.mp4", analysis="ok", elapsed_sec=0.1)

    monkeypatch.setattr(main_module, "analyze_video", fake_analyze_video)

    class _Tripwire:
        def lookup(self, key):  # pragma: no cover - 被调用即失败
            raise AssertionError("/videotest 不许查视觉缓存")

        def put(self, key, analysis):  # pragma: no cover
            raise AssertionError("/videotest 不许写视觉缓存")

    plugin.vision_cache = _Tripwire()

    first = run(plugin._diagnose(tiny_video, event=_Event()))
    second = run(plugin._diagnose(tiny_video, event=_Event()))

    assert len(calls) == 2, "两次诊断 = 两次真实调用"
    assert "✅ 准备成功" in first and "✅ 准备成功" in second


# --------------------------------------- 0.3.0：删掉废弃的 @register（结构锁）
#
# 真框架下的行为验收在 ``tests/test_contract.py``（装了 AstrBot 才跑）。
# 这里锁的是**源码结构**：没有 @register 之后，"插件能被识别"这件事完全依赖
# ``Star.__init_subclass__`` 与 metadata.yaml（AUDIT-030 §B.3）。


def test_plugin_registers_without_the_register_decorator():
    import ast

    source = (Path(__file__).resolve().parents[1] / "main.py").read_text(encoding="utf-8")
    tree = ast.parse(source)

    # 1) 不许再从 astrbot.api.star 导入 register
    imported: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and (node.module or "") == "astrbot.api.star":
            imported.update(alias.name for alias in node.names)
    assert "register" not in imported, f"还在导入 register: {sorted(imported)}"
    assert "Star" in imported and "Context" in imported, sorted(imported)

    # 2) 类上不许有任何装饰器（``@register(...)`` 就是唯一那个）
    classes = {
        node.name: node
        for node in ast.walk(tree)
        if isinstance(node, ast.ClassDef)
    }
    plugin = classes["VideoVisionPlugin"]
    decorators = [ast.unparse(item) for item in plugin.decorator_list]
    assert decorators == [], f"类上仍有装饰器: {decorators}"

    # 3) 必须仍是 Star 的子类：``__init_subclass__`` 靠这条自动登记
    bases = [ast.unparse(item) for item in plugin.bases]
    assert "Star" in bases, bases

    # 4) 文本层面也不许留半截（例如注释掉的 register 调用）
    assert "@register(" not in source
    assert "register(" not in source
