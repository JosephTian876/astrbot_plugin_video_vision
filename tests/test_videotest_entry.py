# 测试用假 id，禁止写入真实号码。
"""``/videotest`` 的三个视频入口：同条消息 / 引用消息 / 显式路径。

对应本轮修复（见 FIX-REPORT §5.9）：用户最自然的入口是"引用那条视频消息再发
``/videotest``"，而**"先单独发视频、再单独发 /videotest" 在事件模型下不可能成立**
（两条消息是两个独立事件）。这里把三条入口 + 「找不到视频时必须说清原因」
逐条锁死。

真框架（AstrBot 4.29.1）对齐验证见 ``work/vv-review/reply-verify/verify.py``：
那里用真实的 ``WakingCheckStage`` / ``CommandFilter`` / ``Reply`` / ``Video``
跑同一组场景。
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from types import SimpleNamespace

from astrbot.api.message_components import Reply, Video
from vv_plugin.main import VideoVisionPlugin

# --------------------------------------------------------------- 测试用装配


class _Context:
    """``resolve_provider`` 只需要这一个方法；返回 None = 没有可用 provider。"""

    async def get_using_provider_async(self, umo=None):
        return None


class _Plain:
    """纯文本段。插件不 import ``Plain``（只按鸭子类型看 ``chain``），够用即可。"""

    def __init__(self, text: str) -> None:
        self.text = text


class _Api:
    """假的 OneBot ``call_action``：按预设返回 payload 或抛异常。"""

    def __init__(self, payload=None, error: BaseException | None = None) -> None:
        self.payload = payload
        self.error = error
        self.calls: list[tuple[str, dict]] = []

    async def call_action(self, action: str, **params):
        self.calls.append((action, params))
        if self.error is not None:
            raise self.error
        return self.payload


def _event(text: str, chain=None, api: _Api | None = None):
    event = SimpleNamespace(
        role="admin",
        message_str=text,
        unified_msg_origin="test-umo",
        message_obj=SimpleNamespace(message=list(chain or [])),
        plain_result=lambda value: value,
        get_message_str=lambda: text,
    )
    if api is not None:
        event.bot = SimpleNamespace(api=api)
    return event


def _make_plugin(tmp_path: Path, **raw) -> VideoVisionPlugin:
    plugin = VideoVisionPlugin(
        _Context(), {"provider_mode": "gemini_native", **raw}
    )
    # 真实 cache_root 在 ~/.astrbot 下；测试一律写临时目录
    plugin.cache_root = str(tmp_path / "cache")
    return plugin


def _drive(plugin: VideoVisionPlugin, event, path: str = "") -> str:
    """驱动命令生成器，返回唯一一条回复文本。"""

    async def collect():
        return [item async for item in plugin.cmd_videotest(event, path)]

    items = asyncio.run(collect())
    assert len(items) == 1, items
    return items[0]


# ------------------------------------------------------------------ 三个入口


def test_quoted_video_is_diagnosed(tiny_video, tmp_path, stub_logger):
    """引用视频消息 + ``/videotest``：视频来自 ``Reply.chain``。

    真机证据：2026-10-26 11:34:52 的引用事件产出的报告里，文件名是
    ``media_video_20261026113452316_48f6.mp4`` —— 那是 11:34:52 那一刻才从**被
    引用消息**下载的副本（原视频的副本是 11:34:38 落盘的另一个名字）。
    """
    plugin = _make_plugin(tmp_path)
    reply = Reply(id="9001", chain=[Video(path=tiny_video)], sender_id="10001")
    event = _event("/videotest", chain=[reply, _Plain("/videotest")])

    text = _drive(plugin, event)

    assert "🔍 视频链路诊断" in text
    assert Path(tiny_video).name in text
    assert "文件不存在" not in text
    assert "用法（任选一种）" not in text


def test_same_message_video_with_command_is_diagnosed(
    tiny_video, tmp_path, stub_logger
):
    """视频和 ``/videotest`` 落在同一条消息里（平台支持图文混排时）。"""
    plugin = _make_plugin(tmp_path)
    event = _event("/videotest", chain=[Video(path=tiny_video), _Plain("/videotest")])

    text = _drive(plugin, event)

    assert "🔍 视频链路诊断" in text
    assert Path(tiny_video).name in text


def test_explicit_path_still_works(tiny_video, tmp_path, stub_logger):
    """显式路径永远有效，且优先于消息里的引用。"""
    plugin = _make_plugin(tmp_path)
    reply = Reply(id="9002", chain=[Video(path="/nonexistent/other.mp4")])
    event = _event(f"/videotest {tiny_video}", chain=[reply, _Plain("/videotest")])

    text = _drive(plugin, event, tiny_video)

    assert "🔍 视频链路诊断" in text
    assert Path(tiny_video).name in text
    assert "other.mp4" not in text


# ------------------------------------------------ 找不到视频时必须说清是哪种情况


def test_no_video_prints_usage_and_no_diagnosis(tmp_path, stub_logger):
    """不含视频时**不会误触发**诊断，只给用法。"""
    plugin = _make_plugin(tmp_path)
    event = _event("/videotest", chain=[_Plain("/videotest")])

    text = _drive(plugin, event)

    assert "用法（任选一种）" in text
    assert "🔍 视频链路诊断" not in text
    assert "本条消息和引用的消息里都没有视频" in text
    # 旧文案那句做不到的话必须消失
    assert "再发送 /videotest" not in text
    assert "引用（回复）那条视频消息" in text


def test_quoted_message_without_content_is_explained(tmp_path, stub_logger):
    """平台没把被引用消息交出来（``Reply.chain`` 为空）时要说清这一点。

    这正是用户最容易误判成"命令不生效"的分支：适配器 ``get_msg`` 偶发失败后
    ``Reply.chain`` 是空的，插件看不到任何内容。
    """
    plugin = _make_plugin(tmp_path)
    reply = Reply(id="9003", chain=[], sender_id="10001")
    api = _Api(error=RuntimeError("retcode 100: 消息不存在"))
    event = _event("/videotest", chain=[reply, _Plain("/videotest")], api=api)

    text = _drive(plugin, event)

    assert "🔍 视频链路诊断" not in text
    assert "没有把被引用消息的内容交给插件" in text
    # 重试过一次 get_msg（两次参数形态都试），失败也不影响给用户回话
    assert [call[0] for call in api.calls] == ["get_msg", "get_msg"]


def test_quoted_non_video_message_is_explained(tmp_path, stub_logger):
    """引用了消息，但里面确实没有视频 —— 说"没有视频"，不是"没找到参数"。"""
    plugin = _make_plugin(tmp_path)
    reply = Reply(id="9004", chain=[_Plain("在吗")], sender_id="10001")
    event = _event("/videotest", chain=[reply, _Plain("/videotest")])

    text = _drive(plugin, event)

    assert "🔍 视频链路诊断" not in text
    assert "引用的那条消息里没有视频" in text


def test_quoted_video_is_recovered_via_onebot_get_msg(
    tiny_video, tmp_path, stub_logger
):
    """``Reply.chain`` 为空时按 OneBot ``get_msg`` 重取一次视频。"""
    plugin = _make_plugin(tmp_path)
    reply = Reply(id="9005", chain=[], sender_id="10001")
    api = _Api(
        payload={
            "status": "ok",
            "retcode": 0,
            "data": {
                "message": [
                    {"type": "video", "data": {"file": "x.mp4", "path": tiny_video}}
                ]
            },
        }
    )
    event = _event("/videotest", chain=[reply, _Plain("/videotest")], api=api)

    text = _drive(plugin, event)

    assert "🔍 视频链路诊断" in text
    assert Path(tiny_video).name in text
    assert api.calls[0] == ("get_msg", {"message_id": "9005"})


def test_unreadable_quoted_video_says_so(tmp_path, stub_logger):
    """找到视频但下载失败（链接过期）时，不能报成"没有视频"。"""

    class _UnreadableVideo(Video):
        async def convert_to_file_path(self) -> str:
            raise RuntimeError("url expired")

    plugin = _make_plugin(tmp_path)
    reply = Reply(id="9006", chain=[_UnreadableVideo()], sender_id="10001")
    event = _event("/videotest", chain=[reply, _Plain("/videotest")])

    text = _drive(plugin, event)

    assert "🔍 视频链路诊断" not in text
    assert "没能取到本地文件" in text
    assert "没有视频" not in text
