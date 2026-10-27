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
        # 「按配置拒绝」按体积/时长/两者拆成三条（0.2.1 只有笼统的一条，
        # 于是"仅超时长"也被告知 "it exceeds the size limit"）
        ("rejected_size", "it exceeds the size limit"),
        ("rejected_duration", "its duration exceeds the limit"),
        (
            "rejected_size_and_duration",
            "it exceeds the size and duration limits",
        ),
        ("rejected", "it was rejected by the configured size or duration policy"),
        ("missing", "the file could not be read"),
        ("duration", "its duration could not be read"),
    ],
)
def test_short_reason_prefers_machine_readable_kind(kind, expected):
    assert hooks._short_reason("随便什么原因", kind) == expected


def test_short_reason_has_no_second_phrase_table():
    """结构：码 -> 短语只有 ``core/decide.py`` 一份（本轮结构重构）。

    0.2.1 在本文件里另存了 12 行 ``_SHORT_REASONS``，与判决侧各改各的。现在
    ``_short_reason`` 只查 ``decide.explain``，这里把"不许再长出一张表"钉住：
    对 ``decide.SHORT_REASONS`` 的每个键，hooks 的输出必须**逐字等于**判决侧的
    文案（不是"差不多"，是同一个函数）。
    """
    from vv_plugin.core import decide

    assert not hasattr(hooks, "_SHORT_REASONS"), "本地码表应已删除"
    for code in decide.SHORT_REASONS:
        assert hooks._short_reason("随便什么原因", code) == decide.explain(code), code
    # 判决码（不是对外 kind）也必须走同一句话
    for code in ("no_ffmpeg", "unknown_duration"):
        assert hooks._short_reason("", code) == decide.explain(code), code


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


def test_question_drops_both_attachment_prefixes():
    """``_extract_question`` 的前缀过滤必须与 ``placeholder._PREFIXES`` 同源。

    0.2.1 在 ``hooks`` 里另存了一份逐字节相同的 ``_ATTACHMENT_PREFIXES``
    （ARCH-REVIEW R6），改一处忘另一处就会把 ``[File Attachment: ...]``
    —— QQ/NapCat 投递视频的真实形态 —— 里的本机路径当"用户问题"外发。
    现在唯一来源是 ``core.placeholder.is_attachment_line()``，这条用例把
    两种前缀都钉住。
    """
    for prefix in ("[Video Attachment", "[File Attachment"):
        event = make_event(
            f"{prefix}: name a.mp4, path C:\\Users\\me\\secret\\a.mp4]\n真正的问题"
        )
        text = hooks._extract_question(event, SimpleNamespace())
        assert text == "真正的问题", (prefix, text)
        assert "secret" not in text and "C:\\" not in text

    # 前缀集合只有一份：hooks 里不许再长出同名字面量元组
    import inspect

    source = inspect.getsource(hooks)
    assert "_ATTACHMENT_PREFIXES = (" not in source
    assert "is_attachment_line(" in source


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


# ------------------------------------------------- 0.3.0：排队超时 -> busy


def test_queue_timeout_is_reported_as_busy(tiny_video, stub_logger):
    """限流排队超时 -> ``busy`` 短语，**不是** ``timed out while transcoding``。

    走真实 ``SlotLimiter``：另起一个协程占住唯一的槽，``build_video_blocks``
    的 ``ffmpeg_queue_timeout_sec`` 到点后必须产出 ``ok=False kind="busy"``，
    而不是让 ffmpeg 真跑起来（那样限流形同虚设）。
    """
    from vv_plugin.core.limits import SlotLimiter

    async def scenario():
        limiter = SlotLimiter(1, queue_timeout=0.01, logger=stub_logger)
        gate = asyncio.Event()
        holding = asyncio.Event()

        async def holder():
            async with limiter.slot():
                holding.set()
                await gate.wait()

        task = asyncio.create_task(holder())
        await asyncio.wait_for(holding.wait(), timeout=5)
        parts = [_TextPart(f"[Video Attachment: name a.mp4, path {tiny_video}]")]
        try:
            return await hooks.build_video_blocks(
                parts,
                config=VideoConfig(provider_mode="gemini_native"),
                cache_root="",
                limiter=limiter,
            )
        finally:
            gate.set()
            await task

    prepared, consumed, failures = asyncio.run(scenario())

    assert prepared == []
    assert consumed == set()
    assert failures == ["the server is busy processing other videos"]
    assert "timed out while transcoding" not in " ".join(failures)
    # 排队超时的那条 WARNING 必须出现且只出现一次
    queued = [
        item for item in stub_logger.messages("warning") if "ffmpeg 排队超时" in item
    ]
    assert len(queued) == 1, stub_logger.records
    assert "limit=1" in queued[0]


# -------------------------------- 0.3.0：0 = 关闭限流，逐字节等价 0.2.1


def test_unlimited_matches_the_limited_pipeline_byte_for_byte(tiny_video, stub_logger):
    """``ffmpeg_max_concurrency=0``（``limiter=None``）必须与开限流**逐字节**相同。

    "0 = 回到 0.2.1"不能只是口头承诺：同一份输入分别走
    ``NULL_LIMITER`` 与真实 ``SlotLimiter``（无竞争），产物字节、注入的
    data URI、体积、动作标签必须完全一致；并且不限流那条**不建队列、不打
    排队日志**（``SlotLimiter(0)._acquire`` 直接 return）。
    """
    from vv_plugin.core.limits import NULL_LIMITER, SlotLimiter
    from vv_plugin.core.media import content_hash

    config = VideoConfig(provider_mode="gemini_native")

    def prepare(limiter):
        parts = [_TextPart(f"[Video Attachment: name a.mp4, path {tiny_video}]")]
        return asyncio.run(
            hooks.build_video_blocks(
                parts, config=config, cache_root="", limiter=limiter
            )
        )

    unlimited, consumed_u, failures_u = prepare(None)
    limited_limiter = SlotLimiter(2, queue_timeout=30.0, logger=stub_logger)
    limited, consumed_l, failures_l = prepare(limited_limiter)

    assert failures_u == [] and failures_l == []
    assert consumed_u == consumed_l == {0}
    injection_u = unlimited[0].injection
    injection_l = limited[0].injection

    # 真正发出去的字节必须一模一样（不是"看起来差不多"）
    assert injection_u.data_uri == injection_l.data_uri
    assert content_hash(injection_u.used_path) == content_hash(injection_l.used_path)
    assert injection_u.action == injection_l.action
    assert injection_u.size_mb == injection_l.size_mb
    assert injection_u.duration_sec == injection_l.duration_sec

    # 不限流：没有令牌队列、没有等待、没有排队日志
    assert NULL_LIMITER._queue is None, "关闭限流时不许建 asyncio.Queue"
    assert not [m for m in stub_logger.messages("info") if "ffmpeg 排队" in m]
    # 开限流但无竞争：同样不排队（waits 只统计真的等过的）
    assert limited_limiter.stats.waits == 0
    assert limited_limiter.inflight == 0
    assert limited_limiter.available == 2


# ------------------------------------- 0.3.0：视觉结果缓存（hook 内部单元层）
#
# 端到端（真 httpx + 真端点 + 真 ffmpeg）在 ``tests/test_vision_cache.py``；
# 这里锁住的是"命中路径不调用服务、构造出的结果零成本"以及多视频的顺序契约。


def _fake_injection(content_key: str = "as-is:deadbeef", name: str = "a.mp4"):
    from vv_plugin.core.injector import VideoInjection

    return VideoInjection(
        block={
            "type": "audio_url",
            "audio_url": {"url": "data:video/mp4;base64,AAAA"},
        },
        source_path=name,
        used_path=name,
        size_mb=1.0,
        duration_sec=2.0,
        action="as-is",
        content_key=content_key,
    )


def make_openai_config(**kwargs):
    options = {
        "provider_mode": "openai_compat",
        "openai_base_url": "http://127.0.0.1:9/v1",
        "openai_api_key": "sk-x",
        "openai_model": "vision-model",
    }
    options.update(kwargs)
    return VideoConfig(**options)


def _counting_analyze(monkeypatch, calls: list[list[tuple[str, str]]]):
    from vv_plugin.core.openai_vision import VisionResult

    async def fake_analyze_videos(videos, question, *, config, logger=None):
        calls.append(list(videos))
        return [
            VisionResult(
                True,
                name,
                analysis=f"分析{len(calls)}",
                elapsed_sec=1.7,
                prompt_tokens=120,
                completion_tokens=9,
            )
            for name, _ in videos
        ]

    monkeypatch.setattr(hooks, "analyze_videos", fake_analyze_videos)
    return fake_analyze_videos


def test_vision_cache_hit_returns_a_zero_cost_result(monkeypatch, stub_logger):
    """命中不调用 ``analyze_videos``，且结果显式标记为零成本（elapsed/token = 0）。"""
    from vv_plugin.core.vision_cache import VisionCache

    calls: list = []
    _counting_analyze(monkeypatch, calls)
    cache = VisionCache(ttl_sec=600.0, max_entries=4)
    prepared = [hooks.PreparedVideo(0, "a.mp4", _fake_injection())]
    config = make_openai_config()

    first = asyncio.run(
        hooks._analyze_prepared_videos(
            prepared, "这视频讲了什么", config=config, logger=stub_logger, cache=cache
        )
    )
    second = asyncio.run(
        hooks._analyze_prepared_videos(
            prepared, "这视频讲了什么", config=config, logger=stub_logger, cache=cache
        )
    )

    assert len(calls) == 1, "第二次相同投递不该再调用视觉服务"
    assert first[0].analysis == second[0].analysis == "分析1"
    assert second[0].ok is True
    assert second[0].elapsed_sec == 0.0
    assert second[0].prompt_tokens == 0
    assert second[0].completion_tokens == 0
    assert cache.stats.hits == 1
    assert len([m for m in stub_logger.messages("info") if "视觉缓存命中" in m]) == 1


def test_vision_cache_miss_then_hit_only_sends_the_missing_video(
    monkeypatch, stub_logger
):
    """多视频：命中的不进请求，未命中的才发；返回顺序必须与输入一致。"""
    from vv_plugin.core.vision_cache import VisionCache, cache_key

    calls: list = []
    _counting_analyze(monkeypatch, calls)
    cache = VisionCache(ttl_sec=600.0, max_entries=4)
    config = make_openai_config()
    prepared = [
        hooks.PreparedVideo(0, "a.mp4", _fake_injection("as-is:aaa", "a.mp4")),
        hooks.PreparedVideo(1, "b.mp4", _fake_injection("as-is:bbb", "b.mp4")),
    ]

    # 先只把 a 的结果放进缓存（b 不在里面）
    from vv_plugin.core.openai_vision import build_vision_prompt, _resolve_endpoint

    prompt = build_vision_prompt(config, "问题")
    proto, url = _resolve_endpoint(config.openai_base_url)
    cache.put(
        cache_key(
            video="as-is:aaa",
            prompt=prompt,
            proto=proto,
            url=url,
            model=config.openai_model,
            max_tokens=config.openai_max_tokens,
        ),
        "a 的旧分析",
    )

    results = asyncio.run(
        hooks._analyze_prepared_videos(
            prepared, "问题", config=config, logger=stub_logger, cache=cache
        )
    )

    assert len(calls) == 1
    assert calls[0] == [("b.mp4", "data:video/mp4;base64,AAAA")], "只该发 b"
    assert results[0].analysis == "a 的旧分析"
    assert results[1].analysis == "分析1"
    assert results[0].name == "a.mp4" and results[1].name == "b.mp4"


def test_vision_cache_is_skipped_when_the_content_key_is_empty(
    monkeypatch, stub_logger
):
    """算不出"发出去的字节"时（content_key="")必须直接跳过缓存（fail-open）。"""
    from vv_plugin.core.vision_cache import VisionCache

    calls: list = []
    _counting_analyze(monkeypatch, calls)
    cache = VisionCache(ttl_sec=600.0, max_entries=4)
    prepared = [hooks.PreparedVideo(0, "a.mp4", _fake_injection(""))]

    for _ in range(2):
        asyncio.run(
            hooks._analyze_prepared_videos(
                prepared, "问题", config=make_openai_config(), logger=stub_logger,
                cache=cache,
            )
        )

    assert len(calls) == 2
    assert cache.stats.stores == 0
    assert len(cache) == 0


def test_disabled_vision_cache_calls_the_service_every_time(monkeypatch, stub_logger):
    """``cache=None``（``vision_cache_enabled=false``）：不查不写，逐字节等于 0.2.1。"""
    calls: list = []
    _counting_analyze(monkeypatch, calls)
    prepared = [hooks.PreparedVideo(0, "a.mp4", _fake_injection())]

    for _ in range(2):
        asyncio.run(
            hooks._analyze_prepared_videos(
                prepared, "问题", config=make_openai_config(), logger=stub_logger,
                cache=None,
            )
        )

    assert len(calls) == 2
    assert not [m for m in stub_logger.messages("info") if "视觉缓存" in m]
