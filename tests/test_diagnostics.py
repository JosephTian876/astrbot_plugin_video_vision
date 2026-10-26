"""报告渲染层：``/videotest`` 逐字格式、闸门渲染、启动日志、用法提示。

对应本轮结构重构（ARCH-REVIEW §2「main.py 拆分」）：渲染从 ``main.py`` 搬到
``core/diagnostics.py`` 之后，报告格式第一次可以**在没有 ffmpeg、没有 AstrBot、
不跑子进程**的情况下逐字锁定。``tests/test_main.py`` 里那些"某几行包含某几段文字"
的断言由这里的**整份报告**断言兜底。

同时锁住三条层次约束（``core/`` 不得回头依赖上层）：

1. 不 import ``astrbot``（插件市场安全检查要求）；
2. 不 import ``hooks`` —— provider 由 ``main`` 解析成纯数据传进来；
3. ``render_report`` 的签名里没有 ``event``。
"""

from __future__ import annotations

import ast
import inspect
from pathlib import Path

import pytest

from vv_plugin.core import diagnostics
from vv_plugin.core.config import VideoConfig
from vv_plugin.core.decide import classify
from vv_plugin.core.diagnostics import (
    ProbeFacts,
    ProviderFacts,
    VisionReport,
    gate_lines,
    render_report,
    startup_lines,
    videotest_usage,
)
from vv_plugin.core.injector import PrepareOutcome, VideoInjection

REPO = Path(__file__).resolve().parents[1]
MODULE_PATH = REPO / "core" / "diagnostics.py"


def make_probe(path: str = r"C:\videos\demo.mp4", **kwargs) -> ProbeFacts:
    options = {
        "size_mb": 21.79,
        "duration_sec": 20.0,
        "ffmpeg": r"D:\ffmpeg\bin\ffmpeg.exe",
        "ffprobe": r"D:\ffmpeg\bin\ffprobe.exe",
    }
    options.update(kwargs)
    return ProbeFacts(path=path, **options)


def make_injection(**kwargs) -> VideoInjection:
    # 2.5MB 的 data URI -> 报告里的「编码: data URI 2.38 MB」
    uri = "data:video/mp4;base64," + "A" * 2_500_000
    options = {
        "block": {"type": "audio_url", "audio_url": {"url": uri}},
        "source_path": r"C:\videos\demo.mp4",
        "used_path": r"C:\cache\compressed.mp4",
        "size_mb": 1.79,
        "duration_sec": 20.0,
        "action": "compressed",
        "note": "体积 21.8MB > 闸门 13.0MB -> 1.8MB",
    }
    options.update(kwargs)
    return VideoInjection(**options)


# --------------------------------------------------------------- 逐字格式


def test_render_report_is_byte_exact_for_the_documented_example():
    """README「输出示例」那一份报告，逐字相等（含每个空行与缩进）。"""
    cfg = VideoConfig(
        max_size_mb=13.0,
        compress_max_mb=8.0,
        max_duration_sec=0.0,
        provider_mode="gemini_native",
    )
    probe = make_probe()
    plan = classify(
        size_mb=probe.size_mb, duration_sec=probe.duration_sec, config=cfg,
        mime_or_suffix=".mp4",
    )
    provider = ProviderFacts(
        provider_type="googlegenai_chat_completion",
        provider_id="gemini-2.5-flash",
        modalities=["text", "image", "audio"],
    )

    report = render_report(
        cfg=cfg,
        mode="gemini_native",
        probe=probe,
        plan=plan,
        provider=provider,
        outcome=PrepareOutcome(True, make_injection()),
        elapsed_sec=0.64,
    )

    assert report == (
        "🔍 视频链路诊断\n"
        "文件: demo.mp4\n"
        "体积: 21.79 MB\n"
        "时长: 20.0 s\n"
        "\n"
        "ffmpeg : ✅ D:\\ffmpeg\\bin\\ffmpeg.exe\n"
        "ffprobe: ✅ D:\\ffmpeg\\bin\\ffprobe.exe\n"
        "配置   : 闸门=13.0MB 上限=8.0MB CRF=30 策略=压缩\n"
        "编码上限: data URI 18MB ≈ 原始 13.5MB（编码后超过必失败，与体积闸门无关）\n"
        "模式   : gemini_native\n"
        "凭据   : 视觉服务 ❌ 缺 openai_base_url、openai_api_key\n"
        "provider: type=googlegenai_chat_completion id=gemini-2.5-flash "
        "modalities=[text, image, audio]\n"
        "闸门   : ⓪ File 组件扩展名筛选 → 本指令用显式路径，不适用\n"
        "         ① 类型 googlegenai_chat_completion ✅ 在允许列表\n"
        "         ② modalities 含 audio ✅\n"
        "         ③ 视觉服务凭据 ❌ 缺 openai_base_url、openai_api_key"
        "（本模式不使用）\n"
        "结论   : 体积 21.8MB > 闸门 13.0MB → 压缩\n"
        "prompt : （gemini_native 不调用视觉服务，无 prompt）\n"
        "\n"
        "✅ 准备成功（0.64s）\n"
        "动作: 已压缩\n"
        "产物: 1.79 MB / 20.0 s\n"
        "编码: data URI 2.38 MB\n"
        "MIME   : video/mp4（压缩产物，固定为 mp4）\n"
        "说明: 体积 21.8MB > 闸门 13.0MB -> 1.8MB\n"
        "\n"
        "预计上传耗时: 约 14s（provider 预算 180s）\n"
        "✅ 在预算内"
    )


def test_render_report_reports_the_failure_kind_and_reason():
    cfg = VideoConfig(max_size_mb=13.0, compress_max_mb=8.0, max_duration_sec=0.0)
    probe = make_probe(size_mb=2.0, duration_sec=0.0)
    plan = classify(
        size_mb=2.0, duration_sec=0.0, config=cfg, mime_or_suffix=".mp4"
    )

    report = render_report(
        cfg=cfg,
        mode="gemini_native",
        probe=probe,
        plan=plan,
        outcome=PrepareOutcome(
            False, reason="无法解析视频: C:\\videos\\broken.mp4", kind="corrupt"
        ),
        elapsed_sec=0.08,
    )

    assert "时长: 读取失败" in report
    assert "结论   : 无法解析（时长读取失败）→ 会以 corrupt 失败" in report
    assert report.endswith(
        "❌ 准备失败（0.08s）\n"
        "分类: corrupt\n"
        "原因: 无法解析视频: C:\\videos\\broken.mp4"
    )


def test_render_report_marks_a_guessed_mime_and_warns_about_the_suffix():
    cfg = VideoConfig(max_size_mb=13.0, compress_max_mb=8.0, max_duration_sec=0.0)
    probe = make_probe(path="anim.gif", size_mb=0.17, duration_sec=2.0)
    plan = classify(
        size_mb=0.17, duration_sec=2.0, config=cfg, mime_or_suffix=".gif"
    )

    report = render_report(
        cfg=cfg,
        mode="gemini_native",
        probe=probe,
        plan=plan,
        outcome=PrepareOutcome(
            True,
            make_injection(
                used_path="anim.gif",
                size_mb=0.17,
                duration_sec=2.0,
                action="as-is",
                note="",
            ),
        ),
        elapsed_sec=0.06,
    )

    assert (
        "提醒   : ⚠️ 扩展名 .gif 不在已知视频容器内：显式路径不校验扩展名，"
        "将以 video/mp4 发送，provider 可能拒绝"
    ) in report
    assert "MIME   : video/mp4 (推断)" in report
    assert "说明:" not in report


def test_render_report_head_only_when_the_outcome_is_missing():
    """``outcome=None`` 只渲染诊断前半段（不猜结果）。"""
    cfg = VideoConfig(max_size_mb=13.0, compress_max_mb=8.0, max_duration_sec=0.0)
    probe = make_probe()

    report = render_report(
        cfg=cfg,
        mode="gemini_native",
        probe=probe,
        plan=classify(
            size_mb=probe.size_mb, duration_sec=probe.duration_sec, config=cfg,
            mime_or_suffix=".mp4",
        ),
    )

    assert report.startswith("🔍 视频链路诊断\n")
    assert report.rstrip().endswith("（gemini_native 不调用视觉服务，无 prompt）")
    assert "准备成功" not in report and "准备失败" not in report


def test_render_report_shows_credentials_prompt_and_vision_result():
    cfg = VideoConfig(
        max_size_mb=13.0,
        compress_max_mb=8.0,
        max_duration_sec=0.0,
        provider_mode="openai_compat",
        openai_base_url="https://vision.invalid/v1/chat/completions",
        openai_api_key="sk-unit-test-value",
        openai_model="vision-model",
    )
    probe = make_probe(size_mb=0.93, duration_sec=3.0)
    plan = classify(
        size_mb=0.93, duration_sec=3.0, config=cfg, mime_or_suffix=".mp4"
    )

    report = render_report(
        cfg=cfg,
        mode="openai_compat",
        probe=probe,
        plan=plan,
        outcome=PrepareOutcome(
            True,
            make_injection(
                source_path=r"C:\videos\demo.mp4",
                used_path=r"C:\videos\demo.mp4",
                size_mb=0.93,
                duration_sec=3.0,
                action="as-is",
                note="",
            ),
        ),
        elapsed_sec=0.05,
        vision=VisionReport(
            True,
            elapsed_sec=1.25,
            prompt_tokens=262,
            completion_tokens=17,
            analysis="X" * 500,
        ),
    )

    assert "模式   : openai_compat" in report
    assert "凭据   : 视觉服务 ✅ base_url / api_key / model 齐全" in report
    assert "provider: 不适用（插件直连视觉服务，不经过主对话 provider）" in report
    assert "① Gemini 系 provider 校验 → openai_compat 不适用" in report
    assert "② audio 模态校验 → openai_compat 不适用" in report
    assert "prompt : 请详细描述这段视频" in report
    assert "预计上传耗时: 约 7s（视觉服务预算 180s）" in report
    assert "✅ 视觉调用成功（1.2s | token 262+17）" in report
    assert "— 视觉模型回复节选 —" in report
    # 节选截断到 400 字符 + 一个省略号
    assert "X" * 400 + "…" in report
    assert "X" * 401 not in report


def test_vision_failure_line_is_rendered_after_the_budget():
    cfg = VideoConfig(
        max_size_mb=13.0,
        compress_max_mb=8.0,
        max_duration_sec=0.0,
        provider_mode="openai_compat",
        openai_base_url="https://vision.invalid/v1/chat/completions",
        openai_api_key="sk-unit-test-value",
        openai_model="vision-model",
    )
    probe = make_probe(size_mb=0.93, duration_sec=3.0)

    report = render_report(
        cfg=cfg,
        mode="openai_compat",
        probe=probe,
        plan=classify(
            size_mb=0.93, duration_sec=3.0, config=cfg, mime_or_suffix=".mp4"
        ),
        outcome=PrepareOutcome(True, make_injection(action="as-is", note="")),
        elapsed_sec=0.05,
        vision=VisionReport(False, reason="视觉服务请求超时"),
    )

    assert report.endswith("❌ 视觉调用失败: 视觉服务请求超时")


def test_over_budget_hint_names_the_right_knob(monkeypatch):
    """预计上传耗时超预算时，提示必须指向**本模式**的那个超时旋钮。"""
    monkeypatch.setattr(diagnostics, "UPLOAD_SEC_PER_MB", 300.0)
    cfg = VideoConfig(
        max_size_mb=13.0,
        compress_max_mb=8.0,
        max_duration_sec=0.0,
        provider_mode="gemini_native",
    )
    probe = make_probe(size_mb=1.0, duration_sec=2.0)

    report = render_report(
        cfg=cfg,
        mode="gemini_native",
        probe=probe,
        plan=classify(
            size_mb=1.0, duration_sec=2.0, config=cfg, mime_or_suffix=".mp4"
        ),
        outcome=PrepareOutcome(
            True, make_injection(size_mb=1.0, action="as-is", note="")
        ),
        elapsed_sec=0.01,
    )

    assert "⚠️ 可能超时，建议调低体积闸门或提高 provider timeout" in report


def test_prompt_preview_flags_a_template_without_the_placeholder():
    cfg = VideoConfig(vision_qa_prompt="没有占位符的模板")
    preview = diagnostics.prompt_preview(cfg, "openai_compat")
    assert "缺 {question}" in preview
    native = diagnostics.prompt_preview(VideoConfig(), "gemini_native")
    assert native.startswith("（gemini_native")


def test_prompt_preview_clips_at_two_hundred_chars():
    cfg = VideoConfig(vision_describe_prompt="词" * 1000)
    preview = diagnostics.prompt_preview(cfg, "openai_compat")
    assert preview == "词" * 200 + "…"


# ------------------------------------------------------------------- 闸门


def test_gate_lines_render_each_gate1_branch():
    cfg = VideoConfig(require_gemini_provider=True)
    allowed = "googlegenai_chat_completion"

    ok = "\n".join(gate_lines(cfg, provider_type=allowed, modalities=["audio"]))
    assert "① 类型 googlegenai_chat_completion ✅ 在允许列表" in ok
    assert "② modalities 含 audio ✅" in ok

    blocked = "\n".join(
        gate_lines(cfg, provider_type="openai_chat_completion", modalities=["text"])
    )
    assert "① 类型 openai_chat_completion ❌ 不在允许列表" in blocked
    assert (
        "② modalities 不含 audio ❌（视频块会被替换成 [Audio] 文本，模型看不到画面）"
    ) in blocked

    no_modalities = "\n".join(
        gate_lines(cfg, provider_type=allowed, modalities=None)
    )
    assert (
        "provider: type=googlegenai_chat_completion "
        "id=未声明 modalities=[未声明]"
    ) in no_modalities
    assert (
        "② modalities 未声明 → ⚠️ 放行（视频块可能被替换成 [Audio] 文本）"
    ) in no_modalities

    unresolved = "\n".join(gate_lines(cfg, error="no-context"))
    assert "provider: type=未声明 id=未声明 modalities=[未声明]" in unresolved
    assert "① provider 解析失败 ❌（no-context）" in unresolved

    escape = "\n".join(
        gate_lines(
            VideoConfig(require_gemini_provider=False),
            provider_type="openai_chat_completion",
            modalities=["audio"],
        )
    )
    assert (
        "① require_gemini_provider=false → 放行"
        "（类型 openai_chat_completion，不校验 MIME 兼容性）"
    ) in escape


def test_gate_lines_honours_the_extra_provider_types():
    """闸门 ① 的允许列表与钩子判定共用一个真相源（ARCH-REVIEW R8）。"""
    cfg = VideoConfig(
        require_gemini_provider=True, additional_provider_types=["my_proxy"]
    )
    lines = "\n".join(gate_lines(cfg, provider_type="my_proxy", modalities=["audio"]))
    assert "① 类型 my_proxy ✅ 在允许列表" in lines


def test_gate_lines_in_openai_mode_declare_the_gates_not_applicable():
    cfg = VideoConfig(
        provider_mode="openai_compat",
        openai_base_url="https://vision.invalid/v1",
        openai_api_key="k",
        openai_model="m",
    )
    lines = gate_lines(cfg, mode="openai_compat")
    assert lines[0] == "provider: 不适用（插件直连视觉服务，不经过主对话 provider）"
    assert lines[-1] == (
        "         ③ 视觉服务凭据 ✅ base_url / api_key / model 齐全"
    )


# --------------------------------------------------------------- 启动日志


def test_startup_lines_report_mode_and_mask_the_key():
    cfg = VideoConfig(
        provider_mode="openai_compat",
        openai_base_url="https://vision.invalid/v1",
        openai_api_key="sk-super-secret-value",
        openai_model="vision-model",
    )
    lines = startup_lines(
        cfg, version="9.9.9", ffmpeg="ffmpeg.exe", ffprobe="ffprobe.exe",
        cache_root="/tmp/cache",
    )
    text = "\n".join(message for _level, message in lines)
    assert "[VideoVision] v9.9.9 已加载 | 启用=True | 体积闸门=13.0MB" in text
    assert "模式=openai_compat" in text
    assert "key=…alue" in text
    assert "sk-super-secret-value" not in text
    assert all(level == "info" for level, _ in lines)


def test_startup_lines_warn_about_degradation_and_missing_ffmpeg():
    cfg = VideoConfig()  # 默认 openai_compat + 三项为空 -> 降级
    lines = startup_lines(cfg, version="1", cache_root="/tmp/cache")
    warnings = [message for level, message in lines if level == "warning"]
    assert len(warnings) == 2, warnings
    assert any("已降级为 gemini_native" in item for item in warnings)
    assert any("未找到 ffmpeg" in item for item in warnings)
    infos = [message for level, message in lines if level == "info"]
    assert any("模式=gemini_native" in item for item in infos)


def test_startup_lines_say_the_cache_is_off_when_disabled():
    cfg = VideoConfig(cache_enabled=False, provider_mode="gemini_native")
    text = "\n".join(
        message
        for _level, message in startup_lines(
            cfg, version="1", ffmpeg="f", ffprobe="p", cache_root="/tmp/cache"
        )
    )
    assert "缓存=已关闭" in text


# ------------------------------------------------------------------- 用法


@pytest.mark.parametrize(
    ("reason", "needle"),
    [
        ("no-media", "本条消息和引用的消息里都没有视频"),
        ("quote-content-missing", "没有把被引用消息的内容交给插件"),
        ("quote-no-video", "引用的那条消息里没有视频"),
        ("video-unreadable", "没能取到本地文件"),
    ],
)
def test_videotest_usage_explains_each_reason(reason, needle):
    text = videotest_usage(reason)
    assert needle in text
    assert "用法（任选一种）" in text
    assert "再发送 /videotest" not in text


def test_videotest_usage_without_a_reason_is_the_plain_help():
    assert videotest_usage("") == diagnostics.USAGE_TEXT
    assert videotest_usage("unknown-reason") == diagnostics.USAGE_TEXT


def test_videotest_usage_inserts_the_reason_before_the_warning_block():
    text = videotest_usage("quote-no-video")
    head, _, tail = text.partition("本次没找到视频：")
    assert head.rstrip().endswith("（平台支持时）")
    assert tail.startswith("引用的那条消息里没有视频\n注意：")


# ----------------------------------------------------------- 层次约束（结构）


def test_diagnostics_does_not_import_astrbot_or_hooks():
    """``core/`` 不得回头依赖上层：不 import ``astrbot``，也不 import ``hooks``。"""
    tree = ast.parse(MODULE_PATH.read_text(encoding="utf-8"))
    imported: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            imported.add(node.module or "" if node.level == 0 else f".{node.module}")

    assert not any(name.startswith("astrbot") for name in imported), imported
    assert not any("hooks" in name for name in imported), imported
    assert imported == {
        "__future__",
        "os",
        "collections.abc",
        "dataclasses",
        "typing",
        ".config",
        ".decide",
        ".injector",
        ".media",
        ".openai_vision",
        ".placeholder",
    }, f"diagnostics.py 的依赖变了: {sorted(imported)}"


def test_render_report_takes_no_event():
    """provider 解析留在 ``main``/``hooks``，渲染层只收纯数据。"""
    parameters = set(inspect.signature(render_report).parameters)
    assert "event" not in parameters
    assert "plugin" not in parameters
    assert parameters == {
        "cfg",
        "mode",
        "probe",
        "plan",
        "provider",
        "notices",
        "warnings",
        "outcome",
        "elapsed_sec",
        "vision",
        "uri_limit_mb",
        "uri_raw_limit_mb",
    }


def test_diagnostics_does_no_io():
    """渲染层是纯函数：不做文件/进程/网络 I/O（否则无法在纯单元层面锁格式）。"""
    tree = ast.parse(MODULE_PATH.read_text(encoding="utf-8"))
    names = {node.id for node in ast.walk(tree) if isinstance(node, ast.Name)}
    attrs = {node.attr for node in ast.walk(tree) if isinstance(node, ast.Attribute)}
    forbidden = {"open", "system", "popen", "urlopen", "socket", "subprocess"}
    assert not (names & forbidden), sorted(names & forbidden)
    assert not (attrs & forbidden), sorted(attrs & forbidden)


def test_vision_report_copies_a_vision_result_duck_typed():
    class Stub:
        ok = True
        elapsed_sec = 2.5
        prompt_tokens = 10
        completion_tokens = 3
        analysis = "画面里有一只猫"
        reason = ""

    report = VisionReport.of(Stub())
    assert report == VisionReport(
        True, elapsed_sec=2.5, prompt_tokens=10, completion_tokens=3,
        analysis="画面里有一只猫",
    )
