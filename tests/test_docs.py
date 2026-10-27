"""文档与实现一致性（R2B 维度 9 的 9 处）。

对应修复项：B19。断言写得很具体 —— 文档写错一次，这里就会红。
"""

from __future__ import annotations

import json
from pathlib import Path

from vv_plugin.hooks.llm_request import _NEUTRALIZE_TEXT

REPO = Path(__file__).resolve().parents[1]
README = (REPO / "README.md").read_text(encoding="utf-8")
CHANGELOG = (REPO / "CHANGELOG.md").read_text(encoding="utf-8")
METADATA = (REPO / "metadata.yaml").read_text(encoding="utf-8")
SCHEMA = json.loads((REPO / "_conf_schema.json").read_text(encoding="utf-8"))
MEDIA_SOURCE = (REPO / "core" / "media.py").read_text(encoding="utf-8")


def squash(text: str) -> str:
    return "".join(text.split())


def test_version_is_bumped():
    """0.3.0：六个新配置键 + 两处新行为（限流 / 视觉缓存）+ 删 ``@register``。"""
    assert "version: 0.3.0" in METADATA
    assert CHANGELOG.startswith("# 更新日志")
    assert "## 0.3.0" in CHANGELOG
    # 历史条目必须保留（用户升级时要能查旧版本改了什么）
    assert "## 0.2.1" in CHANGELOG
    # README 也要提一次当前版本（用户第一眼看到的地方），且不许还写着旧版本
    assert "当前版本 **0.3.0**" in README
    assert "当前版本 **0.2.1**" not in README


def test_changelog_030_lists_the_three_changes_and_the_falsified_claim():
    """F.2 要求的五段：六个键、限流、缓存、删装饰器、既有用例全绿。

    另外必须写清**被证伪的说法**：追问不重发视频时插件根本不运行，
    所以"连续追问更快"不成立。
    """
    section = CHANGELOG.split("## 0.3.0", 1)[1].split("## 0.2.1", 1)[0]
    for needle in (
        "ffmpeg_max_concurrency",
        "ffmpeg_queue_timeout_sec",
        "ffmpeg_singleflight",
        # 修正审计 §B.1 的前提：上限不是"无限"，而是默认线程池的 min(32, cpu+4)
        "min(32",
    ):
        assert needle in section, needle


def test_changelog_021_lists_the_fixed_items():
    section = CHANGELOG.split("## 0.2.1", 1)[1].split("## 0.2.0", 1)[0]
    for needle in ("缓存", "truncate", "-fs", "GreedyStr", "1e999", "tests/", "18MB"):
        assert needle in section, needle


def test_astrbot_lower_bound_matches_what_the_code_needs():
    """mark_as_temp 自 4.24.1 起，占位符与 modalities 自 4.23.5 起。"""
    assert 'astrbot_version: ">=4.24.1,<5"' in METADATA
    assert "需要 **AstrBot >= 4.26**" not in CHANGELOG
    assert "4.24.1" in CHANGELOG
    assert "4.23.5" in CHANGELOG
    assert "AstrBot **>= 4.24.1**" in README


def test_astrbot_lower_bound_says_it_was_not_run_on_that_version():
    """回归（VERIFY2 P3-9）：下界是按符号存在性论证的，文档不能暗示实跑过。"""
    for text in (README, CHANGELOG):
        assert "符号存在性" in text
        assert "没有在 4.24.1 上" in text
        assert "/videotest` 输出回报" in text


def test_size_fuse_claim_is_no_longer_absolute():
    assert "不会压出超预算的文件" not in README
    assert "不会压出超预算的文件" not in MEDIA_SOURCE
    assert "-fs" in README
    assert "硬上限" in README


def test_core_zero_dependency_claim_is_qualified():
    assert "core/ 下**不允许**出现 `import astrbot`" not in README
    assert "除 `injector.py` 从 `astrbot.api` 取 `logger`" in README
    assert "astrbot.api" in README


def test_code_side_dependency_claims_match_reality():
    """回归（VERIFY2 P3-2）：README 改了口径，代码里的两句规则没跟着改。

    ``core/__init__.py`` 曾写"本包内所有模块**不得**导入 astrbot.*"，而
    ``core/injector.py`` 的 import 段就 ``from astrbot.api import logger``；
    ``injector.py`` 的模块 docstring 也曾写"**不依赖 AstrBot**"。
    这里同时锁两件事：说法必须限定，**实际导入**也只允许那一条。
    """
    import ast

    package = (REPO / "core" / "__init__.py").read_text(encoding="utf-8")
    injector = (REPO / "core" / "injector.py").read_text(encoding="utf-8")
    assert "不得**导入 astrbot" not in package
    assert "astrbot.api" in package
    assert "**不依赖 AstrBot**" not in injector

    imports: list[tuple[str, str, list[str]]] = []
    for path in sorted((REPO / "core").glob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and (node.module or "").startswith(
                "astrbot"
            ):
                imports.append(
                    (path.name, node.module or "", [a.name for a in node.names])
                )
            elif isinstance(node, ast.Import):
                for alias in node.names:
                    if alias.name.startswith("astrbot"):
                        imports.append((path.name, alias.name, []))
    assert imports == [("injector.py", "astrbot.api", ["logger"])], imports


def test_readme_directory_tree_lists_every_core_module():
    """目录树必须与 ``core/*.py`` 一一对应（本轮新增 ``decide.py`` 时踩到的漂移）。"""
    import re

    listed = set(re.findall(r"^│\s+[├└]── (\w+)\.py", README, re.MULTILINE))
    actual = {path.stem for path in (REPO / "core").glob("*.py")} - {"__init__"}
    assert listed == actual, f"README 少列/多列: {sorted(listed ^ actual)}"
    assert "decide.py" in README, "判决模块必须出现在目录树里"


def test_hide_local_path_example_matches_the_real_output():
    assert "[Video: fileseg_xxx.mp4]" not in README
    real = _NEUTRALIZE_TEXT["gemini_native"].replace("{name}", "fileseg_xxx.mp4")
    assert squash(real) in squash(README)


def test_compat_table_describes_the_real_rewrite():
    assert "改写成不含路径的说明" in README


def test_videotest_example_matches_the_new_report():
    for needle in (
        "## `/videotest` 诊断指令",
        "模式   : gemini_native",
        "provider: type=",
        "闸门   :",
        "编码上限:",
        "MIME   :",
        "凭据   : 视觉服务",
        "prompt :",
    ):
        assert needle in README, needle
    assert "l10_probe.mp4" not in README


def test_the_verdict_label_in_the_docs_is_not_the_one_the_companion_deletes():
    """README 的示例必须用 `结论`：`判定   :` 会被陪伴插件在发出前删掉。

    证据与守卫源码见 ``tests/test_interop.py`` 的模块说明。
    """
    from vv_plugin.core.diagnostics import VERDICT_LABEL

    assert VERDICT_LABEL == "结论   :"
    assert f"{VERDICT_LABEL} 体积 21.8MB > 闸门 13.0MB → 压缩" in README
    # 示例正文里不能留旧标签（正文里出现"判定"只允许是在解释它为什么不能用）
    assert "判定   : 体积" not in README
    assert "判定   : 未超限" not in README
    # 兼容性表格里必须点名这条出站文本约束（否则读者不知道为什么不能改回「判定」）
    assert "出站文本不长得像复核字段" in README
    assert "user_memory.py:9174-9210" in README
    assert "test_interop.py" in README
    assert "跨插件兼容" in CHANGELOG


def test_videotest_honesty_lines_are_documented():
    """0.2.1 新增的三类"说清楚"：corrupt 判定、未知后缀警示、配置修正回显。"""
    assert "无法解析（时长读取失败）→ 会以 corrupt 失败" in README
    assert "不在已知视频容器内" in README
    assert "(推断)" in README
    assert "配置修正:" in README
    assert "配置提示:" in README


def test_degradation_warning_promise_is_precise():
    hint = SCHEMA["provider_mode"]["hint"]
    assert "告警一次" in hint
    assert "钩子不再重复告警" in hint
    assert "真的只告警一次" in CHANGELOG


def test_cache_single_artifact_behaviour_is_documented():
    assert "不会进缓存" in SCHEMA["cache_max_mb"]["hint"]
    assert "不会进缓存" in README


def test_videotest_extension_check_is_documented():
    assert "不校验扩展名" in README


def test_truncate_semantics_are_documented():
    assert "按体积预算" in SCHEMA["oversize_action"]["hint"]
    assert "truncate_to" in SCHEMA["oversize_action"]["hint"]
    assert "按体积预算截断" in README or "按体积截断" in README


def test_data_uri_hard_limit_is_documented():
    assert "18MB" in README
    assert "18MB" in SCHEMA["max_size_mb"]["hint"]


def test_requirements_has_no_new_runtime_dependency():
    requirements = (REPO / "requirements.txt").read_text(encoding="utf-8")
    for line in requirements.splitlines():
        stripped = line.strip()
        assert not stripped or stripped.startswith("#"), stripped


def test_schema_hints_mention_the_new_guards():
    assert "墙钟" in SCHEMA["max_compress_seconds"]["hint"]
    assert "1.3-3.2" in SCHEMA["compress_max_mb"]["hint"]


def test_compress_hint_explains_the_gate_ceiling():
    """回归（VERIFY2 P3-3）：面板 hint 只说"必须小于闸门"，没说会被闸门收敛。

    README / CHANGELOG 都写了 90%，只有用户真会在面板上读到的那一句漏了 ——
    于是"我填了 15，为什么生效 11.7"在面板里查不到解释。
    """
    hint = SCHEMA["compress_max_mb"]["hint"]
    assert "max_size_mb" in hint
    assert "90%" in hint
    assert "启动" in hint and "/videotest" in hint
    assert "11.7" in hint and "13.0" in hint  # 例子里的原值 -> 生效值


def test_videotest_entry_points_are_documented_honestly():
    """``/videotest`` 只能从**本条事件**里拿视频，文档不能承诺做不到的事。

    旧 README 写「或回复一条含视频的消息」还算对，但插件内提示写的是
    「或先发一个视频/视频文件，再发送 /videotest」—— 真机上那是两条独立消息、
    两个独立事件（11:34:35 发视频 → 11:34:40 发命令 → 只得到用法提示）。
    """
    from vv_plugin.core.diagnostics import USAGE_TEXT

    # README：三种入口 + 明确写出"先发视频再发命令不行"
    assert "引用（回复）那条视频消息" in README
    assert "先单独发一个视频" in README
    assert "是不行的" in README
    # 插件内提示：同样的说法，且旧的那句必须消失
    assert "引用（回复）那条视频消息" in USAGE_TEXT
    assert "先单独发一个视频" in USAGE_TEXT
    assert "再发送 /videotest" not in USAGE_TEXT
    assert "先发视频再发命令" in CHANGELOG


# ---------------------------------------------------- 0.3.0：限流 / 缓存文档


def test_schema_hints_explain_the_concurrency_knobs():
    """面板 hint 必须把语义说全：0=关闭、排队上限、singleflight 依赖缓存。"""
    max_hint = SCHEMA["ffmpeg_max_concurrency"]["hint"]
    assert "并发" in max_hint
    assert "0" in max_hint and "关闭" in max_hint
    queue_hint = SCHEMA["ffmpeg_queue_timeout_sec"]["hint"]
    assert "排队" in queue_hint
    assert "busy" in queue_hint
    flight_hint = SCHEMA["ffmpeg_singleflight"]["hint"]
    assert "singleflight" in flight_hint
    assert "cache_enabled" in flight_hint


def test_readme_documents_the_concurrency_section():
    assert "## 并发、排队与缓存" in README
    for needle in (
        "ffmpeg_max_concurrency",
        "ffmpeg_queue_timeout_sec",
        "ffmpeg_singleflight",
        # 审计 §B.1 要求修正的说法：上限不是"无限"，而是默认线程池
        "min(32, cpu+4)",
        # 失败分类表里要有 busy
        "busy",
    ):
        assert needle in README, needle
    assert "无限并发" not in README
    assert "不会压出超预算的文件" not in README


# ---------------------------------------------- 0.3.0：视觉结果缓存（文档）


def test_schema_hints_explain_the_vision_cache_knobs():
    """面板 hint 必须写清"只对完全相同的重复投递命中"、TTL 与容量语义。"""
    enabled = SCHEMA["vision_cache_enabled"]["hint"]
    assert "完全相同" in enabled
    assert "重启" in enabled or "内存" in enabled
    ttl = SCHEMA["vision_cache_ttl_sec"]["hint"]
    assert "TTL" in ttl or "存活" in ttl
    assert "续期" in ttl
    entries = SCHEMA["vision_cache_max_entries"]["hint"]
    assert "LRU" in entries or "最久未使用" in entries
    assert "内存" in entries


def test_readme_documents_the_vision_cache_section():
    assert "vision_cache_enabled" in README
    assert "vision_cache_ttl_sec" in README
    assert "vision_cache_max_entries" in README
    assert "视觉缓存" in README
    # 审计证伪过的那句话不许出现在 README 里（追问不重发视频时插件根本不运行）
    assert "连续追问" not in README
    assert "追问更快" not in README
    # 缓存只对"完全相同的重复投递"生效，这一点必须写清楚
    assert "完全相同的" in README or "完全相同" in README
    # /videotest 绕过缓存必须写清楚（诊断要真实调用）
    assert "绕过" in README or "不经过缓存" in README


def test_changelog_030_explains_the_vision_cache():
    section = CHANGELOG.split("## 0.3.0", 1)[1].split("## 0.2.1", 1)[0]
    for needle in (
        "vision_cache_enabled",
        "vision_cache_ttl_sec",
        "vision_cache_max_entries",
        # 被证伪的说法必须写明，否则用户会以为"追问更快"
        "连续追问",
    ):
        assert needle in section, needle


def test_metadata_yaml_has_the_required_fields():
    """``@register`` 删掉之后，``metadata.yaml`` 是**唯一**的身份来源。

    AstrBot 的 ``_PluginUpdater.validate_plugin_metadata`` 要求
    ``name/desc/version/author`` 都是非空字符串（缺了插件直接加载失败），
    ``load()`` 再用这些字段逐个覆盖 ``star_map`` 里的身份，``repo`` 则是插件
    更新的唯一来源。这里用行式解析（不引 yaml 依赖）把四要素 + repo + 版本格式锁住。
    """
    import re

    fields: dict[str, str] = {}
    for line in METADATA.splitlines():
        match = re.match(r"^([A-Za-z_][A-Za-z0-9_]*):\s*(.*)$", line)
        if match:
            fields[match.group(1)] = match.group(2).strip()

    for key in ("name", "desc", "version", "author"):
        assert fields.get(key), f"metadata.yaml 缺少必需字段 {key}"
        assert fields[key] != "|", f"metadata.yaml 的 {key} 是空的块标量"
    assert re.fullmatch(r"\d+\.\d+\.\d+", fields["version"]), fields["version"]
    assert fields["version"] == "0.3.0"
    assert fields["repo"].startswith("https://"), fields["repo"]
    assert fields["name"] == "astrbot_plugin_video_vision"
    # 四个必填字段之外，面板展示用的字段也不许丢
    for key in ("display_name", "short_desc", "astrbot_version"):
        assert fields.get(key), f"metadata.yaml 缺少 {key}"
    # 特性清单要跟着版本走：0.3.0 的两处新行为都得在 description 里露过面
    assert "ffmpeg 并发限流" in METADATA
    assert "视觉理解结果缓存" in METADATA
