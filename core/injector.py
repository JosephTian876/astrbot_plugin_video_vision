"""把视频占位符转换成 Gemini provider 能识别的视频内容块。

这是插件的**执行**层：除 ``astrbot.api.logger`` 之外不依赖 AstrBot。

为什么留那一个例外：插件市场（LLM Guard）的日志规范要求记录器必须且只能从
``astrbot.api`` 取得，不允许直接用内置 ``logging``（``CHANGELOG.md`` 0.1.1 条）。
所以本模块不是"零框架依赖"，而是"零框架**内部**依赖"——``astrbot.api`` 是公开
API 面，其余（``astrbot.core.*`` / ``Context`` / ``AstrMessageEvent``）一概不碰，
``tests/test_docs.py::test_code_side_dependency_claims_match_reality`` 用 AST 锁着。

处理流程::

    路径存在性 -> 探测体积/时长 -> decide.classify() -> 执行 MediaPlan
        |
        +-- as-is             -> 直接编码源文件
        +-- compress/truncate -> ffmpeg(带缓存) -> 编码
        +-- reject / 无 ffmpeg / 读不出时长 -> 给出原因

判决本身（闸门、oversize_action、截断估算、缓存资格、失败分类）全在
``core/decide.py``：本模块里不再出现任何 ``if size_mb > config.max_size_mb``。
0.2.1 这些知识在本文件与 ``main._diagnose`` 各写了一遍，于是产出了"诊断说谎"
那一族 bug（结论行说「未超限，直接送」，下一行却是「❌ 分类: corrupt」）。

任何一步失败都返回 ``ok=False`` 的结果，**绝不抛异常打断对话**。
"""

from __future__ import annotations

import os
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from astrbot.api import logger as _log

from .config import VideoConfig
from .decide import REJECT_CODES, REJECT_REASON_PREFIX, MediaPlan, classify
from .limits import SingleFlight
from .media import (
    OUTPUT_SUFFIX,
    CompressCache,
    MediaToolError,
    compress_video,
    content_hash,
    file_size_mb,
    probe_duration,
    read_as_data_uri,
    resolve_ffmpeg_tools,
)
from .placeholder import VideoPlaceholder

#: 进程级 singleflight：同一份内容 + 同一组压缩参数**只转码一次**。
#:
#: 键就是下面 ``cache_key``（内容哈希 + 压缩参数 + ``truncate_to``），不再另算
#: 一遍 —— 若按"只有内容哈希"去重，就会把"同一视频、不同压缩参数"也合并掉。
#: 只用 ``threading.Event``：``prepare_video`` 跑在 worker 线程里，跨线程碰
#: ``asyncio`` 原语会 ``attached to a different loop``（见 ``core/limits.py``）。
SINGLEFLIGHT = SingleFlight()


@dataclass(frozen=True)
class VideoInjection:
    """一个准备好注入的内容块及其元信息。"""

    block: dict[str, Any]
    source_path: str
    used_path: str
    size_mb: float
    duration_sec: float
    action: str  # as-is | compressed | truncated | rejected
    note: str = ""
    #: 模型**实际看到的画面**的标识（视觉结果缓存 key 的第一个维度）。
    #:
    #: 直通时是 ``as-is:<源文件内容哈希>``；压缩/截断时是
    #: ``<action>:<压缩签名>``，签名 = 源内容哈希 + 压缩参数
    #: （``compress_max_mb`` / ``crf`` / ``preset`` / ``truncate_to``），
    #: 与压缩缓存 key **完全同源**。**刻意不用产物字节的哈希**：同一源 + 同一组
    #: 参数压出来的 mp4 每次都不一样（熵编码 + 封装时间戳 + ``-fs`` 在 GOP 边界
    #: 截断），取产物哈希会让重复投递永不命中（ARCH-VV-C2 P1-1）。
    #:
    #: 只在 ``openai_compat`` + ``vision_cache_enabled`` 时计算（gemini_native 没有
    #: 可缓存的文本结果，也不该为此白读一遍文件）。算不出来（文件已被删等）时是
    #: 空串 —— 钩子看到空串就**跳过缓存**，绝不因此阻断主流程。
    content_key: str = ""

    @property
    def data_uri(self) -> str:
        """供上层构造媒体 Part 的 data URI。"""
        return self.block["audio_url"]["url"]


@dataclass(frozen=True)
class PrepareOutcome:
    """准备结果。``ok=False`` 时 ``reason`` 说明原因（用于日志与 /videotest）。

    ``kind`` 是失败分类，取值与 ``MediaToolError.KINDS`` 一致，另有
    ``missing``（路径不存在）/ ``rejected``（配置拒绝）/ ``internal``（未预期异常）。
    由判决产生的 kind 一律取自 ``decide.MediaPlan.kind``（唯一真相源），执行期失败
    则取自 ``MediaToolError.kind``。
    它存在的意义是让上层不必靠中文字符串猜测失败原因 —— 0.2.0 里
    ``_short_reason`` 就是靠 ``"ffmpeg" in reason`` 判断，于是超时、退出码、
    未找到、产物为空全部被报成「ffmpeg is not available」。
    """

    ok: bool
    injection: VideoInjection | None = None
    reason: str = ""
    kind: str = ""


def _make_block(used_path: str) -> dict[str, Any]:
    """构造内容块。

    用 ``audio_url`` 承载视频是刻意为之：Gemini provider 的 ``process_audio_url``
    从 data URI 里取 MIME 再交给 ``Part.from_bytes``，因此 ``data:video/*`` 会被
    正确构建成**视频 Part**。详见 hooks/llm_request.py 的说明。

    MIME 必须**跟随实际文件**，不能硬编码：直通时是源格式（webm/mov/mkv…），
    压缩后才是 ffmpeg 产出的 mp4。谎报 MIME 在 Gemini 上碰巧能跑（它会嗅探真实
    容器），但换一个信任声明 MIME 的端点就会失败。
    """
    data_uri = read_as_data_uri(used_path)
    return {"type": "audio_url", "audio_url": {"url": data_uri}}


def prepare_video(
    placeholder: VideoPlaceholder,
    *,
    config: VideoConfig,
    cache_root: str | os.PathLike[str] | None = None,
    logger: Any | None = None,
    debug: bool = False,
) -> PrepareOutcome:
    """按配置准备一个视频，返回可注入的内容块或失败原因。

    ``debug=True`` 时以 INFO 级别输出逐步细节（不受 AstrBot 全局日志级别影响），
    对应插件配置里的 ``debug_log``。

    本函数**绝不抛异常**（模块 docstring 的契约）。0.2.0 里 ``tempfile.mkdtemp``
    在 ``try`` 之外，磁盘满时 ``OSError`` 直接穿出 —— 上游 ``hooks._handle`` 的
    兜底把**整条消息的所有视频**都跳过。现在整个实现体都在 ``try`` 内。

    Returns:
        PrepareOutcome。``ok=False`` 时 ``reason`` 可读、``kind`` 可机读。
    """
    log = logger or _log
    try:
        return _prepare_video(
            placeholder,
            config=config,
            cache_root=cache_root,
            log=log,
            debug=debug,
        )
    # 这条 noqa 在 pin 住的 ruff 0.15.22 下**真的**在被使用：删掉它，同一行
    # 立刻报 BLE001（已验证，见 FIX-REPORT §5.13-G）。ruff ≥0.16 的 BLE001 不再
    # 对"记一条日志 + 返回降级结果"的 handler 报警，所以在 0.17.x 眼里它会是
    # unused（RUF100）—— 那是**升级 ruff 时才需要处理**的事，别现在当成死代码删掉。
    except Exception as exc:  # noqa: BLE001 - 契约要求：任何失败都只返回结果
        log.warning(
            "[VideoVision] 准备视频时出现未预期异常: %s: %s",
            type(exc).__name__,
            exc,
            exc_info=True,
        )
        return PrepareOutcome(
            False,
            reason=f"未预期异常: {type(exc).__name__}: {exc}",
            kind="internal",
        )


def _prepare_video(
    placeholder: VideoPlaceholder,
    *,
    config: VideoConfig,
    cache_root: str | os.PathLike[str] | None,
    log: Any,
    debug: bool,
) -> PrepareOutcome:
    """``prepare_video`` 的实现体（允许抛异常，由外层兜住）。

    只有两步：**探测事实**（体积 / 时长 / ffmpeg 能力）和**执行判决**
    （:func:`core.decide.classify` 给出的 :class:`MediaPlan`）。本函数里不再出现
    任何 ``if size_mb > config.max_size_mb`` —— 那正是 0.2.1 里 ``main._diagnose``
    手工重算一遍、然后与这里分叉的那段知识。
    """

    def dbg(message: str, *args: Any) -> None:
        if debug:
            log.info("[VideoVision][debug] " + message, *args)

    source = placeholder.path

    #: 视觉缓存的 key 第 1 维要不要算。``gemini_native`` 不调用视觉服务、没有可
    #: 缓存的文本结果，那条路径连一次哈希都不做（与 0.2.1 的零开销一致）。
    vision_key_enabled = bool(
        config.vision_cache_enabled
        and config.effective_provider_mode == "openai_compat"
    )

    #: 源文件内容哈希的惰性缓存（只算一次）：压缩缓存 key 与视觉缓存 key 共用。
    #: 算不出来时留空串 —— 空标识必须让上层**跳过缓存**，绝不能拿同一个空串
    #: 去撞别人的条目。
    digest: list[str] = []

    def source_digest() -> str:
        if not digest:
            try:
                digest.append(content_hash(source))
            except Exception as exc:  # noqa: BLE001 - 算不出来就不缓存，绝不阻断
                log.debug("[VideoVision] 源文件哈希算不出来: %s", exc)
                digest.append("")
        return digest[0]

    def content_key_for(identity: str, action: str) -> str:
        """把「可复现标识」包装成 ``VideoInjection.content_key``；不适用时空串。

        直通（as-is）时 ``identity`` 就是源文件内容哈希；压缩/截断时是
        :func:`compress_signature` 的返回值（源内容哈希 + 压缩参数）。

        **为什么不再取"真正发出去的那份字节"的哈希**（0.3.0 的第一版做法）：
        压缩产物**不可复现** —— 同一源 + 同一组参数每次压出来的 mp4 都不一样
        （熵编码 + 封装时间戳 + ``-fs`` 在 GOP 边界截断），实测 3 次同源同参 =
        3 个不同的产物哈希。取产物哈希等于让视觉缓存在 ``cache_enabled=false``
        时**永不命中**（ARCH-VV-C2 P1-1）。而"源 + 参数"完全决定模型看到的画面，
        既稳定又可在压缩**之前**算出来。
        """
        if not vision_key_enabled or not identity:
            return ""
        return f"{action}:{identity}"

    def compress_signature(truncate_to: float) -> str:
        """压缩产物的可复现标识 —— 与**压缩缓存 key 完全同源**。

        同一份签名既当压缩缓存 key（``cache_enabled`` 那段）、又当视觉缓存 key
        的第 1 维，于是"有没有开压缩缓存"不会改变视觉缓存的 key。字段全是数字
        + 白名单字符串（``core/config.py`` 的 ``VALID_PRESETS`` 兜住 preset）。
        """
        source_hash = source_digest()
        if not source_hash:
            return ""
        return content_hash_from_text(
            f"{source_hash}|{config.compress_max_mb}"
            f"|{config.compress_crf}|{config.compress_preset}|{truncate_to}"
        )

    if not source or not os.path.exists(source):
        return PrepareOutcome(False, reason=f"文件不存在: {source}", kind="missing")
    if not os.path.isfile(source):
        return PrepareOutcome(
            False, reason=f"不是普通文件: {source}", kind="missing"
        )

    ffmpeg, ffprobe = resolve_ffmpeg_tools(config.ffmpeg_path)
    size_mb = file_size_mb(source)
    duration = probe_duration(source, ffprobe)
    dbg(
        "探测 %s | 体积=%.2fMB 时长=%.2fs | ffmpeg=%s ffprobe=%s",
        placeholder.name,
        size_mb,
        duration,
        ffmpeg or "缺失",
        ffprobe or "缺失",
    )

    # ---- 判决（唯一真相源，不做 I/O）----
    plan = classify(
        size_mb=size_mb,
        duration_sec=duration,
        config=config,
        mime_or_suffix=Path(source).suffix,
        has_ffmpeg=bool(ffmpeg),
        has_ffprobe=bool(ffprobe),
    )
    dbg(
        "结论 超体积=%s 超时长=%s | 闸门=%.1fMB 时长上限=%.1fs 策略=%s | "
        "plan=%s/%s 截断到=%.1fs 可缓存=%s",
        plan.oversize,
        plan.overlong,
        config.max_size_mb,
        config.max_duration_sec,
        config.oversize_action,
        plan.action,
        plan.reason_code,
        plan.truncate_to_sec,
        plan.cache_eligible,
    )

    # ---- 判决说"执行不了"：分类（kind）与文案都来自 plan ----
    if plan.blocked:
        return PrepareOutcome(
            False, reason=_blocked_reason(plan, config, source), kind=plan.kind
        )

    # ---- 无需处理，直接送 ----
    if plan.action == "as-is":
        dbg("无需处理，直接编码源文件")
        try:
            block = _make_block(source)
        except MediaToolError as exc:
            # 例如编码后超过 data URI 硬上限：必须变成可读的 ok=False，
            # 不能让异常冒到钩子层（那样整条消息的视频都不处理）
            return PrepareOutcome(False, reason=f"编码失败: {exc}", kind=exc.kind)
        except Exception as exc:  # noqa: BLE001
            return PrepareOutcome(False, reason=f"编码失败: {exc}", kind="internal")
        dbg("编码完成 | data URI %.2fMB", len(block["audio_url"]["url"]) / 1024 / 1024)
        return PrepareOutcome(
            True,
            VideoInjection(
                block=block,
                source_path=source,
                used_path=source,
                size_mb=size_mb,
                duration_sec=duration,
                action=plan.outcome_action,
                # 直通：发出去的就是源文件，标识 = 源内容哈希
                content_key=content_key_for(
                    source_digest() if vision_key_enabled else "",
                    plan.outcome_action,
                ),
            ),
        )

    # ---- 压缩 / 截断：参数全部来自 plan（要不要截、截到几秒都在里面）----
    truncate_to = plan.truncate_to_sec
    if plan.action == "truncate" and plan.oversize and truncate_to > 0:
        dbg(
            "truncate 按体积预算估算: %.1fMB -> %.1fs（源 %.1fs / %.1fMB）",
            config.max_size_mb,
            truncate_to,
            duration,
            size_mb,
        )

    # ---- 压缩产物的可复现标识（= 压缩缓存 key = 视觉缓存 key 的第 1 维）----
    # 一处算、两处用：``cache_enabled`` 的开关于是**不会**改变视觉缓存的 key。
    # 只在真要用时才付这次哈希（15MB 源 ≈ 10.7ms）；两个缓存都关时连哈希都不做。
    cache_ready = bool(config.cache_enabled and cache_root)
    signature = ""
    if vision_key_enabled or cache_ready:
        try:
            signature = compress_signature(truncate_to)
        except Exception as exc:  # noqa: BLE001 - 算不出来就不缓存，绝不阻断
            log.debug("[VideoVision] 压缩签名算不出来（本次不缓存）: %s", exc)

    # ---- 缓存查找 ----
    cache: CompressCache | None = None
    cache_key = ""
    if cache_ready and signature:
        try:
            cache_key = signature
            cache = CompressCache(cache_root, config.cache_max_mb)
            hit = cache.get(cache_key)
            if hit:
                cached_size = file_size_mb(hit)
                log.debug("[VideoVision] 压缩缓存命中: %s", cache_key)
                dbg("缓存命中 %s | %.2fMB", cache_key, cached_size)
                try:
                    block = _make_block(hit)
                except MediaToolError as exc:
                    return PrepareOutcome(
                        False, reason=f"缓存文件编码失败: {exc}", kind=exc.kind
                    )
                except Exception as exc:  # noqa: BLE001
                    return PrepareOutcome(
                        False, reason=f"缓存文件编码失败: {exc}", kind="internal"
                    )
                return PrepareOutcome(
                    True,
                    VideoInjection(
                        block=block,
                        source_path=source,
                        used_path=hit,
                        size_mb=cached_size,
                        duration_sec=plan.expected_duration_sec,
                        action=plan.outcome_action,
                        note=f"缓存命中；{plan.note}",
                        # 与实时压缩同一条标识：产物字节不可复现，能复现的只有
                        # "源 + 参数"（也就是这个 cache_key 本身）
                        content_key=content_key_for(signature, plan.outcome_action),
                    ),
                )
        except Exception as exc:  # noqa: BLE001
            log.debug("[VideoVision] 缓存不可用，继续实时压缩: %s", exc)
            cache = None

    # ---- 实时压缩 ----
    # 注意：mkdtemp 也可能抛 OSError（磁盘满 / 权限），它必须留在 try 内，
    # 由外层 prepare_video 兜住 —— 0.2.0 把它放在 try 之外，直接穿出契约。
    temp_dirs: list[str] = []
    flight_enabled = bool(
        config.ffmpeg_singleflight and cache is not None and cache_key
    )
    if config.ffmpeg_singleflight and not flight_enabled:
        # 逐视频只打一条 debug：启动时 `main` 已经用 WARNING 说过这件事
        # （ffmpeg_singleflight=true + 没有缓存 = 自相矛盾，见 AUDIT-030 §C.1-7）。
        log.debug(
            "[VideoVision] ffmpeg_singleflight=true 但压缩缓存不可用"
            "（cache_enabled=%s，cache_root=%s），本次跳过 singleflight",
            config.cache_enabled,
            "已配置" if cache_root else "空",
        )

    def _compress_once() -> str:
        """跑一次压缩并把产物发布到缓存；失败抛 ``MediaToolError``。

        临时目录在**这里**创建（而不是在飞行之前）：只有真的要动手压缩的线程
        才需要它 —— follower 直接复用 leader 已发布到缓存的产物，不建临时目录。
        """
        temp_dir = tempfile.mkdtemp(prefix="vv_compress_")
        temp_dirs.append(temp_dir)
        produced = Path(temp_dir) / f"out{OUTPUT_SUFFIX}"
        dbg(
            "开始压缩 | 目标上限=%.1fMB CRF=%s preset=%s 截断到=%.1fs 超时=%.0fs",
            config.compress_max_mb,
            config.compress_crf,
            config.compress_preset,
            truncate_to,
            config.max_compress_seconds,
        )
        compress_video(
            source,
            produced,
            max_mb=config.compress_max_mb,
            crf=config.compress_crf,
            preset=config.compress_preset,
            max_duration_sec=truncate_to,
            ffmpeg=ffmpeg,
            ffprobe=ffprobe,
            timeout=config.max_compress_seconds,
        )
        if cache is not None and cache_key:
            return cache.put(cache_key, produced)
        return str(produced)

    def _cached_artifact() -> str | None:
        """leader 是否已把**可共享**的产物放进缓存（follower 醒来后查一次）。"""
        if cache is None or not cache_key:
            return None
        return cache.get(cache_key)

    def _on_takeover(exc: BaseException | None) -> None:
        log.warning(
            "[VideoVision] 同一文件的另一次转码失败（%s），本条改为自行转码",
            exc,
        )

    def _on_overdue() -> None:
        """leader 超过等待预算还没结束：只**告警**，不另起一份（P2-1）。

        旧实现（VERIFY-VV-R9 §3.3）在这里另起一份：超时路径把 leader 的登记删了，
        同一个 key 于是出现两个并发 leader（实测 ``concurrent_peak=2``）。
        现在唯一的后果是"这条消息多等一会儿"，而且能查得到是谁慢。
        """
        log.warning(
            "[VideoVision] 同一文件的转码超过 %.0fs 仍未结束（%s），本条继续等待",
            config.max_compress_seconds + 5.0,
            cache_key,
        )

    try:
        compress_started = time.monotonic()
        try:
            if flight_enabled:
                # 键 = 上面已经算好的 cache_key（内容哈希 + 压缩参数 + truncate_to）。
                # 临界区同时包住 compress_video 与 cache.put —— 否则 follower
                # 醒来时缓存里还没有产物（AUDIT-030 §C.1-6）。
                final_path = SINGLEFLIGHT.run(
                    cache_key,
                    _compress_once,
                    probe=_cached_artifact,
                    wait_timeout=config.max_compress_seconds + 5.0,
                    on_join=lambda: dbg("加入同一文件的转码 %s", cache_key),
                    on_takeover=_on_takeover,
                    on_overdue=_on_overdue,
                )
            else:
                final_path = _compress_once()
        except MediaToolError as exc:
            return PrepareOutcome(
                False, reason=f"压缩失败: {exc}", kind=exc.kind
            )
        except Exception as exc:  # noqa: BLE001
            return PrepareOutcome(
                False, reason=f"压缩异常: {exc}", kind="internal"
            )

        new_size = file_size_mb(final_path)
        expected_duration = plan.expected_duration_sec
        new_duration = probe_duration(final_path, ffprobe) or expected_duration
        dbg(
            "压缩完成 | 耗时=%.2fs %.2fMB -> %.2fMB | 产物时长=%.2fs",
            time.monotonic() - compress_started,
            size_mb,
            new_size,
            new_duration,
        )

        note = f"{plan.note} -> {new_size:.1f}MB"
        if plan.applies_truncate:
            note += f"（按体积预算截断到 {truncate_to:.1f}s）"
        if expected_duration and new_duration < expected_duration * 0.8:
            # -fs 触顶会让 ffmpeg 提前封口：时长比预期更短，是硬上限的代价
            note += "（受 -fs 体积硬上限约束，产物比预期更短）"
        if new_size > config.max_size_mb:
            note += "（仍超闸门，已尽力）"
            log.warning(
                "[VideoVision] 压缩后仍超闸门: %.1fMB > %.1fMB",
                new_size,
                config.max_size_mb,
            )

        try:
            block = _make_block(final_path)
        except MediaToolError as exc:
            return PrepareOutcome(
                False, reason=f"压缩产物编码失败: {exc}", kind=exc.kind
            )
        except Exception as exc:  # noqa: BLE001
            return PrepareOutcome(
                False, reason=f"压缩产物编码失败: {exc}", kind="internal"
            )

        return PrepareOutcome(
            True,
            VideoInjection(
                block=block,
                source_path=source,
                used_path=final_path,
                size_mb=new_size,
                duration_sec=new_duration,
                action=plan.outcome_action,
                note=note,
                # 标识取"源 + 参数"（= cache_key），**不**取产物字节：后者每次
                # 都不一样，取它就等于让视觉缓存永不命中（ARCH-VV-C2 P1-1）
                content_key=content_key_for(signature, plan.outcome_action),
            ),
        )
    finally:
        # 产物此刻已编码进内存（data URI）或已移入缓存，临时文件不再需要。
        # 放在 finally 里，任何提前 return / 异常路径都不会泄漏临时目录。
        for directory in temp_dirs:
            _cleanup(directory)


def _blocked_reason(plan: MediaPlan, config: VideoConfig, source: str) -> str:
    """把「为什么执行不了」渲染成一句话（文案与 0.2.1 逐字一致）。

    判决只给 code，人话在这里拼：它需要 ``source`` 与 ``config.ffmpeg_path``，
    而 ``decide.classify`` 刻意不看这两样（纯函数、零 I/O）。
    """
    if plan.reason_code in {"empty", "corrupt"}:
        return f"{plan.note}: {source}"
    if plan.reason_code in REJECT_CODES:
        return f"{REJECT_REASON_PREFIX}（{plan.note}）"
    if plan.reason_code == "no_ffmpeg":
        hint = (
            f"配置的 ffmpeg_path 无效: {config.ffmpeg_path}"
            if config.ffmpeg_path
            else "请安装 ffmpeg 或在插件配置里指定 ffmpeg_path"
        )
        return f"需要压缩但未找到 ffmpeg（{plan.note}）。{hint}"
    if plan.reason_code == "unknown_duration":
        return f"无法读取时长，拒绝盲压缩（{plan.note}）"
    return plan.note


def content_hash_from_text(text: str) -> str:
    """对字符串取短哈希（缓存 key 用）。"""
    import hashlib

    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:32]


def _cleanup(path: str | os.PathLike[str]) -> None:
    import shutil

    try:
        shutil.rmtree(path, ignore_errors=True)
    except Exception:
        pass
