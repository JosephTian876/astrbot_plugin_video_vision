"""OpenAI 兼容视觉服务客户端。

``provider_mode=openai_compat`` 时使用：插件自己把视频发给配置里的
OpenAI 兼容端点做理解，再把理解结果作为文字注入对话 —— 主对话 provider
可以是任意模型，不要求它自己能看视频。

线上格式（实测结论，2026-09-29，Gemini 系中转站 + gemini-3.x）
------------------------------------------------------------------
视频必须以 **``image_url`` 内容块 + ``data:video/*;base64,...``** 的形式发送。
名字唬人，但 MIME 由 data URI 头部决定，端点会按视频处理::

    {"type": "image_url", "image_url": {"url": "data:video/mp4;base64,..."}}

对照实测（prompt token 262 -> 416 证明媒体真的进了模型）：

- ``image_url`` + video data URI  ✅ 模型能描述画面内容
- ``video_url``                    ❌ HTTP 200 但视频被静默丢弃（prompt token 不变）
- ``file`` / ``file_data``         ❌ 同上
- ``input_video``                  ❌ 同上

Responses API（``/responses``）也实测支持（2026-09-30，同站）：
- ``input_image`` + data URI ✅（input token 402，模型描述正确）
- ``input_file`` + file_data   ❌ 静默丢弃（276 token = 仅文本）

接口路径由用户在 ``openai_base_url`` 里**写全**：以 ``/responses`` 结尾走
Responses 协议，以 ``/chat/completions`` 结尾走 Chat Completions，
只填到 ``/v1`` 则兜底按 ``/chat/completions`` 处理。

**不依赖 AstrBot**；httpx 在使用时才导入（AstrBot 经 google-genai 自带 httpx）。
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass
from typing import Any

from .config import VideoConfig


@dataclass(frozen=True)
class VisionResult:
    """单个视频的视觉理解结果。"""

    ok: bool
    name: str
    analysis: str = ""
    reason: str = ""
    elapsed_sec: float = 0.0
    prompt_tokens: int = 0
    completion_tokens: int = 0


#: 视觉服务响应正文的硬上限（字符）。1MB 正文原样拼进 prompt 会直接顶爆主模型
#: 的上下文（实测第三方端点可能返回整段 HTML 错误页或复读）。超限截断并告警。
MAX_ANALYSIS_CHARS = 16 * 1024

#: 已告警过的模板，避免每轮刷屏
_warned_templates: set[str] = set()


def _warn_missing_placeholder(logger: Any | None) -> None:
    """``vision_qa_prompt`` 缺 ``{question}`` 时告警一次。"""
    if "vision_qa_prompt" in _warned_templates:
        return
    _warned_templates.add("vision_qa_prompt")
    if logger is not None:
        logger.warning(
            "[VideoVision] vision_qa_prompt 里没有 {question} 占位符："
            "用户的问题已追加到模板末尾（建议把 {question} 加回去，"
            "否则模板里的指示不会针对用户的具体问题）"
        )


def build_vision_prompt(
    config: VideoConfig, question: str, *, logger: Any | None = None
) -> str:
    """按「用户是否附带问题」挑选提示词。

    B2 策略：用户有问题就把原话带给视觉模型，得到针对性回答；
    没有问题（纯发视频）退化为通用详细描述。

    若用户把 ``vision_qa_prompt`` 里的 ``{question}`` 删掉了，0.2.0 会把模板
    原样发出去 —— 用户的问题被**静默丢弃**（实测：模板"没有占位符的模板" +
    问题"讲了什么" → 视觉模型只收到前者）。现在把问题追加到模板末尾并告警
    一次，绝不静默丢用户问题。

    Args:
        config: 插件配置。
        question: 用户随视频附带的文字（可为空）。
        logger: 可选日志器，用于告警「模板缺占位符」。

    Returns:
        最终发给视觉服务的提示词。
    """
    question = (question or "").strip()
    if question and config.vision_with_question:
        template = config.vision_qa_prompt or ""
        if "{question}" in template:
            prompt = template.replace("{question}", question)
        else:
            prompt = f"{template.rstrip()}\n\n用户的问题：「{question}」"
            _warn_missing_placeholder(logger)
        if prompt.strip():
            return prompt
    return config.vision_describe_prompt


def _resolve_endpoint(base_url: str) -> tuple[str, str]:
    """按用户填写的地址结尾决定协议与最终 URL。

    - 以 ``/responses`` 结尾 -> Responses API
    - 以 ``/chat/completions`` 结尾 -> Chat Completions
    - 其他（如只填到 ``/v1``）-> 兜底补 ``/chat/completions``
    """
    url = (base_url or "").strip().rstrip("/")
    if url.endswith("/responses"):
        return "responses", url
    if url.endswith("/chat/completions"):
        return "chat", url
    return "chat", url + "/chat/completions"


def _build_payload(
    kind: str, prompt: str, data_uri: str, config: VideoConfig
) -> dict:
    if kind == "responses":
        # Responses API。视频借道 input_image + data URI（实测唯一有效形态，
        # input_file 会被静默丢弃）。
        return {
            "model": config.openai_model,
            "input": [
                {
                    "role": "user",
                    "content": [
                        {"type": "input_text", "text": prompt},
                        {"type": "input_image", "image_url": data_uri},
                    ],
                }
            ],
            "max_output_tokens": config.openai_max_tokens,
        }
    return {
        "model": config.openai_model,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    # 见模块 docstring：实测只有 image_url 形态会被
                    # OpenAI 兼容端点按视频处理，video_url 等会被静默丢弃。
                    {"type": "image_url", "image_url": {"url": data_uri}},
                ],
            }
        ],
        "max_tokens": config.openai_max_tokens,
    }


def _parse_response(kind: str, data: dict) -> tuple[str, int, int]:
    """返回 (analysis, prompt_tokens, completion_tokens)，两种协议归一。"""
    if kind == "responses":
        text = str(data.get("output_text") or "").strip()
        if not text:
            texts = []
            for item in data.get("output") or []:
                for part in (item.get("content") or []):
                    if isinstance(part, dict) and part.get("type") == "output_text":
                        texts.append(str(part.get("text") or ""))
            text = "\n".join(s for s in texts if s).strip()
        usage = data.get("usage") or {}
        return (
            text,
            int(usage.get("input_tokens") or 0),
            int(usage.get("output_tokens") or 0),
        )
    choices = data.get("choices")
    if not isinstance(choices, list) or not choices:
        # ``choices: []``（或干脆没有该字段）不是"解析失败"，是"服务返回空内容"。
        # 0.2.0 直接 data["choices"][0] -> IndexError -> 报"响应解析失败"，
        # 把一次服务端空回答伪装成了客户端解析 bug。
        usage = data.get("usage") or {}
        return (
            "",
            int(usage.get("prompt_tokens") or 0),
            int(usage.get("completion_tokens") or 0),
        )
    first = choices[0] if isinstance(choices[0], dict) else {}
    message = first.get("message") or {}
    analysis = (message.get("content") or "").strip()
    usage = data.get("usage") or {}
    return (
        analysis,
        int(usage.get("prompt_tokens") or 0),
        int(usage.get("completion_tokens") or 0),
    )


def _classify_error(exc: Exception) -> str:
    """把异常翻译成一条不含敏感信息的短原因（写日志/注入给模型都用它）。

    返回值统一以「视觉服务」开头。``hooks/llm_request._short_reason`` 依赖这个
    前缀把这批原因**整体前置判定** —— 否则「可调小体积闸门」里的「闸门」二字
    会让一次超时被归类成「体积超限」，模型收到的说明就是错的。
    """
    text = f"{type(exc).__name__}: {exc}"
    low = text.lower()
    if "certificate" in low or "ssl" in low:
        return (
            "视觉服务 SSL 证书校验失败（自查证书是否过期；"
            "确认信任该端点后可在插件配置里关闭 vision_verify_ssl）"
        )
    if "timeout" in low or "timed out" in low:
        return "视觉服务请求超时（可调大 openai_timeout_sec 或调小体积闸门）"
    # 注意别用裸的 "name" 做匹配：KeyError('name') 这类无关异常会被误判成连接问题。
    if (
        "connect" in low
        or "resolve" in low
        or "getaddrinfo" in low
        or "nodename" in low
        or "name or service not known" in low
        or "name resolution" in low
        or "unreachable" in low
    ):
        return "视觉服务无法连接（检查 openai_base_url 与网络/代理）"
    return f"视觉服务调用失败: {text[:180]}"


async def analyze_video(
    data_uri: str,
    name: str,
    prompt: str,
    *,
    config: VideoConfig,
    logger: Any | None = None,
) -> VisionResult:
    """把单个视频发给 OpenAI 兼容端点做理解。任何失败都返回 ok=False。"""
    log = logger

    try:
        import httpx
    except ImportError:
        return VisionResult(
            False,
            name,
            reason="视觉服务不可用：缺少 httpx（AstrBot 官方环境自带，请检查安装）",
        )

    kind, url = _resolve_endpoint(config.openai_base_url)
    payload = _build_payload(kind, prompt, data_uri, config)
    headers = {
        "Authorization": f"Bearer {config.openai_api_key}",
        "Content-Type": "application/json",
    }

    async def _post() -> Any:
        async with httpx.AsyncClient(
            timeout=httpx.Timeout(config.openai_timeout_sec),
            verify=config.vision_verify_ssl,
        ) as client:
            return await client.post(url, json=payload, headers=headers)

    started = time.monotonic()
    try:
        # httpx 的 Timeout 是**分阶段**上限（连接/写入/读取各算各的），
        # 一个慢速滴灌的端点可以永远碰不到其中任何一条。这里再加一道整体上限，
        # 确保单个视频最多占用 openai_timeout_sec，不会把整个请求挂死。
        resp = await asyncio.wait_for(_post(), timeout=config.openai_timeout_sec)
        elapsed = time.monotonic() - started
    except Exception as exc:  # noqa: BLE001 - 网络异常一律消化
        return VisionResult(
            False,
            name,
            reason=_classify_error(exc),
            elapsed_sec=time.monotonic() - started,
        )

    if resp.status_code != 200:
        snippet = resp.text[:300] if resp.text else ""
        return VisionResult(
            False,
            name,
            reason=f"视觉服务返回 HTTP {resp.status_code}: {snippet}",
            elapsed_sec=elapsed,
        )

    try:
        data = resp.json()
        analysis, prompt_tok, completion_tok = _parse_response(kind, data)
    except Exception as exc:  # noqa: BLE001
        return VisionResult(
            False, name, reason=f"视觉服务响应解析失败: {exc}", elapsed_sec=elapsed
        )

    if not analysis:
        return VisionResult(
            False, name, reason="视觉服务返回了空内容", elapsed_sec=elapsed
        )

    # 第三方的回答是不可信数据，长度也必须设上限：把 1MB 正文原样拼进 prompt
    # 会直接顶爆主模型上下文，还会让日志/内存失真。
    if len(analysis) > MAX_ANALYSIS_CHARS:
        if log is not None:
            log.warning(
                "[VideoVision] 视觉服务返回正文过长（%d 字符），已截断为 %d 字符",
                len(analysis),
                MAX_ANALYSIS_CHARS,
            )
        analysis = analysis[:MAX_ANALYSIS_CHARS] + "…（响应过长，已截断）"

    if log is not None:
        log.info(
            "[VideoVision] 视觉理解完成: %s | %s | %.1fs | token %s+%s",
            name,
            kind,
            elapsed,
            prompt_tok,
            completion_tok,
        )
    return VisionResult(
        True,
        name,
        analysis=analysis,
        elapsed_sec=elapsed,
        prompt_tokens=prompt_tok,
        completion_tokens=completion_tok,
    )


async def analyze_videos(
    videos: list[tuple[str, str]],
    question: str,
    *,
    config: VideoConfig,
    logger: Any | None = None,
) -> list[VisionResult]:
    """并发分析多个视频。

    Args:
        videos: [(文件名, data URI), ...]
        question: 用户随消息附带的文字（可为空串）。
    """
    prompt = build_vision_prompt(config, question, logger=logger)
    tasks = [
        analyze_video(uri, name, prompt, config=config, logger=logger)
        for name, uri in videos
    ]
    results = await asyncio.gather(*tasks, return_exceptions=True)
    out: list[VisionResult] = []
    for (name, _), result in zip(videos, results):
        if isinstance(result, VisionResult):
            out.append(result)
        else:  # gather 捕到的漏网异常
            out.append(
                VisionResult(False, name, reason=_classify_error(result))
            )
    return out
