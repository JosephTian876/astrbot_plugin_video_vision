"""视觉缓存 key 的"身份"维度：声明 MIME + API key 指纹（AUDIT-VV-R9 §2.2/§2.3）。

旧实现（``KEY_VERSION == 1``）的实测：

* 同一份字节改名 ``.mov`` 后与 ``.mp4`` **共用 key** —— 第 2 次 **0 次 POST**
  就拿到了 ``.mp4`` 那次的答案（端点若按 MIME 拒绝/差别对待，失败会被缓存掩盖）；
* 换 ``api_key`` 后同样复用旧答案（走中转站按 key 路由的用户"改对 key 之后，
  第一次重复投递仍然拿到错 key 时的回答"）。

这里锁住三件事：**同字节不同后缀必须 miss**、**换 key 必须 miss**、
**完全相同仍然命中**（防过度收紧）。
"""

from __future__ import annotations

from vv_plugin.core.vision_cache import KEY_VERSION, cache_key, key_fingerprint


def fields(**overrides):
    base = {
        "video": "as-is:bc98a7426643f2859c2f0c991b47f409",
        "prompt": "描述这段视频",
        "proto": "chat",
        "url": "https://example.invalid/v1/chat/completions",
        "model": "vision-model",
        "max_tokens": 512,
        "mime": "video/mp4",
        "api_key": "sk-unit-test",
    }
    base.update(overrides)
    return base


def test_the_declared_mime_is_part_of_the_key():
    """同一份字节、不同后缀 = 声明给端点的 MIME 不同 = 一次新的语义。"""
    mp4 = cache_key(**fields(mime="video/mp4"))
    mov = cache_key(**fields(mime="video/quicktime"))

    assert mp4 != mov, "同字节换后缀必须 miss（旧实现这里相等 -> 真串味）"
    assert len({mp4, mov}) == 2


def test_the_api_key_is_part_of_the_key():
    """换 key 视为不同缓存：中转站按 key 路由到不同后端/模型时回答会变。"""
    first = cache_key(**fields(api_key="sk-AAAAAA"))
    second = cache_key(**fields(api_key="sk-BBBBBB"))

    assert first != second, "换 api_key 必须 miss（旧实现这里相等 -> 真串味）"


def test_the_api_key_never_appears_in_the_key_or_its_payload():
    """载荷里只有 12 位指纹：明文 key 不许进任何可能被打印的结构。"""
    key = cache_key(**fields(api_key="sk-super-secret-value"))
    assert "sk-super-secret-value" not in key
    assert "super" not in key
    assert len(key) == 64 and all(ch in "0123456789abcdef" for ch in key)


def test_identical_inputs_still_share_the_key():
    """防过度收紧：完全相同（含同一个 key、同一个 MIME）必须仍然命中。"""
    assert cache_key(**fields()) == cache_key(**fields())


def test_mime_is_normalised():
    """大小写/空白不是语义差异，不许把同一个类型拆成两条缓存。"""
    assert cache_key(**fields(mime="VIDEO/MP4")) == cache_key(
        **fields(mime="  video/mp4  ")
    )


def test_key_fingerprint_is_stable_short_and_irreversible():
    assert key_fingerprint("sk-AAAAAA") == key_fingerprint("sk-AAAAAA")
    assert key_fingerprint("sk-AAAAAA") != key_fingerprint("sk-BBBBBB")
    assert len(key_fingerprint("sk-AAAAAA")) == 12
    assert key_fingerprint("") == ""  # 未配置 = 空指纹（请求本来就发不出去）
    assert key_fingerprint(None) == ""  # type: ignore[arg-type]


def test_the_payload_version_was_bumped_for_this_change():
    """旧条目可能已经串味，必须整批作废（AUDIT 的最小修复要求 +1）。"""
    assert KEY_VERSION == 2
