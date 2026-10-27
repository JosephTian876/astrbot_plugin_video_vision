"""仓库卫生守卫：被 git 跟踪的文件里不许出现本机私有路径或真实号码。

0.3.0 发布前的独立验收（VERIFY-C 第 4 项，FAIL）在测试里找到 5 处本机私有路径
和 5 处真实 QQ 号。它们不影响任何断言，却会把开发机信息带进公开仓库 —— 而且下
一个人照着抄一遍就会扩散。这里把"不许再出现"变成会失败的断言，失败时打印
``文件:行号``，方便直接定位。

判定口径（逐行扫描 ``git ls-files`` 列出的全部文本文件）：

* **私有路径** —— 一行里的盘符绝对路径按反斜杠切段后命中任一条：
  - 第一段是 ``Users``，第二段既不是占位写法也不在 :data:`PLACEHOLDER_USER_NAMES`；
  - 任一段等于 :data:`BANNED_USER_NAMES` 里的名字（本机用户名）；
  - 第一段是本机 AstrBot 安装目录（:data:`BANNED_INSTALL_DIRS`）。
  反斜杠先归一化（连续多个折成一个、``/`` 也折成反斜杠），所以 docstring 里的
  转义写法和正斜杠写法同样逃不掉；不含盘符的占位写法（如 ``<AstrBot python>``）
  天然放行。
* **号码** —— 长度 10 或 11 的独立数字串（现代 QQ 号 / 手机号的形状）。
  真实事故值是 **10 位**的，只查 11 位会漏掉它自己；5-9 位的数字串与普通常量
  （``65536`` / ``86400`` / ``300000`` …）无法区分，所以另用字段名口径兜住：
  ``sender_id=`` 之类的字段后面跟 5 位以上数字时，必须是 :data:`ALLOWED_NUMBERS`
  里的假号。
* 已知假号与公开编号列在 :data:`ALLOWED_NUMBERS` 并写明理由，其余一律失败。

守卫自己的源码提交后也在扫描范围内，所以本文件里不能出现完整的私有路径字面量，
也不能出现裸的 10-11 位数字 —— 合成样例都是拼接出来的。
"""

from __future__ import annotations

import functools
import re
import subprocess
from pathlib import Path
from typing import Callable

REPO = Path(__file__).resolve().parents[1]

#: 本机用户名（小写比较）。守卫必须知道要禁什么，所以名字本身出现在块名单里。
BANNED_USER_NAMES = frozenset({"uwu"})
#: 盘符根目录段就是 AstrBot 安装目录（本机路径，不是仓库里的相对路径）。
BANNED_INSTALL_DIRS = frozenset({"astrbot"})
#: ``Users`` 下允许出现的假用户名 —— 不是任何人的真实账号。
PLACEHOLDER_USER_NAMES = frozenset({"me", "user", "username", "demo"})
#: 允许出现的假号码 / 公开编号：值 -> 为什么它不是个人信息。
ALLOWED_NUMBERS = {
    "10001": "假 QQ 号（测试用）",
    "10086": "假手机号（测试用）",
    "40335025358": "公开的 GitHub Actions run id（VERIFY-C B2 已复核为非个人信息）",
    "0123456789": "十六进制字符表字面量（test_vision_cache 的 hex 断言），不是号码",
}

#: 盘符绝对路径（归一化之后），到空白 / 引号 / 尖括号为止。
DRIVE_PATH_RE = re.compile(r"""(?<![A-Za-z0-9])[A-Za-z]:\\[^\s"'<>`|]*""")
#: 独立的 10-11 位数字串（前后都不能紧贴别的数字）。
NUMBER_RE = re.compile(r"(?<!\d)(\d{10,11})(?!\d)")
#: 身份字段后面跟 5 位以上数字（兜住 5-9 位的旧 QQ 号，且不误伤普通常量）。
ID_FIELD_RE = re.compile(
    r"(?i)\b(?:sender_id|user_id|qq|uin|phone|mobile|tel)\b\s*[=:]\s*[\"']?(\d{5,11})"
)


def _normalize(line: str) -> str:
    """把转义写法与正斜杠折成单反斜杠：两种写法都逃不掉。"""
    collapsed = re.sub(r"\\{2,}", lambda _match: "\\", line)
    return collapsed.replace("/", "\\")


@functools.lru_cache(maxsize=1)
def _tracked_files() -> tuple[tuple[str, str], ...]:
    """``git ls-files`` 里的全部文本文件：``(相对路径, 内容)``；二进制跳过。"""
    listing = subprocess.run(
        ["git", "ls-files", "-z"],
        cwd=REPO,
        check=True,
        capture_output=True,
    )
    files: list[tuple[str, str]] = []
    for raw_name in listing.stdout.split(b"\0"):
        if not raw_name:
            continue
        relative = raw_name.decode("utf-8")
        # 跟踪的文件一定存在；读不到就让用例报错，不能静默跳过。
        data = (REPO / relative).read_bytes()
        if b"\0" in data:  # logo.png 之类的二进制
            continue
        files.append((relative, data.decode("utf-8", errors="replace")))
    return tuple(files)


def _is_placeholder_segment(segment: str) -> bool:
    """``<name>`` / ``%VAR%`` / ``${VAR}`` / ``$VAR`` / 纯点号都算占位写法。"""
    text = segment.strip("`'\"[]() ")
    return not text or text.startswith(("<", "%", "$", "{")) or not text.strip(".")


def _is_private_path(
    token: str, banned_users: frozenset[str] = BANNED_USER_NAMES
) -> bool:
    """``token`` 是归一化后的盘符路径；返回它是否含本机私有信息。"""
    parts = [part for part in token.split("\\")[1:] if part]
    if not parts:
        return False
    folded = [part.casefold() for part in parts]
    if any(name in folded for name in banned_users):
        return True
    if folded[0] in BANNED_INSTALL_DIRS:
        return True
    if folded[0] == "users" and len(parts) >= 2:
        if _is_placeholder_segment(parts[1]):
            return False
        return folded[1] not in PLACEHOLDER_USER_NAMES
    return False


def _private_path_details(
    line: str, banned_users: frozenset[str] = BANNED_USER_NAMES
) -> list[str]:
    """返回一行里命中的私有路径（空列表 = 干净）。"""
    return [
        f"本机私有路径 {token}"
        for token in DRIVE_PATH_RE.findall(_normalize(line))
        if _is_private_path(token, banned_users)
    ]


def _number_details(line: str) -> list[str]:
    """返回一行里命中的真实号码形状（空列表 = 干净）。"""
    details = [
        f"真实号码形状的数字串 {number}"
        for number in NUMBER_RE.findall(line)
        if number not in ALLOWED_NUMBERS
    ]
    details.extend(
        f"身份字段里的数字 {number}"
        for number in ID_FIELD_RE.findall(line)
        if number not in ALLOWED_NUMBERS
    )
    return details


def _scan(check: Callable[[str], list[str]]) -> list[str]:
    """逐行跑 ``check``，把命中格式化成 ``文件:行号: 说明``。"""
    problems: list[str] = []
    for relative, text in _tracked_files():
        for lineno, line in enumerate(text.splitlines(), 1):
            problems.extend(
                f"{relative}:{lineno}: {detail}" for detail in check(line)
            )
    return problems


def _assert_scan_covers_the_repository() -> None:
    """反空转：git 不可用 / 输出为空时，下面的断言会"绿得没有意义"。"""
    names = {relative for relative, _ in _tracked_files()}
    assert len(names) >= 50, f"只扫到 {len(names)} 个跟踪文件：{sorted(names)}"
    expected = {"main.py", "README.md", "CHANGELOG.md", "_conf_schema.json"}
    assert expected <= names, f"git ls-files 少列了：{sorted(expected - names)}"
    assert "tests/test_videotest_entry.py" in names


def test_tracked_files_have_no_private_paths() -> None:
    _assert_scan_covers_the_repository()
    problems = _scan(_private_path_details)
    assert not problems, (
        "发现本机私有路径（改成占位符或中性假路径）：\n" + "\n".join(problems)
    )


def test_tracked_files_have_no_real_qq_or_phone_numbers() -> None:
    _assert_scan_covers_the_repository()
    problems = _scan(_number_details)
    assert not problems, (
        "发现真实号码（换成 10001 / 10086 之类的假号）：\n" + "\n".join(problems)
    )


def test_detectors_flag_synthetic_samples() -> None:
    """反空转：检测函数本身失效时，上面两条会静默变绿。

    合成样例故意拼接：这样这一行**自己**不会被上面两条判成私有路径 / 真实号码。
    """
    users_path = "X:" + "\\" + "Users" + "\\" + "someone" + "\\" + "a.mp4"
    assert _private_path_details(users_path), "Users 下的真实用户名没被识别"
    root_path = "X:" + "\\" + "someone" + "\\" + "a.mp4"
    assert _private_path_details(
        root_path, frozenset({"someone"})
    ), "盘符根目录里的本机用户名没被识别"
    placeholder = "X:" + "\\" + "Users" + "\\" + "<name>" + "\\" + "a.mp4"
    assert not _private_path_details(placeholder), "占位写法被误判成私有路径"
    synthetic_number = "9" * 10
    assert _number_details("id=" + synthetic_number), "10 位号码没被识别"
    assert _number_details("id=" + "9" * 11), "11 位号码没被识别"
    assert not _number_details("sender_id=" + "10001"), "白名单里的假号被误判"
