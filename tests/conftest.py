"""pytest 公共装配。

三件事：

1. 把 ``tests/stubs`` 放进 ``sys.path`` —— 插件在模块级 ``import astrbot``，
   而测试机（CI / 用户本地）不装 AstrBot。桩只覆盖插件用到的符号。
2. 把**仓库根目录**挂成一个固定别名的包 ``vv_plugin``。插件内部用的是相对
   导入（``from ..core.config import ...``），必须以包的形式导入；用固定别名
   而不是目录名，是为了让"克隆到任意目录名都能跑测试"。
3. 提供 ffmpeg 素材与 logger 夹具。素材**现场生成**（不提交二进制），HTTP
   全部走本机 ``http.server`` 假端点，**不联网**。
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
import types
from pathlib import Path

import pytest

TESTS_DIR = Path(__file__).resolve().parent
REPO_ROOT = TESTS_DIR.parent
STUBS_DIR = TESTS_DIR / "stubs"
PLUGIN_PACKAGE = "vv_plugin"

# ---- 1. 桩包 ----
if str(STUBS_DIR) not in sys.path:
    sys.path.insert(0, str(STUBS_DIR))

# ---- 2. 插件包（固定别名，与目录名无关）----
if PLUGIN_PACKAGE not in sys.modules:
    _pkg = types.ModuleType(PLUGIN_PACKAGE)
    _pkg.__path__ = [str(REPO_ROOT)]  # type: ignore[attr-defined]
    _pkg.__doc__ = "被测插件的包别名（tests/conftest.py 装配）"
    sys.modules[PLUGIN_PACKAGE] = _pkg


def _find_media_tools() -> tuple[str, str] | None:
    """定位 ffmpeg / ffprobe：``VV_FFMPEG_DIR`` -> ``PATH``。

    刻意**不再**硬编码 ``D:\\ffmpeg\\bin``（0.2.1 的做法）：那是开发机的私有路径，
    写进公开仓库没有意义；更糟的是它**优先于 PATH**，于是本机永远测不到"PATH 上
    有 ffmpeg"这条路径，CI 上的行为也就无法在本地复现。
    """
    directory = os.environ.get("VV_FFMPEG_DIR", "")
    if directory:
        for name in ("ffmpeg", "ffmpeg.exe"):
            ffmpeg = Path(directory) / name
            ffprobe = Path(directory) / name.replace("ffmpeg", "ffprobe")
            if ffmpeg.is_file() and ffprobe.is_file():
                return str(ffmpeg), str(ffprobe)
    ffmpeg = shutil.which("ffmpeg")
    ffprobe = shutil.which("ffprobe")
    if ffmpeg and ffprobe:
        return ffmpeg, ffprobe
    return None


def _ffmpeg_is_required() -> bool:
    """``VV_REQUIRE_FFMPEG=1`` 时缺 ffmpeg 必须**失败**，不能静默跳过。

    否则 CI 会绿得莫名其妙：29 个用例（含参数化约 40+ 例）只是被 skip，
    覆盖腰斩而没人发现（ARCH-REVIEW §7.3-6）。
    """
    return os.environ.get("VV_REQUIRE_FFMPEG", "").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }


MEDIA_TOOLS = _find_media_tools()

#: 缺 ffmpeg 时给用户看的那句话（skip 与 fail 共用）
_FFMPEG_HELP = "需要真实 ffmpeg/ffprobe（可用 VV_FFMPEG_DIR 指定所在目录，或装进 PATH）"


@pytest.fixture(scope="session")
def media_tools() -> tuple[str, str]:
    """(ffmpeg, ffprobe) 真实可执行文件路径；没有则跳过相关用例。"""
    if MEDIA_TOOLS is None:
        if _ffmpeg_is_required():
            pytest.fail(
                f"{_FFMPEG_HELP}；当前 VV_REQUIRE_FFMPEG=1，缺 ffmpeg 视为失败"
                "（CI 里就是这么设的）",
                pytrace=False,
            )
        pytest.skip(_FFMPEG_HELP)
    return MEDIA_TOOLS


def _generate(ffmpeg: str, output: Path, args: list[str]) -> Path:
    output.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [
            ffmpeg,
            "-y",
            "-hide_banner",
            "-loglevel",
            "error",
            *args,
            str(output),
        ],
        check=True,
        capture_output=True,
        timeout=120,
    )
    return output


@pytest.fixture(scope="session")
def tiny_video(media_tools, tmp_path_factory) -> str:
    """2 秒 160x120 视频（几十 KB），用于正常链路。"""
    ffmpeg, _ = media_tools
    output = tmp_path_factory.mktemp("media") / "tiny.mp4"
    return str(
        _generate(
            ffmpeg,
            output,
            [
                "-f", "lavfi", "-i", "testsrc2=s=160x120:d=2:r=10",
                "-f", "lavfi", "-i", "sine=f=440:d=2",
                "-c:v", "libx264", "-preset", "ultrafast", "-crf", "28",
                "-pix_fmt", "yuv420p", "-c:a", "aac", "-b:a", "64k",
            ],
        )
    )


@pytest.fixture(scope="session")
def noisy_video(media_tools, tmp_path_factory) -> str:
    """4 秒高熵视频（CRF 18 下仍有几百 KB），用于压缩/截断/体积上限。"""
    ffmpeg, _ = media_tools
    output = tmp_path_factory.mktemp("media") / "noise.mp4"
    return str(
        _generate(
            ffmpeg,
            output,
            [
                "-f", "lavfi", "-i",
                "nullsrc=s=320x240:d=4:r=12,geq=random(1)*255:128:128",
                "-f", "lavfi", "-i", "sine=f=300:d=4",
                "-c:v", "libx264", "-preset", "ultrafast", "-crf", "18",
                "-pix_fmt", "yuv420p", "-c:a", "aac", "-b:a", "64k",
            ],
        )
    )


@pytest.fixture(scope="session")
def long_video(media_tools, tmp_path_factory) -> str:
    """30 秒低码率视频：用极小预算压它时能暴露"码率下限反超预算"。"""
    ffmpeg, _ = media_tools
    output = tmp_path_factory.mktemp("media") / "long.mp4"
    return str(
        _generate(
            ffmpeg,
            output,
            [
                "-f", "lavfi", "-i", "testsrc2=s=160x120:d=30:r=10",
                "-f", "lavfi", "-i", "sine=f=220:d=30",
                "-c:v", "libx264", "-preset", "ultrafast", "-crf", "28",
                "-pix_fmt", "yuv420p", "-c:a", "aac", "-b:a", "64k",
            ],
        )
    )


@pytest.fixture(scope="session")
def ffmpeg_dir(media_tools) -> str:
    """ffmpeg 所在目录（``ffmpeg_path`` 支持填目录）。"""
    return str(Path(media_tools[0]).parent)


@pytest.fixture
def stub_logger():
    """清空并返回桩 logger（插件与测试共用同一个对象）。"""
    from astrbot.api import logger as recording_logger

    recording_logger.clear()
    yield recording_logger
    recording_logger.clear()


@pytest.fixture
def sparse_file(tmp_path):
    """造一个指定字节数的**稀疏**文件（秒建大文件，不占磁盘）。"""

    def _make(name: str, size_bytes: int) -> str:
        path = tmp_path / name
        with open(path, "wb") as handle:
            handle.truncate(size_bytes)
        return str(path)

    return _make


@pytest.fixture
def make_placeholder():
    """构造一个 VideoPlaceholder（避免每个测试重复拼参数）。"""
    from vv_plugin.core.placeholder import VideoPlaceholder

    def _make(path: str, name: str | None = None) -> VideoPlaceholder:
        return VideoPlaceholder(name=name or os.path.basename(path), path=path)

    return _make
