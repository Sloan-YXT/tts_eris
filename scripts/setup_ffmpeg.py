#!/usr/bin/env python3
"""下载 ffmpeg 到项目根目录，供 video_to_audio 等脚本使用。"""
from __future__ import annotations

import os
import platform
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path
from urllib.request import urlretrieve

# BtbN FFmpeg Builds - Windows 64-bit GPL shared (较小体积)
FFMPEG_WIN64_URL = (
    "https://github.com/BtbN/FFmpeg-Builds/releases/download/latest/"
    "ffmpeg-master-latest-win64-gpl-shared.zip"
)

_ROOT = Path(__file__).resolve().parent.parent
FFMPEG_DIR = _ROOT / "ffmpeg"


def get_ffmpeg_exe() -> Path | None:
    """返回项目内 ffmpeg 可执行文件路径，若不存在则返回 None。"""
    if platform.system() != "Windows":
        return None
    # 结构: ffmpeg/bin/ffmpeg.exe
    exe = FFMPEG_DIR / "bin" / "ffmpeg.exe"
    return exe if exe.exists() else None


def setup_ffmpeg() -> bool:
    """下载并解压 ffmpeg 到项目目录。"""
    if platform.system() != "Windows":
        print("当前仅支持 Windows，请手动安装 ffmpeg")
        return False

    if get_ffmpeg_exe():
        print(f"ffmpeg 已存在: {get_ffmpeg_exe()}")
        return True

    zip_path = _ROOT / "ffmpeg.zip"
    try:
        print(f"正在下载: {FFMPEG_WIN64_URL}")
        urlretrieve(FFMPEG_WIN64_URL, zip_path)
    except Exception as e:
        print(f"下载失败: {e}")
        return False

    try:
        if FFMPEG_DIR.exists():
            shutil.rmtree(FFMPEG_DIR)
        FFMPEG_DIR.mkdir(parents=True)

        with zipfile.ZipFile(zip_path, "r") as zf:
            names = zf.namelist()
            # 找到顶层目录（如 ffmpeg-master-latest-win64-gpl-shared）
            parts = [n.split("/")[0] for n in names if "/" in n]
            prefix = parts[0] if parts else ""
            for name in names:
                if not name.startswith(prefix + "/"):
                    continue
                rel = name[len(prefix) + 1 :].rstrip("/")
                if not rel:
                    continue
                dest = FFMPEG_DIR / rel.replace("/", os.sep)
                if name.endswith("/"):
                    dest.mkdir(parents=True, exist_ok=True)
                else:
                    dest.parent.mkdir(parents=True, exist_ok=True)
                    with zf.open(name) as src, open(dest, "wb") as dst:
                        dst.write(src.read())

        exe = get_ffmpeg_exe()
        if exe:
            print(f"ffmpeg 已安装: {exe}")
            # 验证
            out = subprocess.run([str(exe), "-version"], capture_output=True, text=True)
            if out.returncode == 0:
                print("验证通过")
        else:
            print("解压后未找到 ffmpeg.exe，请检查 zip 结构")
    finally:
        zip_path.unlink(missing_ok=True)

    return get_ffmpeg_exe() is not None


if __name__ == "__main__":
    ok = setup_ffmpeg()
    sys.exit(0 if ok else 1)
