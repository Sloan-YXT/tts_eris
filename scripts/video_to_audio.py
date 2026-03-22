#!/usr/bin/env python3
"""
从链接或本地视频提取音频，生成 MP3 和 GPT-SoVITS 所需的 WAV。

WAV 规格：24kHz, 16bit, 单声道（符合 GPT-SoVITS 参考音频要求）
"""
from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path

# 项目根目录
_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))

# 将项目内 ffmpeg 加入 PATH，供 yt-dlp 等子进程使用
_FFMPEG_BIN = _ROOT / "ffmpeg" / "bin"
if _FFMPEG_BIN.exists():
    os.environ["PATH"] = str(_FFMPEG_BIN) + os.pathsep + os.environ.get("PATH", "")

# GPT-SoVITS 参考音频规格
WAV_SAMPLE_RATE = 24000
WAV_CHANNELS = 1
WAV_BIT_DEPTH = 16


def get_ffmpeg_path() -> str:
    """优先使用项目内的 ffmpeg，否则使用系统 PATH。"""
    local_exe = _ROOT / "ffmpeg" / "bin" / "ffmpeg.exe"
    if local_exe.exists():
        return str(local_exe)
    return "ffmpeg"


def check_ffmpeg() -> bool:
    """检查 ffmpeg 是否可用。"""
    path = get_ffmpeg_path()
    if path != "ffmpeg":
        return True
    return shutil.which("ffmpeg") is not None


def check_ytdlp() -> bool:
    """检查 yt-dlp 是否可用。"""
    try:
        import yt_dlp
        return True
    except ImportError:
        return False


def download_from_url(url: str, output_dir: Path) -> Path:
    """从 URL 下载视频，返回本地路径。"""
    from yt_dlp import YoutubeDL

    ffmpeg_path = get_ffmpeg_path()
    opts = {
        "outtmpl": str(output_dir / "%(title)s.%(ext)s"),
        "format": "bestvideo[ext=mp4]+bestaudio[ext=m4a]/best[ext=mp4]/best",
        "merge_output_format": "mp4",
        "quiet": False,
    }
    if ffmpeg_path != "ffmpeg":
        opts["ffmpeg_location"] = ffmpeg_path
    with YoutubeDL(opts) as ydl:
        info = ydl.extract_info(url, download=True)
        filename = ydl.prepare_filename(info)
        return Path(filename)


def extract_mp3(input_path: Path, output_mp3: Path, max_duration: float | None = None) -> tuple[bool, str]:
    """提取 MP3。"""
    cmd = [get_ffmpeg_path(), "-y", "-i", str(input_path), "-vn", "-acodec", "libmp3lame", "-b:a", "128k"]
    if max_duration:
        cmd.extend(["-t", str(max_duration)])
    cmd.append(str(output_mp3))
    out = subprocess.run(cmd, capture_output=True, text=True)
    return out.returncode == 0, out.stderr or ""


def extract_wav(input_path: Path, output_wav: Path, max_duration: float | None = None) -> tuple[bool, str]:
    """提取 WAV（24kHz, 16bit, mono，符合 GPT-SoVITS）。"""
    cmd = [
        get_ffmpeg_path(), "-y", "-i", str(input_path),
        "-vn", "-ar", str(WAV_SAMPLE_RATE), "-ac", str(WAV_CHANNELS),
        "-acodec", "pcm_s16le",
    ]
    if max_duration:
        cmd.extend(["-t", str(max_duration)])
    cmd.append(str(output_wav))
    out = subprocess.run(cmd, capture_output=True, text=True)
    return out.returncode == 0, out.stderr or ""


def main():
    parser = argparse.ArgumentParser(
        description="从链接或本地视频提取音频，生成 MP3 和 GPT-SoVITS 所需的 WAV",
    )
    parser.add_argument(
        "input",
        help="视频 URL 或本地文件路径",
    )
    parser.add_argument(
        "-o", "--output-dir",
        type=Path,
        default=None,
        help="输出目录（默认与输入同目录）",
    )
    parser.add_argument(
        "--mp3-only",
        action="store_true",
        help="仅生成 MP3",
    )
    parser.add_argument(
        "--wav-only",
        action="store_true",
        help="仅生成 WAV",
    )
    parser.add_argument(
        "--max-duration",
        type=float,
        default=None,
        help="截取前 N 秒（用于长视频裁剪为参考音频）",
    )
    args = parser.parse_args()

    if not check_ffmpeg():
        print("错误: 未找到 ffmpeg")
        print("  运行: python scripts/setup_ffmpeg.py  (下载到项目目录)")
        print("  或系统安装: winget install ffmpeg")
        return 1

    input_path: Path | None = None
    is_url = args.input.startswith(("http://", "https://"))

    if is_url:
        if not check_ytdlp():
            print("错误: 请先安装 yt-dlp: pip install yt-dlp")
            return 1
        out_dir = args.output_dir or Path.cwd()
        out_dir.mkdir(parents=True, exist_ok=True)
        print(f"正在下载: {args.input}")
        try:
            input_path = download_from_url(args.input, out_dir)
        except Exception as e:
            print(f"下载失败: {e}")
            return 1
    else:
        input_path = Path(args.input)
        if not input_path.exists():
            print(f"错误: 文件不存在 {input_path}")
            return 1
        out_dir = args.output_dir or input_path.parent

    out_dir.mkdir(parents=True, exist_ok=True)
    stem = input_path.stem

    output_mp3 = out_dir / f"{stem}.mp3"
    output_wav = out_dir / f"{stem}.wav"

    gen_mp3 = not args.wav_only
    gen_wav = not args.mp3_only

    if gen_mp3:
        ok, err = extract_mp3(input_path, output_mp3, args.max_duration)
        if not ok:
            print(f"MP3 转换失败: {err}")
            return 1
        print(f"MP3: {output_mp3}")

    if gen_wav:
        ok, err = extract_wav(input_path, output_wav, args.max_duration)
        if not ok:
            print(f"WAV 转换失败: {err}")
            return 1
        print(f"WAV: {output_wav} ({WAV_SAMPLE_RATE}Hz, {WAV_BIT_DEPTH}bit)")

    print("完成")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
