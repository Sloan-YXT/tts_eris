#!/usr/bin/env python3
"""
WAV 剪辑：按时间范围截取音频片段。

用法：
  python scripts/clip_wav.py 输入.wav -s 1 -e 10 -o 输出.wav
  python scripts/clip_wav.py 输入.wav --start 1 --end 10 -o 输出.wav
"""
from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
_FFMPEG_BIN = _ROOT / "ffmpeg" / "bin"
if _FFMPEG_BIN.exists():
    os.environ["PATH"] = str(_FFMPEG_BIN) + os.pathsep + os.environ.get("PATH", "")

WAV_SAMPLE_RATE = 24000
WAV_CHANNELS = 1


def get_ffmpeg_path() -> str:
    local_exe = _ROOT / "ffmpeg" / "bin" / "ffmpeg.exe"
    return str(local_exe) if local_exe.exists() else "ffmpeg"


def clip_wav(
    input_path: Path,
    output_path: Path,
    start_sec: float,
    end_sec: float,
) -> bool:
    """
    截取 WAV 的 [start_sec, end_sec] 区间。
    输出 24kHz、16bit、单声道（符合 GPT-SoVITS）。
    """
    duration = end_sec - start_sec
    if duration <= 0:
        return False
    cmd = [
        get_ffmpeg_path(), "-y",
        "-i", str(input_path),
        "-ss", str(start_sec),
        "-t", str(duration),
        "-vn", "-ar", str(WAV_SAMPLE_RATE), "-ac", str(WAV_CHANNELS),
        "-acodec", "pcm_s16le",
        str(output_path),
    ]
    out = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace")
    if out.returncode != 0:
        if out.stderr:
            print(out.stderr, file=sys.stderr)
    return out.returncode == 0


def main():
    parser = argparse.ArgumentParser(description="WAV 按时间范围剪辑")
    parser.add_argument("input", type=Path, help="输入 WAV/MP4/MP3 路径")
    parser.add_argument("-s", "--start", type=float, required=True, help="起始时间（秒）")
    parser.add_argument("-e", "--end", type=float, required=True, help="结束时间（秒）")
    parser.add_argument("-o", "--output", type=Path, default=None, help="输出路径（默认 输入名_clip.wav）")
    args = parser.parse_args()

    if not args.input.exists():
        print(f"错误: 文件不存在 {args.input}")
        return 1

    if args.start >= args.end:
        print("错误: start 必须小于 end")
        return 1

    out = args.output or args.input.parent / f"{args.input.stem}_clip.wav"
    out.parent.mkdir(parents=True, exist_ok=True)

    if clip_wav(args.input, out, args.start, args.end):
        print(f"已保存: {out} ({args.start}s - {args.end}s)")
        return 0
    print("剪辑失败（请查看上方 ffmpeg 报错）")
    return 1


if __name__ == "__main__":
    sys.exit(main())
