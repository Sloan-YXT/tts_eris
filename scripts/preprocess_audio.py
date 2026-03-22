#!/usr/bin/env python3
"""参考音频预处理：静音切除、音量标准化。"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

# 添加项目根目录到 path
_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_root))

from src.audio_preprocess import preprocess_reference


def main():
    parser = argparse.ArgumentParser(description="预处理参考音频")
    parser.add_argument("input", type=Path, help="输入音频路径")
    parser.add_argument("-o", "--output", type=Path, default=None, help="输出路径（默认覆盖输入）")
    parser.add_argument("--top-db", type=float, default=25.0, help="静音切除阈值 dB")
    parser.add_argument("--target-db", type=float, default=-20.0, help="目标音量 dB")
    args = parser.parse_args()

    inp = args.input
    if not inp.exists():
        print(f"错误: 文件不存在 {inp}")
        return 1

    out = args.output or inp
    preprocess_reference(
        inp,
        output_path=out,
        top_db=args.top_db,
        target_db=args.target_db,
    )
    print(f"已保存: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
