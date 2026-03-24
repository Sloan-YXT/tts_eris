"""
多风格训练 Step 2: 重新预处理。

由于 wav 文件路径改变（从 wavs/ 到 wavs/<emotion>/），
需要重新运行 preprocess_text.py 生成新的 train.list 和 val.list，
然后运行 style_gen.py 为所有 wav 生成 style vector (.wav.npy)。
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SBV2_DIR = ROOT / "Style-BERT-VITS2"
DATA_DIR = SBV2_DIR / "Data" / "eris"
PYTHON = sys.executable
MODEL_NAME = "eris"


def main() -> None:
    print(f"{'='*60}")
    print(f"  多风格训练 Step 2: 重新预处理文本")
    print(f"{'='*60}")

    # 确认 esd.list 已更新
    esd_path = DATA_DIR / "esd.list"
    lines = esd_path.read_text(encoding="utf-8").strip().splitlines()
    print(f"\nesd.list: {len(lines)} 条")
    if not lines:
        print("[FAIL] esd.list 为空")
        sys.exit(1)
    print(f"示例: {lines[0][:80]}...")

    # 运行 preprocess_text.py (从 SBV2_DIR 运行)
    print(f"\n运行 preprocess_text.py...")
    cmd = [
        PYTHON, "preprocess_text.py",
        "--config-path", f"Data/{MODEL_NAME}/config.json",
        "--transcription-path", f"Data/{MODEL_NAME}/esd.list",
        "--train-path", f"Data/{MODEL_NAME}/train.list",
        "--val-path", f"Data/{MODEL_NAME}/val.list",
        "--val-per-lang", "3",
        "--yomi_error", "skip",
        "--use_jp_extra",
    ]
    result = subprocess.run(cmd, cwd=str(SBV2_DIR))

    if result.returncode != 0:
        print(f"\n[FAIL] preprocess_text.py 失败 (code={result.returncode})")
        sys.exit(1)

    # 验证输出
    train_list = DATA_DIR / "train.list"
    val_list = DATA_DIR / "val.list"

    train_n = len(train_list.read_text(encoding="utf-8").strip().splitlines()) if train_list.exists() else 0
    val_n = len(val_list.read_text(encoding="utf-8").strip().splitlines()) if val_list.exists() else 0

    print(f"\n结果:")
    print(f"  train.list: {train_n} 条")
    print(f"  val.list:   {val_n} 条")

    if train_n == 0:
        print("[FAIL] train.list 为空，无法继续训练")
        sys.exit(1)

    if train_list.exists() and train_n > 0:
        sample = train_list.read_text(encoding="utf-8").splitlines()[0]
        print(f"  示例: {sample[:100]}...")

    # 生成 style vector (.wav.npy)
    print(f"\n运行 style_gen.py...")
    cmd_style = [
        PYTHON, "style_gen.py",
        "--config", f"Data/{MODEL_NAME}/config.json",
        "--num_processes", "1",
    ]
    result_style = subprocess.run(cmd_style, cwd=str(SBV2_DIR))
    if result_style.returncode != 0:
        print(f"\n[FAIL] style_gen.py 失败 (code={result_style.returncode})")
        sys.exit(1)

    # 验证 npy 生成
    wavs_dir = DATA_DIR / "wavs"
    npy_count = len(list(wavs_dir.rglob("*.wav.npy")))
    print(f"  style vectors: {npy_count} 个 .wav.npy")

    print(f"\n{'='*60}")
    print(f"  Step 2 完成!")
    print(f"  下一步: 运行 sbv2_multistyle_step3_train.py")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
