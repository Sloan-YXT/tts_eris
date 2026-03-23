"""
Step 3: Style-BERT-VITS2 预处理流水线。
  3a. 重采样到 44100Hz + 响度归一化 + 去静音
  3b. 文本预处理（生成 train.list / val.list）
  3c. 生成 BERT 特征
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SBV2_DIR = ROOT / "Style-BERT-VITS2"
PYTHON = sys.executable
MODEL_NAME = "eris"
DATA_DIR = SBV2_DIR / "Data" / MODEL_NAME


def run(args: list[str], label: str) -> None:
    print(f"\n{'='*60}")
    print(f"  {label}")
    print(f"{'='*60}\n")
    result = subprocess.run(args, cwd=str(SBV2_DIR))
    if result.returncode != 0:
        print(f"\n[FAIL] {label}")
        sys.exit(1)
    print(f"\n[OK] {label}")


def create_config() -> None:
    """从 JP-Extra 模板创建训练配置。"""
    template = SBV2_DIR / "configs" / "config_jp_extra.json"
    dst = DATA_DIR / "config.json"

    config = json.loads(template.read_text(encoding="utf-8"))

    config["data"]["training_files"] = f"Data/{MODEL_NAME}/train.list"
    config["data"]["validation_files"] = f"Data/{MODEL_NAME}/val.list"
    config["data"]["use_jp_extra"] = True
    config["data"]["n_speakers"] = 1
    config["data"]["spk2id"] = {MODEL_NAME: 0}

    config["train"]["batch_size"] = 4
    config["train"]["epochs"] = 100
    config["train"]["eval_interval"] = 500
    config["train"]["bf16_run"] = True

    dst.write_text(json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"训练配置已创建: {dst}")


def main() -> None:
    # 0. 创建配置
    create_config()

    # 3a. 重采样
    run([
        PYTHON, "resample.py",
        "-i", f"Data/{MODEL_NAME}/raw",
        "-o", f"Data/{MODEL_NAME}/wavs",
        "--sr", "44100",
        "--num_processes", "8",
        "--normalize",
        "--trim",
    ], "3a. 重采样 + 归一化 + 去静音")

    # 3b. 文本预处理
    run([
        PYTHON, "preprocess_text.py",
        "--config-path", f"Data/{MODEL_NAME}/config.json",
        "--transcription-path", f"Data/{MODEL_NAME}/esd.list",
        "--train-path", f"Data/{MODEL_NAME}/train.list",
        "--val-path", f"Data/{MODEL_NAME}/val.list",
        "--val-per-lang", "3",
        "--yomi_error", "skip",
        "--use_jp_extra",
    ], "3b. 文本预处理（phoneme + train/val split）")

    # 3c. BERT 特征
    run([
        PYTHON, "bert_gen.py",
        "--config-path", f"Data/{MODEL_NAME}/config.json",
        "--num-processes", "1",
        "--device", "cuda",
    ], "3c. 生成 BERT 特征")

    print(f"\n{'='*60}")
    print(f"  预处理全部完成!")
    print(f"{'='*60}")

    train_list = DATA_DIR / "train.list"
    val_list = DATA_DIR / "val.list"
    if train_list.exists():
        n_train = len(train_list.read_text(encoding="utf-8").strip().splitlines())
        n_val = len(val_list.read_text(encoding="utf-8").strip().splitlines())
        print(f"  训练集: {n_train} 条")
        print(f"  验证集: {n_val} 条")

    print(f"\n下一步：运行 sbv2_step4_train.py 开始训练")


if __name__ == "__main__":
    main()
