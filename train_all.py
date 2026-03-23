"""
一键训练：从 voices/ 到多风格模型，全自动。

前提：
  1. 已运行 install.py
  2. voices/eris_avl_*/ 下已有切片音频和台词标注（GPT-SoVITS WebUI 产出）
  3. credentials.txt 中配置了 Gemini API Key（情绪标注用）

用法：python train_all.py
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
VENV_PY = ROOT / ".venv" / "Scripts" / "python.exe"
PYTHON = str(VENV_PY)

STEPS = [
    ("数据准备: voices/ → SBV2 格式",          "sbv2_step2_prepare_data.py"),
    ("预处理: 重采样 + 文本 + BERT 特征",       "sbv2_step3_preprocess.py"),
    ("情绪标注: Gemini 批量分类",               "class_annotate.py"),
    ("多风格准备: 按情绪分组 wav",              "sbv2_multistyle_step1_prep.py"),
    ("多风格预处理: 重新生成 train/val",         "sbv2_multistyle_step2_preprocess.py"),
    ("多风格训练: 100 epochs",                  "sbv2_multistyle_step3_train.py"),
]


def main() -> None:
    if not VENV_PY.exists():
        print("错误: .venv 不存在，请先运行 python install.py")
        sys.exit(1)

    total = len(STEPS)
    for i, (label, script) in enumerate(STEPS, 1):
        print(f"\n{'=' * 60}")
        print(f"  [{i}/{total}] {label}")
        print(f"  脚本: {script}")
        print(f"{'=' * 60}\n")

        result = subprocess.run([PYTHON, str(ROOT / script)])
        if result.returncode != 0:
            print(f"\n[FAIL] Step {i} 失败: {script}")
            sys.exit(1)
        print(f"\n[OK] Step {i} 完成")

    print(f"\n{'=' * 60}")
    print(f"  全部训练完成!")
    print(f"{'=' * 60}")
    print(f"\n启动服务: python run.py")
    print(f"请求示例: curl \"http://localhost:8010/tts?text=こんにちは&style=happy\" -o out.wav")


if __name__ == "__main__":
    main()
