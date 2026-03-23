"""
一键训练：从 voices/ 到模型，全自动。

前提：
  1. 已运行 install.py
  2. voices/eris_avl_*/ 下已有切片音频和台词标注（GPT-SoVITS WebUI 产出）
  3. 多风格模式需要 credentials.txt 中配置 Gemini API Key（情绪标注用）

用法：
  python train_all.py           # 多风格训练（含 Gemini 情绪标注，需要 credentials.txt）
  python train_all.py --basic   # 单风格训练（跳过情绪标注，无需 Gemini API）
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
VENV_PY = ROOT / ".venv" / "Scripts" / "python.exe"
PYTHON = str(VENV_PY)

STEPS_MULTI = [
    ("数据准备: voices/ → SBV2 格式",          "sbv2_step2_prepare_data.py"),
    ("预处理: 重采样 + 文本 + BERT 特征",       "sbv2_step3_preprocess.py"),
    ("情绪标注: Gemini 批量分类",               "class_annotate.py"),
    ("多风格准备: 按情绪分组 wav",              "sbv2_multistyle_step1_prep.py"),
    ("多风格预处理: 重新生成 train/val",         "sbv2_multistyle_step2_preprocess.py"),
    ("多风格训练: 100 epochs",                  "sbv2_multistyle_step3_train.py"),
]

STEPS_BASIC = [
    ("数据准备: voices/ → SBV2 格式",          "sbv2_step2_prepare_data.py"),
    ("预处理: 重采样 + 文本 + BERT 特征",       "sbv2_step3_preprocess.py"),
    ("单风格训练",                              "sbv2_step4_train.py"),
]


def main() -> None:
    if not VENV_PY.exists():
        print("错误: .venv 不存在，请先运行 python install.py")
        sys.exit(1)

    basic = "--basic" in sys.argv
    steps = STEPS_BASIC if basic else STEPS_MULTI
    mode = "单风格" if basic else "多风格"
    print(f"训练模式: {mode}")

    total = len(steps)
    for i, (label, script) in enumerate(steps, 1):
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
    print(f"  {mode}训练完成!")
    print(f"{'=' * 60}")
    print(f"\n启动服务: python run.py")


if __name__ == "__main__":
    main()
