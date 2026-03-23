"""
一键安装：创建 .venv → 安装依赖 → 下载预训练模型。

用法：python install.py
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SBV2_DIR = ROOT / "Style-BERT-VITS2"
VENV_PY = ROOT / ".venv" / "Scripts" / "python.exe"


def main() -> None:
    # 1. 克隆 SBV2
    if not SBV2_DIR.exists():
        print("克隆 Style-BERT-VITS2...")
        subprocess.run(
            ["git", "clone", "https://github.com/litagin02/Style-BERT-VITS2.git"],
            cwd=str(ROOT), check=True,
        )
    else:
        print("Style-BERT-VITS2 已存在，跳过克隆")

    # 2. 创建 .venv + 安装基础依赖
    print("\n" + "=" * 60)
    print("  创建虚拟环境 + 安装基础依赖")
    print("=" * 60 + "\n")
    subprocess.run([sys.executable, str(ROOT / "setup_env.py")], check=True)

    # 3. 安装 SBV2 依赖 + 预训练模型
    print("\n" + "=" * 60)
    print("  安装 SBV2 依赖 + 下载预训练模型")
    print("=" * 60 + "\n")
    subprocess.run([str(VENV_PY), str(ROOT / "sbv2_install.py")], check=True)

    print("\n" + "=" * 60)
    print("  安装完成!")
    print("=" * 60)
    print("\n下一步: python train_all.py")


if __name__ == "__main__":
    main()
