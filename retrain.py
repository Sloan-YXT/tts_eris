"""
重新训练脚本：清除旧预处理缓存，然后启动完整训练流程。

用法：
    python retrain.py
"""
import os
import shutil
import subprocess
import sys
from pathlib import Path

ROOT   = Path(__file__).resolve().parent
GSV    = ROOT / "GPT-SoVITS"
LOGS   = GSV / "logs" / "eris"
PYTHON = ROOT / ".venv" / "Scripts" / "python.exe"

TO_DELETE_FILES = [
    LOGS / "2-name2text.txt",
    LOGS / "6-name2semantic.tsv",
]
TO_DELETE_DIRS = [
    LOGS / "5-wav32k",
    LOGS / "7-SV",
]


def main():
    print("=" * 55)
    print("  清除旧预处理缓存")
    print("=" * 55)

    for f in TO_DELETE_FILES:
        if f.exists():
            f.unlink()
            print(f"  删除  {f.relative_to(GSV)}")
        else:
            print(f"  跳过  {f.relative_to(GSV)}（不存在）")

    for d in TO_DELETE_DIRS:
        if d.exists():
            shutil.rmtree(d)
            print(f"  删除  {d.relative_to(GSV)}/")
        else:
            print(f"  跳过  {d.relative_to(GSV)}/（不存在）")

    print("\n启动训练...\n")

    env = os.environ.copy()
    env["PYTHONIOENCODING"] = "utf-8"
    env["PYTHONUTF8"] = "1"

    result = subprocess.run(
        [str(PYTHON), str(ROOT / "train.py")],
        env=env,
    )
    sys.exit(result.returncode)


if __name__ == "__main__":
    main()
