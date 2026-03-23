#!/usr/bin/env python3
"""
启动 TTS 服务。

用法：
  python run.py          # SBV2 多风格 TTS (端口 8010)
  python run.py gsv      # GPT-SoVITS TTS (端口 8010，需先启动 api_v2.py)
"""
import sys
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent
_venv_py = ROOT / ".venv" / "Scripts" / "python.exe"
if _venv_py.exists() and Path(sys.prefix).resolve() != _venv_py.parent.parent.resolve():
    os.execv(str(_venv_py), [str(_venv_py), __file__] + sys.argv[1:])

import uvicorn

if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "sbv2"

    if mode == "gsv":
        print("GPT-SoVITS 模式 (端口 8010)")
        print("请确认 GPT-SoVITS/api_v2.py 已在端口 9880 运行\n")
        uvicorn.run("src.main:app", host="0.0.0.0", port=8010, reload=False)
    else:
        uvicorn.run("run_sbv2:app", host="0.0.0.0", port=8010, reload=False)
