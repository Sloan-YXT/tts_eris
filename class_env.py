"""
情绪分类流水线依赖安装与验证脚本。

已安装（首次运行时记录）：
  google-genai==1.68.0   Gemini API
  funasr==1.3.1          SenseVoice 推理（本地备用）
  modelscope==1.35.1     SenseVoice 模型下载
  httpx==0.28.1          异步 HTTP（DeepSeek 路由）
  openai==*              DeepSeek 兼容客户端（若未装则安装）

用法：
  python class_env.py
"""
from __future__ import annotations

import importlib
import subprocess
import sys
from pathlib import Path

PYTHON = sys.executable
ROOT = Path(__file__).resolve().parent

# ── 需要的包：(import名, pip包名, 最低版本要求或None) ─────────────────────────
REQUIREMENTS = [
    ("google.genai",  "google-genai>=1.0.0",  "1.0.0"),
    ("funasr",        "funasr>=1.1.3",         "1.1.3"),
    ("modelscope",    "modelscope>=1.9.0",     "1.9.0"),
    ("httpx",         "httpx>=0.24.0",         "0.24.0"),
    ("openai",        "openai>=1.0.0",         "1.0.0"),
]


def pip_install(pkg: str) -> bool:
    result = subprocess.run(
        [PYTHON, "-m", "pip", "install", pkg, "-q"],
        capture_output=True, text=True,
    )
    return result.returncode == 0


def check_import(module: str) -> bool:
    try:
        importlib.import_module(module)
        return True
    except ImportError:
        return False


def get_version(module: str) -> str:
    try:
        mod = importlib.import_module(module)
        return getattr(mod, "__version__", "unknown")
    except Exception:
        return "unknown"


def main() -> None:
    print("=" * 55)
    print("  情绪分类流水线依赖检查 & 安装")
    print("=" * 55)

    all_ok = True

    for import_name, pip_spec, min_ver in REQUIREMENTS:
        if check_import(import_name):
            ver = get_version(import_name)
            print(f"  OK   {import_name} ({ver})")
        else:
            print(f"  缺失 {import_name}，正在安装 {pip_spec} ...")
            if pip_install(pip_spec):
                if check_import(import_name):
                    ver = get_version(import_name)
                    print(f"  OK   {import_name} ({ver}) 安装成功")
                else:
                    print(f"  FAIL {import_name} 安装后仍无法导入")
                    all_ok = False
            else:
                print(f"  FAIL {import_name} 安装失败")
                all_ok = False

    print()
    print("=" * 55)
    print("  运行时验证")
    print("=" * 55)

    # 验证 google.genai 客户端可实例化
    try:
        from google import genai
        _ = genai.Client.__init__  # 确认类存在
        print("  OK   google.genai.Client 可用")
    except Exception as e:
        print(f"  FAIL google.genai: {e}")
        all_ok = False

    # 验证 openai 兼容客户端（DeepSeek 用）
    try:
        from openai import OpenAI
        _ = OpenAI.__init__
        print("  OK   openai.OpenAI 可用（DeepSeek 兼容客户端）")
    except Exception as e:
        print(f"  FAIL openai: {e}")
        all_ok = False

    # 验证 httpx
    try:
        import httpx
        _ = httpx.Client()
        print("  OK   httpx.Client 可用")
    except Exception as e:
        print(f"  FAIL httpx: {e}")
        all_ok = False

    # 验证 funasr AutoModel 可导入
    try:
        from funasr import AutoModel
        _ = AutoModel
        print("  OK   funasr.AutoModel 可导入")
    except Exception as e:
        print(f"  FAIL funasr.AutoModel: {e}")
        all_ok = False

    # 验证 SenseVoice 目录存在
    sv_dir = ROOT / "SenseVoice" / "model.py"
    if sv_dir.exists():
        print(f"  OK   SenseVoice/model.py 存在")
    else:
        print(f"  WARN SenseVoice/model.py 不存在（本地推理将不可用）")
        # 不算失败，SenseVoice 是可选的

    print()
    if all_ok:
        print("  全部通过，环境就绪。")
        sys.exit(0)
    else:
        print("  存在失败项，请查看上方日志。")
        sys.exit(1)


if __name__ == "__main__":
    main()
