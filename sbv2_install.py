"""
Style-BERT-VITS2 一键安装脚本（依赖 + 预训练模型）。

已处理的已知问题：
  - torch/torchaudio：venv 已有 2.10+cu128，跳过旧版限制
  - faster-whisper==0.10.1 要求 av==10.*：改装最新版
  - pyopenjtalk-dict / onnxruntime-directml：Windows 文件锁问题，
    安装前先检查是否可用，可用则跳过，避免覆盖报错

用法（首次）：
  python sbv2_install.py

重复运行安全，已安装的包和已下载的模型均会跳过。
"""
from __future__ import annotations

import importlib
import subprocess
import sys
from pathlib import Path

PYTHON = sys.executable
ROOT = Path(__file__).resolve().parent
SBV2_DIR = ROOT / "Style-BERT-VITS2"
REQ_FILE = SBV2_DIR / "requirements.txt"

# 跳过：已有更新版本
SKIP_PREFIXES = [
    "torch<",
    "torchaudio<",
    "faster-whisper",       # 单独处理
    "onnxruntime-directml", # 单独处理
    "pyopenjtalk-dict",     # 单独处理
]

# 需要检查可导入性后再决定是否安装的包
CAREFUL_PKGS: dict[str, str] = {
    # import名 -> pip包名
    "onnxruntime": "onnxruntime-directml",
    "pyopenjtalk": "pyopenjtalk-dict",
}


def _run_pip(*args: str, capture: bool = False) -> tuple[int, str]:
    cmd = [PYTHON, "-m", "pip", "install", *args]
    result = subprocess.run(cmd, text=True, capture_output=capture)
    return result.returncode, (result.stderr or "") if capture else ""


def can_import(module: str) -> bool:
    try:
        importlib.import_module(module)
        return True
    except ImportError:
        return False


def install_pkg(spec: str) -> bool:
    """安装单个包，返回是否成功。"""
    code, err = _run_pip(spec, "--only-binary", "av", capture=True)
    if code != 0:
        # 如果是 WinError 5（文件锁），检查包是否其实已可用
        if "WinError 5" in err or "Access is denied" in err:
            # 取包名（去掉版本约束）
            import re
            mod = re.split(r"[<>=!;]", spec)[0].strip().replace("-", "_")
            if can_import(mod):
                print(f"  [SKIP] {spec}（文件锁，但包已可用）")
                return True
        print(f"  [FAIL] {spec}")
        print(err[-400:])
        return False
    return True


# ── 主流程 ────────────────────────────────────────────────────────────────────

def step_deps() -> None:
    print("\n" + "=" * 60)
    print("  Step 1/2  安装依赖")
    print("=" * 60)

    # 升级 pip（静默）
    subprocess.run([PYTHON, "-m", "pip", "install", "--upgrade", "pip", "-q"],
                   check=True)

    # 解析 requirements.txt，过滤冲突项
    lines = REQ_FILE.read_text(encoding="utf-8").splitlines()
    to_install: list[str] = []
    for line in lines:
        s = line.strip()
        if not s or s.startswith("#"):
            continue
        if any(s.startswith(p) for p in SKIP_PREFIXES):
            continue
        to_install.append(s)

    # 批量安装
    print(f"\n安装 {len(to_install)} 个依赖包...")
    code, err = _run_pip(*to_install, "--only-binary", "av", capture=True)
    if code != 0:
        # 批量失败则逐个安装，精确定位问题
        print("批量安装失败，逐个尝试...\n")
        failed: list[str] = []
        for pkg in to_install:
            ok = install_pkg(pkg)
            if not ok:
                failed.append(pkg)
            else:
                print(f"  [OK]   {pkg}")
        if failed:
            print(f"\n以下包安装失败，请手动处理：")
            for f in failed:
                print(f"  {f}")
            sys.exit(1)
    else:
        print("[OK] 批量安装完成")

    # 处理文件锁敏感包：可用则跳过，否则安装
    for mod_name, pip_name in CAREFUL_PKGS.items():
        if can_import(mod_name):
            print(f"  [SKIP] {pip_name}（已可用）")
        else:
            print(f"  安装 {pip_name}...")
            if not install_pkg(pip_name):
                print(f"  [WARN] {pip_name} 安装失败，继续...")

    # 安装最新版 faster-whisper（兼容 av 17.x）
    print("  安装 faster-whisper（最新版）...")
    code, _ = _run_pip("faster-whisper", capture=True)
    print(f"  {'[OK]' if code == 0 else '[FAIL]'} faster-whisper")

    # 验证
    print("\n验证关键依赖：")
    checks = ["torch", "torchaudio", "gradio", "transformers",
              "faster_whisper", "pyopenjtalk", "librosa"]
    failed_checks = []
    for pkg in checks:
        try:
            mod = importlib.import_module(pkg)
            ver = getattr(mod, "__version__", "?")
            print(f"  OK   {pkg} ({ver})")
        except ImportError:
            print(f"  FAIL {pkg}")
            failed_checks.append(pkg)

    import torch
    if torch.cuda.is_available():
        print(f"  OK   CUDA ({torch.cuda.get_device_name(0)}, torch {torch.__version__})")
    else:
        print("  WARN CUDA 不可用，将使用 CPU")

    if failed_checks:
        print(f"\n依赖验证失败：{failed_checks}")
        sys.exit(1)


def step_models() -> None:
    print("\n" + "=" * 60)
    print("  Step 2/2  下载预训练模型")
    print("=" * 60)

    import os
    os.chdir(SBV2_DIR)
    sys.path.insert(0, str(SBV2_DIR))

    import json, shutil
    import yaml
    from huggingface_hub import hf_hub_download

    def dl(repo_id: str, filename: str, local_dir: str) -> None:
        dest = Path(local_dir) / filename
        if dest.exists():
            print(f"  SKIP {filename}")
            return
        dest.parent.mkdir(parents=True, exist_ok=True)
        print(f"  DOWN {filename}")
        hf_hub_download(repo_id, filename, local_dir=local_dir)

    # BERT 模型
    print("\nBERT 模型：")
    with open("bert/bert_models.json", encoding="utf-8") as f:
        bert_models = json.load(f)
    for k, v in bert_models.items():
        for file in v["files"]:
            dl(v["repo_id"], file, f"bert/{k}")

    # 默认示例音色（jvnv + koharune-ami + amitaro）
    print("\n默认音色模型：")
    jvnv_files = [
        "jvnv-F1-jp/config.json",
        "jvnv-F1-jp/jvnv-F1-jp_e160_s14000.safetensors",
        "jvnv-F1-jp/style_vectors.npy",
        "jvnv-F2-jp/config.json",
        "jvnv-F2-jp/jvnv-F2_e166_s20000.safetensors",
        "jvnv-F2-jp/style_vectors.npy",
        "jvnv-M1-jp/config.json",
        "jvnv-M1-jp/jvnv-M1-jp_e158_s14000.safetensors",
        "jvnv-M1-jp/style_vectors.npy",
        "jvnv-M2-jp/config.json",
        "jvnv-M2-jp/jvnv-M2-jp_e159_s17000.safetensors",
        "jvnv-M2-jp/style_vectors.npy",
    ]
    for f in jvnv_files:
        dl("litagin/style_bert_vits2_jvnv", f, "model_assets")
    for f in ["config.json", "style_vectors.npy", "koharune-ami.safetensors"]:
        dl("litagin/sbv2_koharune_ami", f"koharune-ami/{f}", "model_assets")
    for f in ["config.json", "style_vectors.npy", "amitaro.safetensors"]:
        dl("litagin/sbv2_amitaro", f"amitaro/{f}", "model_assets")

    # SLM 模型（微调用）
    print("\nSLM 模型：")
    dl("microsoft/wavlm-base-plus", "pytorch_model.bin", "slm/wavlm-base-plus")

    # 预训练底模（微调用）
    print("\n预训练底模：")
    for f in ["G_0.safetensors", "D_0.safetensors", "DUR_0.safetensors"]:
        dl("litagin/Style-Bert-VITS2-1.0-base", f, "pretrained")

    # JP-Extra 底模（日语专用，推荐）
    print("\nJP-Extra 底模：")
    for f in ["G_0.safetensors", "D_0.safetensors", "WD_0.safetensors"]:
        dl("litagin/Style-Bert-VITS2-2.0-base-JP-Extra", f, "pretrained_jp_extra")

    # configs/paths.yml
    paths_yml = Path("configs/paths.yml")
    if not paths_yml.exists():
        shutil.copy("configs/default_paths.yml", paths_yml)
        print("\nconfigs/paths.yml 已创建")


def main() -> None:
    if not SBV2_DIR.exists():
        print(f"找不到 Style-BERT-VITS2 目录：{SBV2_DIR}")
        print("请先执行：git clone https://github.com/litagin02/Style-BERT-VITS2.git")
        sys.exit(1)

    step_deps()
    step_models()

    print("\n" + "=" * 60)
    print("  全部完成！")
    print("=" * 60)
    print(f"\n启动 WebUI：")
    print(f"  cd Style-BERT-VITS2 && python app.py")
    print(f"启动推理编辑器：")
    print(f"  cd Style-BERT-VITS2 && python server_editor.py --inbrowser")


if __name__ == "__main__":
    main()
