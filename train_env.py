"""
训练环境准备：下载 GPT-SoVITS 微调训练所需的全部资源。

五步流程各自需要的资源：
  步骤 1  视频转音频        ffmpeg（setup_env.py 已安装）
  步骤 2  UVR5 人声分离     HP5_only_main_vocal.pth + VR-DeEchoAggressive.pth
  步骤 3  自动切片          GPT-SoVITS 内置，无需额外下载
  步骤 4  Faster-Whisper ASR  faster-whisper-large-v3 模型
  步骤 5  训练              预训练模型（setup_env.py 已下载）

用法：
    conda activate gpt-sovits
    python train_env.py
"""
import os
import sys
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parent
GSV_DIR = ROOT / "GPT-SoVITS"
UVR5_WEIGHTS = GSV_DIR / "tools" / "uvr5" / "uvr5_weights"
ASR_MODELS = GSV_DIR / "tools" / "asr" / "models"
WHISPER_DIR = ASR_MODELS / "faster-whisper-large-v3"

# UVR5 模型列表：(文件名, HuggingFace repo 内路径)
UVR5_MODELS = [
    "HP5_only_main_vocal.pth",
    "VR-DeEchoAggressive.pth",
]
UVR5_HF_REPO = "Delik/uvr5_weights"

# Faster-Whisper large-v3 文件列表
WHISPER_HF_REPO = "Systran/faster-whisper-large-v3"
WHISPER_MS_REPO = "XXXXRT/faster-whisper"
WHISPER_FILES = [
    "config.json",
    "model.bin",
    "tokenizer.json",
    "vocabulary.json",
    "preprocessor_config.json",
]


def section(title: str):
    print(f"\n{'=' * 55}")
    print(f"  {title}")
    print("=" * 55)


HF_MIRRORS = [
    "https://huggingface.co",
    "https://hf-mirror.com",
]


def check_hf_endpoint() -> str | None:
    """返回可用的 HuggingFace 端点 URL，全部不可达则返回 None。"""
    import requests
    for endpoint in HF_MIRRORS:
        try:
            requests.get(f"{endpoint}/api/models/gpt2", timeout=5)
            return endpoint
        except Exception:
            continue
    return None


def download_uvr5_models(hf_endpoint: str | None):
    section("步骤 2 — UVR5 人声分离模型")
    UVR5_WEIGHTS.mkdir(parents=True, exist_ok=True)

    missing = [m for m in UVR5_MODELS if not (UVR5_WEIGHTS / m).exists()]
    if not missing:
        print("  所有 UVR5 模型已存在，跳过。")
        for m in UVR5_MODELS:
            print(f"  OK  {m}")
        return

    for m in UVR5_MODELS:
        dest = UVR5_WEIGHTS / m
        if dest.exists():
            print(f"  OK  {m}（已存在）")
            continue

        print(f"  下载 {m} ...")
        if hf_endpoint:
            _hf_download_file(UVR5_HF_REPO, m, dest, hf_endpoint)
        else:
            _ms_download_uvr5(m, dest)


def _hf_download_file(repo_id: str, filename: str, dest: Path, endpoint: str):
    import os
    os.environ["HF_ENDPOINT"] = endpoint
    from huggingface_hub import hf_hub_download
    try:
        tmp = hf_hub_download(
            repo_id=repo_id,
            filename=filename,
            local_dir=dest.parent,
            local_dir_use_symlinks=False,
        )
        tmp_path = Path(tmp)
        if tmp_path.resolve() != dest.resolve() and not dest.exists():
            import shutil
            shutil.copy2(tmp_path, dest)
        print(f"  DONE  {dest.name}")
    except Exception as e:
        print(f"  [ERROR] 下载失败 ({endpoint}): {e}")
        sys.exit(1)


def _ms_download_uvr5(filename: str, dest: Path):
    """通过直链下载（hf-mirror 之外的备用方案）。"""
    import urllib.request
    url = f"https://hf-mirror.com/{UVR5_HF_REPO}/resolve/main/{filename}"
    print(f"  直链下载: {url}")
    try:
        urllib.request.urlretrieve(url, dest)
        print(f"  DONE  {dest.name}")
    except Exception as e:
        print(f"  [ERROR] 直链下载失败: {e}")
        print(f"  请手动下载并放至: {dest}")
        sys.exit(1)


def download_whisper_model(hf_endpoint: str | None):
    section("步骤 4 — Faster-Whisper large-v3 ASR 模型")

    all_exist = all((WHISPER_DIR / f).exists() for f in WHISPER_FILES)
    if all_exist:
        print("  Faster-Whisper large-v3 已存在，跳过。")
        print(f"  位置: {WHISPER_DIR}")
        return

    WHISPER_DIR.mkdir(parents=True, exist_ok=True)

    if hf_endpoint:
        import os
        os.environ["HF_ENDPOINT"] = hf_endpoint
        print(f"  从 {hf_endpoint} 下载: {WHISPER_HF_REPO}")
        print(f"  目标目录: {WHISPER_DIR}")
        try:
            from huggingface_hub import snapshot_download
            snapshot_download(
                repo_id=WHISPER_HF_REPO,
                local_dir=str(WHISPER_DIR),
                local_dir_use_symlinks=False,
                allow_patterns=WHISPER_FILES,
            )
            print("  DONE  faster-whisper-large-v3")
        except Exception as e:
            print(f"  [ERROR] 下载失败: {e}")
            sys.exit(1)
    else:
        print(f"  从 ModelScope 下载: {WHISPER_MS_REPO}")
        print(f"  目标目录: {ASR_MODELS}")
        try:
            from modelscope import snapshot_download
            ms_patterns = [f"faster-whisper-large-v3/{f}" for f in WHISPER_FILES]
            snapshot_download(
                model_id=WHISPER_MS_REPO,
                local_dir=str(ASR_MODELS),
                allow_file_pattern=ms_patterns,
            )
            print("  DONE  faster-whisper-large-v3（ModelScope）")
        except Exception as e:
            print(f"  [ERROR] ModelScope 下载失败: {e}")
            sys.exit(1)


# WebUI 运行时依赖：(pip 包名, import 名)
WEBUI_PKGS = [
    ("psutil",          "psutil"),
    ("tensorboard",     "tensorboard"),
    ("pytorch_lightning", "pytorch_lightning"),
    ("peft",            "peft"),
    ("pandas",          "pandas"),
    ("ffmpeg-python",   "ffmpeg"),   # import ffmpeg
]


def _find_base_conda() -> Path | None:
    """从 conda env 路径推断 base conda 根目录。
    例：D:/miniconda3/envs/gpt-sovits → D:/miniconda3"""
    conda_root = _find_conda_env_root()
    if conda_root:
        # 标准结构：base/envs/env_name
        envs_dir = conda_root.parent
        if envs_dir.name == "envs":
            base = envs_dir.parent
            if (base / "Library" / "bin").exists():
                return base
    # 也可能直接就是 base
    base_candidates = [Path.home() / "miniconda3", Path.home() / "anaconda3",
                       Path("D:/miniconda3"), Path("C:/miniconda3")]
    for c in base_candidates:
        if (c / "Library" / "bin" / "liblzma.dll").exists():
            return c
    return None


def _find_conda_env_root() -> Path | None:
    """从 pyvenv.cfg 的 home 字段推断 conda env 根目录。
    支持两种情况：直接 conda env 或基于 conda env 的 venv。"""
    # 当前 prefix 本身可能就是 conda env
    conda_lib = Path(sys.prefix) / "Library" / "bin"
    if conda_lib.exists():
        return Path(sys.prefix)
    # 否则读 pyvenv.cfg 找 home
    cfg = Path(sys.prefix) / "pyvenv.cfg"
    if cfg.exists():
        for line in cfg.read_text(encoding="utf-8").splitlines():
            if line.startswith("home"):
                home = Path(line.split("=", 1)[1].strip())
                # Windows conda: python.exe 在 env 根目录，home 就是 env root
                # 普通 venv: home 指向 Scripts/，env root = parent
                for candidate in (home, home.parent):
                    if (candidate / "Library" / "bin").exists():
                        return candidate
    return None


def _find_conda_name() -> str | None:
    """从 pyvenv.cfg 或 CONDA_DEFAULT_ENV 推断 conda env 名称。"""
    env_name = os.environ.get("CONDA_DEFAULT_ENV")
    if env_name:
        return env_name
    conda_root = _find_conda_env_root()
    if conda_root:
        # env root 末段即 env 名（如 gpt-sovits）
        return conda_root.name
    return None


def install_conda_deps():
    """修复系统级 DLL 缺失问题。"""
    section("系统 DLL 修复")
    import shutil

    conda_root = _find_conda_env_root()
    if not conda_root:
        print("  [WARN] 未找到 conda env 根目录，跳过")
        return

    # _lzma.pyd 所在目录——DLL 放在这里 Windows 一定能找到
    dll_dest_dir = conda_root / "DLLs"
    dll_dest_dir.mkdir(exist_ok=True)

    # 需要从 base conda 复制的 DLL 列表
    base = _find_base_conda()
    base_lib_bin = base / "Library" / "bin" if base else None

    dlls_to_copy = ["liblzma.dll"]  # 可按需扩展

    for dll_name in dlls_to_copy:
        dest = dll_dest_dir / dll_name
        if dest.exists():
            print(f"  OK    {dll_name}")
            continue

        # 先在 env 自己的 Library/bin 找
        src = conda_root / "Library" / "bin" / dll_name
        if not src.exists() and base_lib_bin:
            src = base_lib_bin / dll_name
        if not src.exists():
            print(f"  [WARN] 找不到 {dll_name}，跳过（可能不影响运行）")
            continue

        shutil.copy2(src, dest)
        print(f"  DONE  {dll_name}  ({src} → {dest})")


def patch_sitecustomize():
    """把 conda Library/bin 和本地 ffmpeg/bin 同时加入 DLL 路径和 PATH。"""
    section("修复 ffmpeg PATH")

    # site-packages 写入目标（venv 或 conda env 均适用）
    site_pkgs = Path(sys.prefix) / "Lib" / "site-packages"

    ffmpeg_dirs = []
    # 本地项目 ffmpeg
    local_ffmpeg = ROOT / "ffmpeg" / "bin"
    if local_ffmpeg.exists():
        ffmpeg_dirs.append(str(local_ffmpeg))
    # conda env Library/bin
    conda_root = _find_conda_env_root()
    if conda_root:
        ffmpeg_dirs.append(str(conda_root / "Library" / "bin"))
    else:
        print("  [WARN] 未找到 conda env Library/bin")
    # base conda Library/bin（含 liblzma.dll 等系统 DLL）
    base = _find_base_conda()
    if base:
        base_lib = str(base / "Library" / "bin")
        if base_lib not in ffmpeg_dirs:
            ffmpeg_dirs.append(base_lib)

    if not ffmpeg_dirs:
        print("  [WARN] 未找到任何 ffmpeg 目录，跳过")
        return

    content = (
        '"""Add DLL directories and ffmpeg to PATH on startup (Windows fix)."""\n'
        "import os, sys\n"
        "if sys.platform == 'win32':\n"
        f"    _ffmpeg_dirs = {ffmpeg_dirs!r}\n"
        "    for _d in _ffmpeg_dirs:\n"
        "        if os.path.isdir(_d):\n"
        "            if hasattr(os, 'add_dll_directory'):\n"
        "                try: os.add_dll_directory(_d)\n"
        "                except Exception: pass\n"
        "            if _d not in os.environ.get('PATH', ''):\n"
        "                os.environ['PATH'] = _d + os.pathsep + os.environ.get('PATH', '')\n"
    )
    (site_pkgs / "sitecustomize.py").write_text(content, encoding="utf-8")
    for d in ffmpeg_dirs:
        print(f"  OK    {d}")
    print("  sitecustomize.py 已更新，ffmpeg 将对所有子进程可见。")


def install_webui_deps():
    section("WebUI 依赖检查")
    missing_pip = []
    for pip_name, import_name in WEBUI_PKGS:
        r = subprocess.run(
            [sys.executable, "-c", f"import {import_name}"],
            capture_output=True,
        )
        if r.returncode != 0:
            missing_pip.append(pip_name)
            print(f"  MISS  {pip_name}")
        else:
            print(f"  OK    {pip_name}")

    if missing_pip:
        print(f"\n  安装缺失包: {' '.join(missing_pip)}")
        r = subprocess.run(
            [sys.executable, "-m", "pip", "install"] + missing_pip,
        )
        if r.returncode != 0:
            print(f"  [ERROR] 安装失败，请手动运行: pip install {' '.join(missing_pip)}")
            sys.exit(1)
        for pkg in missing_pip:
            print(f"  DONE  {pkg}")
    else:
        print("  所有 WebUI 依赖已就绪。")


def verify():
    section("验证")
    ok = True

    for m in UVR5_MODELS:
        p = UVR5_WEIGHTS / m
        status = "OK  " if p.exists() else "MISS"
        if not p.exists():
            ok = False
        print(f"  [{status}] UVR5: {m}")

    for f in WHISPER_FILES:
        p = WHISPER_DIR / f
        status = "OK  " if p.exists() else "MISS"
        if not p.exists():
            ok = False
        print(f"  [{status}] Whisper: {f}")

    print()
    if ok:
        print("所有训练资源已就绪！")
        print()
        print("下一步：")
        print("  conda activate gpt-sovits")
        print(f"  cd {GSV_DIR}")
        print("  python webui.py")
    else:
        print("[WARN] 部分文件缺失，请检查上方日志。")
    return ok


def main():
    print("=" * 55)
    print("  GPT-SoVITS 训练环境准备")
    print("=" * 55)

    # 检测网络
    print("\n检测网络连通性...")
    hf_endpoint = check_hf_endpoint()
    if hf_endpoint:
        print(f"  下载源: {hf_endpoint}")
    else:
        print("  HuggingFace / hf-mirror 均不可达，切换 ModelScope")

    install_conda_deps()
    patch_sitecustomize()
    install_webui_deps()
    download_uvr5_models(hf_endpoint)
    download_whisper_model(hf_endpoint)
    verify()


if __name__ == "__main__":
    main()
