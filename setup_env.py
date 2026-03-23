"""
一键安装脚本。在项目根目录下用任意 Python 运行即可：

    python setup_env.py

脚本会自动找到 Python 3.10，在项目内创建 .venv，并安装所有依赖。
"""
import subprocess
import sys
import os
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parent
VENV = ROOT / ".venv"
VENV_PY = VENV / "Scripts" / "python.exe"
GSV_DIR = ROOT / "GPT-SoVITS"

# ─────────────────────────────────────────────────────────────────────────────
# Phase 1（Bootstrap）：找 Python 3.10 → 创建 .venv → 用 .venv/python 重启
# 判断当前是否已在 .venv 内：sys.prefix 指向 .venv 目录
# ─────────────────────────────────────────────────────────────────────────────
IN_VENV = Path(sys.prefix).resolve() == VENV.resolve()

if not IN_VENV:
    # 1. 找 Python 3.10
    def find_python310() -> Path:
        # 当前 Python 就是 3.10
        if sys.version_info[:2] == (3, 10):
            return Path(sys.executable)

        # Windows py launcher
        py_launcher = shutil.which("py")
        if py_launcher:
            r = subprocess.run(
                [py_launcher, "-3.10", "-c", "import sys; print(sys.executable)"],
                capture_output=True, text=True,
            )
            if r.returncode == 0:
                return Path(r.stdout.strip())

        # conda 常见路径
        conda_roots = [
            Path(r"D:\miniconda3"),
            Path(r"C:\ProgramData\miniconda3"),
            Path.home() / "miniconda3",
            Path.home() / "anaconda3",
        ]
        for root in conda_roots:
            envs_dir = root / "envs"
            search_dirs = [root]
            if envs_dir.exists():
                search_dirs += list(envs_dir.iterdir())
            for d in search_dirs:
                p = d / "python.exe"
                if not p.exists():
                    continue
                r = subprocess.run(
                    [str(p), "-c", "import sys; print(sys.version_info[:2])"],
                    capture_output=True, text=True,
                )
                if "(3, 10)" in r.stdout:
                    return p

        return None

    py310 = find_python310()
    if py310 is None:
        print("[ERROR] 未找到 Python 3.10")
        print("        请安装 Python 3.10：https://www.python.org/downloads/release/python-31011/")
        print("        或通过 conda 创建：conda create -n py310 python=3.10")
        sys.exit(1)

    print(f"[INFO] 使用 Python 3.10：{py310}")

    # 2. 创建 .venv（如果不存在或损坏）
    if not VENV_PY.exists():
        print(f"[INFO] 创建虚拟环境 {VENV} ...")
        subprocess.run([str(py310), "-m", "venv", str(VENV)], check=True)
        print("[INFO] 虚拟环境创建完成")
    else:
        print(f"[INFO] 虚拟环境已存在：{VENV}")

    # 3. 用 .venv/python 重新执行本脚本（Phase 2）
    print("[INFO] 切换到虚拟环境，继续安装...\n")
    result = subprocess.run([str(VENV_PY), __file__] + sys.argv[1:])
    sys.exit(result.returncode)


# ─────────────────────────────────────────────────────────────────────────────
# Phase 2（Install）：此时已在 .venv 内，sys.executable = .venv/Scripts/python.exe
# ─────────────────────────────────────────────────────────────────────────────

def run(cmd, **kwargs):
    print(f"\n>>> {cmd}\n")
    result = subprocess.run(cmd, shell=True, **kwargs)
    if result.returncode != 0:
        print(f"[ERROR] 命令失败 (exit {result.returncode}): {cmd}")
        sys.exit(1)


def pip(*args):
    run(f'"{sys.executable}" -m pip {" ".join(args)}')


def is_installed(pkg):
    r = subprocess.run(
        [sys.executable, "-m", "pip", "show", pkg],
        capture_output=True,
    )
    return r.returncode == 0


def find_conda_mingw() -> str | None:
    """找 conda 中的 m2w64 gcc，用于编译 pyopenjtalk。"""
    conda_roots = [
        Path(r"D:\miniconda3"),
        Path(r"C:\ProgramData\miniconda3"),
        Path.home() / "miniconda3",
        Path.home() / "anaconda3",
    ]
    for root in conda_roots:
        gcc = root / "Library" / "mingw-w64" / "bin" / "gcc.exe"
        if gcc.exists():
            return str(gcc.parent)
    return None


def find_conda_library_bin() -> list[str]:
    """找 conda 的 Library/bin，用于 DLL 路径（torchcodec/ffmpeg）。"""
    conda_roots = [
        Path(r"D:\miniconda3"),
        Path(r"C:\ProgramData\miniconda3"),
        Path.home() / "miniconda3",
        Path.home() / "anaconda3",
    ]
    found = []
    for root in conda_roots:
        d = root / "Library" / "bin"
        if d.exists():
            found.append(str(d))
        d2 = root / "Library" / "mingw-w64" / "bin"
        if d2.exists():
            found.append(str(d2))
    return found


def main():
    print("=" * 60)
    print(f"TTS 环境安装脚本  [{sys.executable}]")
    print("=" * 60)

    # 0. 升级 pip
    pip("install --upgrade pip")

    # 1. PyTorch CUDA 12.8
    print("\n[1/7] 安装 PyTorch (CUDA 12.8)...")
    pip(
        "install torch torchvision torchaudio",
        "--index-url https://download.pytorch.org/whl/cu128",
    )
    r = subprocess.run(
        [sys.executable, "-c",
         "import torch; print('CUDA:', torch.cuda.is_available(), '|',"
         " torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'N/A')"],
        capture_output=True, text=True,
    )
    print(r.stdout.strip() or r.stderr.strip())

    # 2. GPT-SoVITS 依赖
    print("\n[2/7] 安装 GPT-SoVITS 依赖...")
    gsv_req = GSV_DIR / "requirements.txt"
    if not gsv_req.exists():
        print(f"[ERROR] 找不到 {gsv_req}")
        sys.exit(1)

    # editdistance 先单独装预编译 wheel，避免触发 C++ 编译
    pip("install editdistance --only-binary :all:")

    skip_prefixes = ("--no-binary", "pyopenjtalk", "python_mecab_ko", "onnxruntime;", "editdistance")
    filtered = []
    for line in gsv_req.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        if any(stripped.lower().startswith(p.lower()) for p in skip_prefixes):
            print(f"  [跳过] {stripped}")
            continue
        filtered.append(stripped)

    tmp_req = ROOT / "_gsv_req_tmp.txt"
    tmp_req.write_text("\n".join(filtered), encoding="utf-8")
    pip(f'install -r "{tmp_req}"')
    tmp_req.unlink()

    pip("install opencc")

    # 3. 额外推理依赖
    print("\n[3/7] 安装额外推理依赖...")
    pip("install faster-whisper gruut auraloss torchcodec")

    # 4. ffmpeg DLL（供 torchcodec 使用）
    print("\n[4/7] 安装 ffmpeg DLL...")
    conda_lib_bins = find_conda_library_bin()
    if conda_lib_bins:
        print(f"  找到 conda ffmpeg DLL 路径：{conda_lib_bins[0]}")
    else:
        # 尝试通过 conda 安装 ffmpeg 到 base
        print("  尝试通过 conda 安装 ffmpeg...")
        subprocess.run(
            "conda install ffmpeg -y",
            shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        )
        conda_lib_bins = find_conda_library_bin()
        if not conda_lib_bins:
            print("  [WARN] 未找到 conda ffmpeg DLL，torchcodec 可能无法使用")

    # 5. pyopenjtalk（日语支持，需要 C 编译器）
    print("\n[5/7] 安装 pyopenjtalk...")
    if is_installed("pyopenjtalk"):
        print("  已安装，跳过")
    else:
        mingw_bin = find_conda_mingw()
        if mingw_bin:
            print(f"  使用 conda MinGW 编译：{mingw_bin}")
            env = os.environ.copy()
            env["PATH"] = mingw_bin + os.pathsep + env.get("PATH", "")
            r = subprocess.run(
                [sys.executable, "-m", "pip", "install", "pyopenjtalk"],
                env=env,
            )
        else:
            # 没有 mingw，尝试通过 conda 先安装工具链
            print("  未找到 MinGW，尝试通过 conda 安装 m2w64-toolchain...")
            subprocess.run(
                "conda install -c conda-forge m2w64-toolchain -y",
                shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
            )
            mingw_bin = find_conda_mingw()
            env = os.environ.copy()
            if mingw_bin:
                env["PATH"] = mingw_bin + os.pathsep + env.get("PATH", "")
            r = subprocess.run(
                [sys.executable, "-m", "pip", "install", "pyopenjtalk"],
                env=env,
            )
        if r.returncode != 0:
            print("  [ERROR] pyopenjtalk 安装失败，日语 TTS 将不可用")
            sys.exit(1)

    # 预下载 OpenJTalk 字典（避免首次请求时从 GitHub 下载）
    print("  预下载 OpenJTalk 字典（约 23MB）...")
    r = subprocess.run(
        [sys.executable, "-c",
         "import pyopenjtalk; pyopenjtalk.g2p('テスト'); print('  字典已就绪')"],
    )
    if r.returncode != 0:
        print("  [WARN] 字典下载失败，首次 TTS 请求时将自动重试")

    # 6. 本项目依赖 + sitecustomize.py
    print("\n[6/7] 安装 TTS API 依赖...")
    pip(f'install -r "{ROOT / "requirements.txt"}"')

    print("  写入 sitecustomize.py（Windows DLL + PATH 修复）...")
    site_pkgs = Path(sys.prefix) / "Lib" / "site-packages"
    # 收集所有需要加入 DLL 搜索路径和 PATH 的目录
    dll_dirs = [str(ROOT / "ffmpeg" / "bin")] + conda_lib_bins
    dll_dirs_repr = repr(dll_dirs)
    (site_pkgs / "sitecustomize.py").write_text(
        '"""Add DLL directories and ffmpeg to PATH on startup (Windows fix)."""\n'
        "import os, sys\n"
        "if sys.platform == 'win32':\n"
        f"    _ffmpeg_dirs = {dll_dirs_repr}\n"
        "    for _d in _ffmpeg_dirs:\n"
        "        if os.path.isdir(_d):\n"
        "            # DLL 搜索路径（供 Python C 扩展使用）\n"
        "            if hasattr(os, 'add_dll_directory'):\n"
        "                try: os.add_dll_directory(_d)\n"
        "                except Exception: pass\n"
        "            # PATH（供 os.system/subprocess 调用 ffmpeg CLI 使用）\n"
        "            if _d not in os.environ.get('PATH', ''):\n"
        "                os.environ['PATH'] = _d + os.pathsep + os.environ.get('PATH', '')\n",
        encoding="utf-8",
    )

    # 7. 模型
    print("\n[附] 下载 fast-langdetect 模型...")
    lang_cache = GSV_DIR / "GPT_SoVITS" / "pretrained_models" / "fast_langdetect"
    lang_cache.mkdir(parents=True, exist_ok=True)
    dl_tmp = ROOT / "_dl_langdetect_tmp.py"
    dl_tmp.write_text(
        f'import os\nos.environ["FTLANG_CACHE"] = r"{lang_cache}"\n'
        "from fast_langdetect import detect\ndetect('test')\nprint('  ok')\n",
        encoding="utf-8",
    )
    subprocess.run([sys.executable, str(dl_tmp)])
    dl_tmp.unlink()

    print("\n[7/7] 检查 GPT-SoVITS 预训练模型...")
    key_file = (
        GSV_DIR / "GPT_SoVITS" / "pretrained_models"
        / "gsv-v2final-pretrained"
        / "s1bert25hz-5kh-longer-epoch=12-step=369668.ckpt"
    )
    if key_file.exists():
        print("  预训练模型已存在，跳过")
    else:
        print("  下载中（约 1.5GB）...")
        r = subprocess.run(
            [sys.executable, "-c",
             "from modelscope import snapshot_download; "
             "snapshot_download('AI-ModelScope/GPT-SoVITS', "
             "local_dir='GPT_SoVITS/pretrained_models')"],
            cwd=str(GSV_DIR),
        )
        if r.returncode != 0:
            print("[ERROR] 预训练模型下载失败，请检查网络")
            sys.exit(1)

    print("\n[附] 检查 ffmpeg CLI...")
    ffmpeg_exe = ROOT / "ffmpeg" / "bin" / "ffmpeg.exe"
    if ffmpeg_exe.exists():
        print("  已存在，跳过")
    else:
        run(f'"{sys.executable}" "{ROOT / "scripts" / "setup_ffmpeg.py"}"')

    # 验证
    print("\n验证关键依赖...")
    checks = ["torch", "torchcodec", "pyopenjtalk", "faster_whisper", "gruut"]
    all_ok = True
    for pkg in checks:
        r = subprocess.run(
            [sys.executable, "-c", f"import {pkg}; print('  ok: {pkg}')"],
            capture_output=True, text=True,
        )
        if r.returncode == 0:
            print(r.stdout.strip())
        else:
            print(f"  MISS: {pkg}")
            all_ok = False

    print("\n" + "=" * 60)
    print("安装完成！" if all_ok else "[WARN] 部分依赖未通过验证，查看上方日志")
    print()
    print("启动方式：")
    print(f"  终端 1：cd GPT-SoVITS && {VENV_PY} api_v2.py -a 127.0.0.1 -p 9880")
    print(f"  终端 2：{VENV_PY} run.py")
    print("=" * 60)


if __name__ == "__main__":
    main()
