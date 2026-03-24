"""
人工审查可疑切片。

流程：
  1. python audit_voices.py              → 生成 audit_suspects.txt
  2. python manual_audit.py 0            → 复制可疑 wav 到 audit/ 供试听
  3. 在 audit/ 文件夹中听，把确认要删的 wav 从 audit/ 删掉
  4. python manual_audit.py 1            → 把你删掉的（suspects 中有但 audit/ 中没有的）
                                            从 slicer_opt 原始位置 + combined.list 移除

用法：
  python manual_audit.py 0    复制可疑切片到 audit/
  python manual_audit.py 1    清理已确认的坏切片
"""
from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
_venv_py = ROOT / ".venv" / "Scripts" / "python.exe"
if _venv_py.exists() and Path(sys.prefix).resolve() != _venv_py.parent.parent.resolve():
    os.execv(str(_venv_py), [str(_venv_py), __file__] + sys.argv[1:])

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
sys.stderr.reconfigure(encoding="utf-8", errors="replace")

from shared import SLICER_DIR, ASR_LIST
SUSPECTS_FILE = ROOT / "audit_suspects.txt"
AUDIT_DIR = ROOT / "audit"


def parse_suspects() -> list[str]:
    """解析 audit_suspects.txt，返回 name 列表（格式: 子目录/文件stem）。"""
    if not SUSPECTS_FILE.exists():
        print(f"[ERROR] 找不到 {SUSPECTS_FILE}，请先运行 python audit_voices.py")
        sys.exit(1)

    names: list[str] = []
    for line in SUSPECTS_FILE.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        # 跳过标题行
        if not line or line.startswith("可疑") or line.startswith("阈值"):
            continue
        # 格式: 子目录/文件stem  |  原因  |  台词
        name = line.split("|")[0].strip()
        if "/" in name:
            names.append(name)
    return names


def name_to_src(name: str) -> Path:
    """name (子目录/stem) → slicer_opt 原始 wav 路径。"""
    subdir, stem = name.split("/", 1)
    return SLICER_DIR / subdir / f"{stem}.wav"


def name_to_audit(name: str) -> Path:
    """name (子目录/stem) → audit/子目录/stem.wav（保留原始目录结构）。"""
    subdir, stem = name.split("/", 1)
    return AUDIT_DIR / subdir / f"{stem}.wav"


def recycle(path: Path) -> bool:
    """移到回收站。"""
    if not path.exists():
        return False
    try:
        ps_cmd = (
            f'Add-Type -AssemblyName Microsoft.VisualBasic; '
            f'[Microsoft.VisualBasic.FileIO.FileSystem]::DeleteFile('
            f'"{path}", "OnlyErrorDialogs", "SendToRecycleBin")'
        )
        subprocess.run(
            ["powershell", "-Command", ps_cmd],
            check=True, capture_output=True,
        )
        return True
    except Exception:
        return False


# ── 模式 0: 复制可疑切片到 audit/ ─────────────────────────────────────────────

def step0_copy():
    names = parse_suspects()
    if not names:
        print("没有可疑片段")
        return

    # 清理旧的 audit/
    if AUDIT_DIR.exists():
        shutil.rmtree(AUDIT_DIR)
    AUDIT_DIR.mkdir()

    copied = 0
    missing = 0
    for name in names:
        src = name_to_src(name)
        dst = name_to_audit(name)
        dst.parent.mkdir(parents=True, exist_ok=True)
        if src.exists():
            shutil.copy2(src, dst)
            copied += 1
        else:
            print(f"  [MISS] {src}")
            missing += 1

    print(f"\n已复制 {copied} 个可疑切片到 audit/")
    if missing:
        print(f"  {missing} 个源文件不存在（已被删？）")
    print(f"\n下一步：")
    print(f"  1. 打开 audit/ 文件夹逐个试听")
    print(f"  2. 把确认要删除的 wav 从 audit/ 中删掉")
    print(f"  3. 留下的 = 要保留的")
    print(f"  4. 运行: python manual_audit.py 1")


# ── 模式 1: 清理已确认的坏切片 ────────────────────────────────────────────────

def step1_cleanup():
    names = parse_suspects()
    if not names:
        print("没有可疑片段")
        return

    if not AUDIT_DIR.exists():
        print("[ERROR] audit/ 目录不存在，请先运行 python manual_audit.py 0")
        sys.exit(1)

    # 看 audit/ 中还剩哪些（= 用户保留的）
    kept = set()
    for f in AUDIT_DIR.rglob("*.wav"):
        # 还原 name: audit/子目录/stem.wav → 子目录/stem
        rel = f.relative_to(AUDIT_DIR)
        kept.add(f"{rel.parent}/{rel.stem}")

    # suspects 中有但 audit/ 中没有的 = 确认要删的
    to_remove = [n for n in names if n not in kept]

    if not to_remove:
        print("没有需要清理的切片（audit/ 中的文件都保留了）")
        return

    print(f"确认删除 {len(to_remove)} 个坏切片：\n")

    # 1. 移动原始 wav 到回收站
    recycled = 0
    for name in to_remove:
        src = name_to_src(name)
        if recycle(src):
            print(f"  recycled: {name}")
            recycled += 1
        elif not src.exists():
            print(f"  [已删] {name}")
        else:
            print(f"  [FAIL] {name}")

    # 2. 从 combined.list 移除对应行
    if ASR_LIST.exists():
        # 构建要删除的路径集合
        remove_keys = set()
        for name in to_remove:
            subdir, stem = name.split("/", 1)
            remove_keys.add(f"output/slicer_opt/{subdir}/{stem}.wav")

        original_lines = ASR_LIST.read_text(encoding="utf-8").splitlines()
        new_lines = []
        removed_lines = 0
        for line in original_lines:
            wav_rel = line.split("|")[0].strip().replace("\\", "/")
            if wav_rel in remove_keys:
                removed_lines += 1
            else:
                new_lines.append(line)

        ASR_LIST.write_text("\n".join(new_lines) + "\n", encoding="utf-8")
        print(f"\ncombined.list: 删除 {removed_lines} 行（{len(original_lines)} → {len(new_lines)}）")

    print(f"\n清理完成: {recycled} 个 wav 移到回收站")
    print(f"保留: {len(kept)} 个")
    print(f"\n现在可以运行: python train_all.py")


# ── 入口 ──────────────────────────────────────────────────────────────────────

def main():
    if len(sys.argv) < 2 or sys.argv[1] not in ("0", "1"):
        print("用法:")
        print("  python manual_audit.py 0    复制可疑切片到 audit/ 供试听")
        print("  python manual_audit.py 1    清理已确认的坏切片")
        sys.exit(1)

    if sys.argv[1] == "0":
        step0_copy()
    else:
        step1_cleanup()


if __name__ == "__main__":
    main()
