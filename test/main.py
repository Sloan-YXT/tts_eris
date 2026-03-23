"""
自动化 TTS 测试调度。

执行顺序：
  1. 启动 GPT-SoVITS api_v2.py（端口 9880）和 run.py（端口 8010），若已在运行则跳过
  2. 等待两个服务就绪
  3. 读取 voices/eris_test_* 音色目录
  4. 读取 test/text.txt 文本列表
  5. 对每个音色创建 test/output/{voice_id}/ 子文件夹
  6. 循环 M×N 次请求，已存在的 WAV 直接跳过（支持断点续跑）
  7. 完成后终止本次启动的子进程

用法：
  python test/main.py
  python test/main.py --n 50        # 只取前 50 条文本
"""
from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

import httpx

SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = SCRIPT_DIR.parent
PYTHON = ROOT / ".venv" / "Scripts" / "python.exe"
GSV = ROOT / "GPT-SoVITS"

TTS_URL = "http://127.0.0.1:8010"
GSV_URL = "http://127.0.0.1:9880"

TEXT_FILE = SCRIPT_DIR / "text.txt"
OUTPUT_DIR = SCRIPT_DIR / "output"


# ── 工具 ──────────────────────────────────────────────────────────────────────

def is_up(url: str) -> bool:
    try:
        httpx.get(url, timeout=2)
        return True
    except Exception:
        return False


def wait_for(url: str, label: str, timeout: int = 180) -> None:
    print(f"等待 {label} 就绪 ", end="", flush=True)
    for _ in range(timeout):
        if is_up(url):
            print(" OK")
            return
        time.sleep(1)
        print(".", end="", flush=True)
    print()
    raise TimeoutError(f"{label} 在 {timeout}s 内未就绪")


def get_voices() -> list[str]:
    voices_dir = ROOT / "voices"
    return sorted(
        d.name
        for d in voices_dir.iterdir()
        if d.is_dir()
        and (d / "config.json").exists()
        and d.name.startswith("eris_test_")
    )


# ── 主流程 ────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=None, help="只取前 N 条文本（默认全部）")
    args = parser.parse_args()

    procs: list[subprocess.Popen] = []

    # 1. 启动 GPT-SoVITS api_v2.py
    if is_up(f"{GSV_URL}/docs") or is_up(GSV_URL):
        print("GPT-SoVITS (9880) 已在运行，跳过启动")
    else:
        print("启动 GPT-SoVITS api_v2.py ...")
        procs.append(subprocess.Popen(
            [str(PYTHON), "api_v2.py", "-a", "127.0.0.1", "-p", "9880"],
            cwd=str(GSV),
        ))

    # 2. 启动 run.py（每次强制重启，确保新建的 eris_test_* 音色目录被加载）
    if is_up(f"{TTS_URL}/health"):
        print("TTS API (8010) 已在运行，终止旧进程以重新加载音色目录...")
        subprocess.run(
            ["powershell", "-Command",
             "Get-NetTCPConnection -LocalPort 8010 -ErrorAction SilentlyContinue "
             "| ForEach-Object { Stop-Process -Id $_.OwningProcess -Force -ErrorAction SilentlyContinue }"],
            capture_output=True,
        )
        time.sleep(2)
    print("启动 run.py ...")
    procs.append(subprocess.Popen(
        [str(PYTHON), str(ROOT / "run.py")],
        cwd=str(ROOT),
    ))

    try:
        wait_for(GSV_URL, "GPT-SoVITS (9880)", timeout=180)
        wait_for(f"{TTS_URL}/health", "TTS API (8010)", timeout=60)

        # 3. 读取文本
        all_texts = [
            l.strip()
            for l in TEXT_FILE.read_text(encoding="utf-8").splitlines()
            if l.strip()
        ]
        texts = all_texts[: args.n] if args.n else all_texts

        # 4. 读取音色
        voices = get_voices()

        if not texts:
            print("text.txt 为空，退出")
            return
        if not voices:
            print("未找到 eris_test_* 音色，请先运行 character_search.py，退出")
            return

        total = len(voices) * len(texts)
        print(f"\n文本: {len(texts)} 条  |  音色: {len(voices)} 个  |  总请求: {total}")
        print(f"输出目录: {OUTPUT_DIR}\n")

        OUTPUT_DIR.mkdir(exist_ok=True)
        done = 0
        skipped = 0
        errors = 0

        # 5. M×N 请求
        for voice_id in voices:
            voice_out = OUTPUT_DIR / voice_id
            voice_out.mkdir(exist_ok=True)

            for i, text in enumerate(texts):
                out_file = voice_out / f"text_{i:03d}.wav"

                # 断点续跑：文件已存在则跳过
                if out_file.exists():
                    done += 1
                    skipped += 1
                    continue

                try:
                    r = httpx.post(
                        f"{TTS_URL}/tts",
                        json={"text": text, "voice_id": voice_id},
                        timeout=60,
                    )
                    if r.status_code == 200:
                        out_file.write_bytes(r.content)
                    else:
                        errors += 1
                        print(f"  [{voice_id}] {i:03d} HTTP {r.status_code}: {r.text[:120]}")
                except Exception as e:
                    errors += 1
                    print(f"  [{voice_id}] {i:03d} 异常: {e}")

                done += 1
                if done % 200 == 0 or done == total:
                    elapsed_pct = 100 * done // total
                    print(f"进度 {done}/{total} ({elapsed_pct}%)  跳过: {skipped}  错误: {errors}")

        print(f"\n全部完成！  成功: {total - errors - skipped}  跳过: {skipped}  错误: {errors}")

    finally:
        for p in procs:
            try:
                p.terminate()
            except Exception:
                pass


if __name__ == "__main__":
    main()
