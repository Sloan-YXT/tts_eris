"""
Step 2: 将 eris_avl_* 音频和台词整理为 Style-BERT-VITS2 训练格式。

输出：
  Style-BERT-VITS2/Data/eris/raw/     所有 wav 文件
  Style-BERT-VITS2/Data/eris/esd.list  台词列表
"""
from __future__ import annotations

import json
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parent
VOICES_DIR = ROOT / "voices"
SBV2_DIR = ROOT / "Style-BERT-VITS2"
DATA_DIR = SBV2_DIR / "Data" / "eris"
RAW_DIR = DATA_DIR / "raw"
ESD_FILE = DATA_DIR / "esd.list"

SPEAKER = "eris"
LANG = "JP"


def main() -> None:
    RAW_DIR.mkdir(parents=True, exist_ok=True)

    voice_dirs = sorted(
        d for d in VOICES_DIR.iterdir()
        if d.is_dir() and d.name.startswith("eris_avl_") and (d / "reference.wav").exists()
    )

    if not voice_dirs:
        print("未找到 eris_avl_* 音色目录")
        return

    lines: list[str] = []
    copied = 0
    skipped_no_text = 0

    for voice_dir in voice_dirs:
        cfg_path = voice_dir / "config.json"
        wav_path = voice_dir / "reference.wav"

        # 读台词
        text = ""
        if cfg_path.exists():
            try:
                text = json.loads(cfg_path.read_text(encoding="utf-8")).get("prompt_text", "")
            except Exception:
                pass

        if not text.strip():
            skipped_no_text += 1
            continue

        # 复制 wav
        dst_name = f"{voice_dir.name}.wav"
        dst = RAW_DIR / dst_name
        if not dst.exists():
            shutil.copy2(wav_path, dst)

        # esd.list 格式: Data/eris/wavs/文件名|说话人|语言|台词
        # 路径相对于 SBV2_DIR（preprocess_text.py 的 cwd）
        lines.append(f"Data/eris/wavs/{dst_name}|{SPEAKER}|{LANG}|{text.strip()}")
        copied += 1

    ESD_FILE.write_text("\n".join(lines), encoding="utf-8")

    print(f"数据准备完成：")
    print(f"  音频文件: {copied} 个 -> {RAW_DIR}")
    print(f"  台词列表: {ESD_FILE}")
    print(f"  跳过（无台词）: {skipped_no_text}")

    # 显示前 3 行样例
    print(f"\nesd.list 样例：")
    for line in lines[:3]:
        print(f"  {line}")


if __name__ == "__main__":
    main()
