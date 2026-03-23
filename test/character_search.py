"""
搜索 combined.list 中所有来源（1/2/3）适合做参考音频的切片，
为每个建立 voices/eris_test_{n:03d}/ 音色文件夹。
重复执行时清空旧 eris_test_* 结果后重建。
"""
from __future__ import annotations

import json
import shutil
import wave
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = SCRIPT_DIR.parent
GSV = ROOT / "GPT-SoVITS"
VOICES_DIR = ROOT / "voices"
COMBINED_LIST = GSV / "output" / "asr_opt" / "combined.list"

CFG_BASE = {
    "prompt_language": "ja",
    "text_language": "ja",
    "speed": 1.0,
    "gpt_model": "GPT_weights_v2Pro/eris-e50.ckpt",
    "sovits_model": "SoVITS_weights_v2Pro/eris_e20_s1520.pth",
}

EMOTIONAL_WORDS = [
    "わよ", "なさい", "ちょうだい", "やめて",
    "ばか", "バカ", "最低", "怖", "泣",
]

SKIP_PUNCTUATION = set("！？!?…（）()「」【】♪～")


def get_duration(wav_path: Path) -> float:
    try:
        with wave.open(str(wav_path)) as w:
            return w.getnframes() / w.getframerate()
    except Exception:
        return 0.0


def is_suitable(text: str, duration: float) -> bool:
    if not (3.0 <= duration <= 10.0):
        return False
    if len(text) < 10 or len(text) > 30:
        return False
    if any(c in text for c in SKIP_PUNCTUATION):
        return False
    if any(w in text for w in EMOTIONAL_WORDS):
        return False
    return True


def main():
    # 1. 清除旧 eris_test_* 文件夹
    removed = 0
    for d in VOICES_DIR.glob("eris_test_*"):
        shutil.rmtree(d)
        removed += 1
    if removed:
        print(f"清除旧音色文件夹: {removed} 个")

    # 2. 读取 combined.list，解析所有条目
    if not COMBINED_LIST.exists():
        print(f"combined.list 不存在: {COMBINED_LIST}")
        return

    lines = COMBINED_LIST.read_text(encoding="utf-8").splitlines()
    candidates = []
    seen_paths: set[str] = set()

    for line in lines:
        parts = line.split("|")
        if len(parts) < 4:
            continue
        rel_path = parts[0].strip()
        lang = parts[2].strip()
        text = parts[3].strip()

        if lang != "JA":
            continue

        # 解析 WAV 路径（相对于 GSV 目录）
        wav_path = GSV / rel_path.replace("\\", "/")
        key = str(wav_path).lower()
        if key in seen_paths:
            continue
        if not wav_path.exists():
            continue
        seen_paths.add(key)

        duration = get_duration(wav_path)
        if not is_suitable(text, duration):
            continue

        candidates.append((wav_path, text, round(duration, 2)))

    print(f"找到合适切片: {len(candidates)} 个")

    # 3. 建立音色文件夹
    for i, (wav_path, text, duration) in enumerate(candidates, 1):
        dst_dir = VOICES_DIR / f"eris_test_{i:03d}"
        dst_dir.mkdir(exist_ok=True)
        shutil.copy2(wav_path, dst_dir / "reference.wav")
        cfg = {"prompt_text": text, **CFG_BASE}
        (dst_dir / "config.json").write_text(
            json.dumps(cfg, ensure_ascii=False, indent=2), encoding="utf-8"
        )

    if candidates:
        print(f"音色文件夹已建立: eris_test_001 ~ eris_test_{len(candidates):03d}")
    else:
        print("未找到任何合适切片，请检查 combined.list 和音频文件。")


if __name__ == "__main__":
    main()
