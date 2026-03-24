"""
多风格训练 Step 1: 数据准备。

1. 备份当前单风格模型 (model_assets/eris → model_assets/eris_v1_neutral)
2. 按 emotion_clips.json 将 wavs 目录下的文件移入情绪子目录
3. 同时移动关联文件 (.wav.npy, .wav.bert.pt, .wav.spec.pt)
4. 重新生成 esd.list (路径更新为子目录)

前提: emotion_clips.json 已存在 (classification/ 下)
"""
from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

from shared import EMOTION_LABELS, CLIPS_FILE, SBV2_DIR

DATA_DIR = SBV2_DIR / "Data" / "eris"
WAVS_DIR = DATA_DIR / "wavs"
ASSETS_DIR = SBV2_DIR / "model_assets"

EMOTIONS = EMOTION_LABELS


def backup_model() -> None:
    """备份当前模型到 model_assets/eris_v1_neutral/"""
    src = ASSETS_DIR / "eris"
    dst = ASSETS_DIR / "eris_v1_neutral"
    if dst.exists():
        print(f"  备份已存在: {dst.name}, 跳过")
        return
    shutil.copytree(src, dst)
    print(f"  已备份: {src.name} → {dst.name}")


def build_wav_to_emotion(clips: dict) -> dict[str, str]:
    """从 clip_map.json + emotion_clips.json 建立 wav 文件名（0001.wav）→ 情绪 映射。"""
    clip_map_file = DATA_DIR / "clip_map.json"
    if not clip_map_file.exists():
        print("  [WARN] clip_map.json 不存在，无法映射情绪")
        return {}

    # clip_map: {"0001.wav": "1/vocal_eris_full..._0000148480", ...}
    clip_map = json.loads(clip_map_file.read_text(encoding="utf-8"))

    wav_to_emotion: dict[str, str] = {}
    for wav_name, clip_id in clip_map.items():
        if clip_id in clips:
            wav_to_emotion[wav_name] = clips[clip_id]["emotion"]

    return wav_to_emotion


def reorganize_wavs(clips: dict) -> dict[str, int]:
    """将 wav 及关联文件按情绪移入子目录。返回每个情绪的文件数。"""
    counts: dict[str, int] = {e: 0 for e in EMOTIONS}

    # 创建子目录
    for emotion in EMOTIONS:
        (WAVS_DIR / emotion).mkdir(exist_ok=True)

    # 获取 wav → emotion 映射
    wav_to_emotion = build_wav_to_emotion(clips)

    # 检查是否有平铺 wav（排除子目录中的）
    flat_wavs = [f for f in sorted(WAVS_DIR.glob("*.wav")) if f.parent == WAVS_DIR]
    if not flat_wavs:
        print("  wavs 目录下无平铺 wav 文件，可能已经分好了")
        for emotion in EMOTIONS:
            counts[emotion] = len(list((WAVS_DIR / emotion).glob("*.wav")))
        return counts

    # 移动文件
    moved = 0
    missing_label = []
    for wav in flat_wavs:
        emotion = wav_to_emotion.get(wav.name)
        if not emotion:
            missing_label.append(wav.name)
            emotion = "neutral"

        if emotion not in EMOTIONS:
            print(f"  [warn] 未知情绪 '{emotion}' for {wav.name}, 归入 neutral")
            emotion = "neutral"

        dst_dir = WAVS_DIR / emotion

        # 移动 wav 及所有关联文件
        for suffix in ["", ".npy"]:
            src_file = WAVS_DIR / f"{wav.name}{suffix}"
            if src_file.exists():
                shutil.move(str(src_file), str(dst_dir / src_file.name))
        for suffix in [".bert.pt", ".spec.pt"]:
            src_file = WAVS_DIR / f"{wav.stem}{suffix}"
            if src_file.exists():
                shutil.move(str(src_file), str(dst_dir / src_file.name))

        counts[emotion] += 1
        moved += 1

    if missing_label:
        print(f"  [warn] {len(missing_label)} 个文件无情绪标签，归入 neutral")

    print(f"  已移动 {moved} 组文件")
    return counts


def regenerate_esd_list() -> int:
    """根据子目录结构重新生成 esd.list，从旧 esd.list 读台词。"""
    # 先读旧 esd.list 建立 wav 文件名 → 台词映射
    esd_path = DATA_DIR / "esd.list"
    old_transcripts: dict[str, str] = {}
    if esd_path.exists():
        for line in esd_path.read_text(encoding="utf-8").splitlines():
            parts = line.split("|")
            if len(parts) >= 4:
                old_transcripts[Path(parts[0]).name] = parts[3]

    lines = []
    for emotion in EMOTIONS:
        emo_dir = WAVS_DIR / emotion
        for wav in sorted(emo_dir.glob("*.wav")):
            if wav.name.endswith(".spec.wav"):
                continue
            transcript = old_transcripts.get(wav.name, "")
            if not transcript:
                continue
            rel_path = wav.relative_to(SBV2_DIR).as_posix()
            lines.append(f"{rel_path}|eris|JP|{transcript}")

    if not lines:
        print("  [FAIL] esd.list 为空，无有效数据")
        sys.exit(1)
    esd_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"  已生成 esd.list: {len(lines)} 条")
    return len(lines)


def main() -> None:
    print(f"{'='*60}")
    print(f"  多风格训练 Step 1: 数据准备")
    print(f"{'='*60}")

    # 加载情绪标注
    if not CLIPS_FILE.exists():
        print(f"[FAIL] 情绪标注文件不存在: {CLIPS_FILE}")
        sys.exit(1)
    clips = json.loads(CLIPS_FILE.read_text(encoding="utf-8"))
    print(f"\n情绪标注: {len(clips)} 条")

    # 1. 备份模型
    print(f"\n[1/3] 备份当前模型...")
    backup_model()

    # 2. 按情绪重组 wavs
    print(f"\n[2/3] 按情绪重组 wav 文件...")
    counts = reorganize_wavs(clips)
    for emotion, n in counts.items():
        print(f"    {emotion:12s}: {n} 条")
    print(f"    {'合计':12s}: {sum(counts.values())} 条")

    # 3. 重新生成 esd.list
    print(f"\n[3/3] 重新生成 esd.list...")
    regenerate_esd_list()

    print(f"\n{'='*60}")
    print(f"  Step 1 完成!")
    print(f"  子目录结构: wavs/{'{' + ','.join(EMOTIONS) + '}'}/")
    print(f"  下一步: 运行 sbv2_multistyle_step2_preprocess.py")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
