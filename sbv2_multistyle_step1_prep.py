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

ROOT = Path(__file__).resolve().parent
SBV2_DIR = ROOT / "Style-BERT-VITS2"
DATA_DIR = SBV2_DIR / "Data" / "eris"
WAVS_DIR = DATA_DIR / "wavs"
ASSETS_DIR = SBV2_DIR / "model_assets"
CLIPS_FILE = ROOT / "classification" / "emotion_clips.json"

EMOTIONS = ["neutral", "gentle", "serious", "confident", "surprised", "happy", "sad"]


def backup_model() -> None:
    """备份当前模型到 model_assets/eris_v1_neutral/"""
    src = ASSETS_DIR / "eris"
    dst = ASSETS_DIR / "eris_v1_neutral"
    if dst.exists():
        print(f"  备份已存在: {dst.name}, 跳过")
        return
    shutil.copytree(src, dst)
    print(f"  已备份: {src.name} → {dst.name}")


def reorganize_wavs(clips: dict) -> dict[str, int]:
    """将 wav 及关联文件按情绪移入子目录。返回每个情绪的文件数。"""
    counts: dict[str, int] = {e: 0 for e in EMOTIONS}

    # 创建子目录
    for emotion in EMOTIONS:
        (WAVS_DIR / emotion).mkdir(exist_ok=True)

    # 检查是否已经分好了
    flat_wavs = list(WAVS_DIR.glob("eris_avl_*.wav"))
    if not flat_wavs:
        print("  wavs 目录下无平铺 wav 文件，可能已经分好了")
        # 统计子目录
        for emotion in EMOTIONS:
            counts[emotion] = len(list((WAVS_DIR / emotion).glob("*.wav")))
        return counts

    # 移动文件
    moved = 0
    missing_label = []
    for wav in sorted(flat_wavs):
        stem = wav.stem  # eris_avl_001
        clip_info = clips.get(stem)
        if not clip_info:
            missing_label.append(stem)
            continue

        emotion = clip_info["emotion"]
        if emotion not in EMOTIONS:
            print(f"  [warn] 未知情绪 '{emotion}' for {stem}, 归入 neutral")
            emotion = "neutral"

        dst_dir = WAVS_DIR / emotion

        # 移动 wav 及所有关联文件
        # .npy 文件格式: stem.wav.npy (含 .wav)
        # .bert.pt/.spec.pt 格式: stem.bert.pt (不含 .wav)
        for suffix in ["", ".npy"]:
            src_file = WAVS_DIR / f"{wav.name}{suffix}"
            if src_file.exists():
                shutil.move(str(src_file), str(dst_dir / src_file.name))
        for suffix in [".bert.pt", ".spec.pt"]:
            src_file = WAVS_DIR / f"{stem}{suffix}"
            if src_file.exists():
                shutil.move(str(src_file), str(dst_dir / src_file.name))

        counts[emotion] += 1
        moved += 1

    if missing_label:
        print(f"  [warn] {len(missing_label)} 个文件无情绪标签: {missing_label[:5]}...")

    print(f"  已移动 {moved} 组文件")
    return counts


def regenerate_esd_list(clips: dict) -> int:
    """根据子目录结构重新生成 esd.list。"""
    lines = []
    for emotion in EMOTIONS:
        emo_dir = WAVS_DIR / emotion
        for wav in sorted(emo_dir.glob("*.wav")):
            if wav.name.endswith(".wav") and not wav.name.endswith(".spec.wav"):
                stem = wav.stem
                clip_info = clips.get(stem, {})
                transcript = clip_info.get("transcript", "")
                if not transcript:
                    continue
                # 相对于 SBV2_DIR 的路径
                rel_path = wav.relative_to(SBV2_DIR).as_posix()
                lines.append(f"{rel_path}|eris|JP|{transcript}")

    esd_path = DATA_DIR / "esd.list"
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
    regenerate_esd_list(clips)

    print(f"\n{'='*60}")
    print(f"  Step 1 完成!")
    print(f"  子目录结构: wavs/{'{' + ','.join(EMOTIONS) + '}'}/")
    print(f"  下一步: 运行 sbv2_multistyle_step2_preprocess.py")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
