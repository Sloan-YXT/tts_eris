"""
Step 2: 从 GPT-SoVITS 切片 + ASR 标注整理为 Style-BERT-VITS2 训练格式。

数据来源：
  GPT-SoVITS/output/slicer_opt/*/   切片 wav
  GPT-SoVITS/output/asr_opt/combined.list  ASR 标注
  classification/emotion_clips.json  Gemini 质量评分（可选）

输出：
  Style-BERT-VITS2/Data/eris/raw/     所有 wav 文件
  Style-BERT-VITS2/Data/eris/esd.list  台词列表

自动跳过：instruments 目录、无台词片段、台词过短（<3字）、质量低于阈值的片段。
"""
from __future__ import annotations

import ast
import json
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parent
GSV_DIR = ROOT / "GPT-SoVITS"
SLICER_DIR = GSV_DIR / "output" / "slicer_opt"
ASR_LIST = GSV_DIR / "output" / "asr_opt" / "combined.list"
CLIPS_FILE = ROOT / "classification" / "emotion_clips.json"
CREDENTIALS_FILE = ROOT / "credentials.txt"

SBV2_DIR = ROOT / "Style-BERT-VITS2"
DATA_DIR = SBV2_DIR / "Data" / "eris"
RAW_DIR = DATA_DIR / "raw"
ESD_FILE = DATA_DIR / "esd.list"

SPEAKER = "eris"
LANG = "JP"

# 跳过的子目录
SKIP_DIRS = {"instruments", "skipped_empty_phoneme"}
# 最短台词字数
MIN_TEXT_LEN = 3


def main() -> None:
    # 从配置文件读取质量阈值
    min_quality = 6
    if CREDENTIALS_FILE.exists():
        creds = ast.literal_eval(CREDENTIALS_FILE.read_text(encoding="utf-8"))
        min_quality = int(creds.get("min_quality", 6))

    # 清理旧数据
    if RAW_DIR.exists():
        old_count = len(list(RAW_DIR.glob("*.wav")))
        if old_count > 0:
            print(f"清理旧训练数据: {old_count} 个 wav")
            for f in RAW_DIR.glob("*.wav"):
                f.unlink()
    RAW_DIR.mkdir(parents=True, exist_ok=True)

    # 1. 解析 ASR 标注，建立 wav 路径 → 台词 映射
    if not ASR_LIST.exists():
        print(f"[ERROR] 找不到 ASR 标注: {ASR_LIST}")
        print("请先在 GPT-SoVITS WebUI 中完成 ASR 标注")
        return

    asr_map: dict[str, str] = {}  # 相对路径 → 台词
    for line in ASR_LIST.read_text(encoding="utf-8").splitlines():
        parts = line.split("|")
        if len(parts) < 4:
            continue
        wav_rel = parts[0].strip().replace("\\", "/")  # 统一路径分隔符
        text = parts[3].strip()
        if text:
            asr_map[wav_rel] = text

    print(f"ASR 标注: {len(asr_map)} 条有效")

    # 2. 遍历 slicer_opt 所有子目录
    if not SLICER_DIR.exists():
        print(f"[ERROR] 找不到切片目录: {SLICER_DIR}")
        return

    subdirs = sorted([
        d for d in SLICER_DIR.iterdir()
        if d.is_dir() and d.name not in SKIP_DIRS
    ])

    # 3. 加载 Gemini 质量评分（可选）
    quality_map: dict[str, int] = {}  # clip_id → quality
    if CLIPS_FILE.exists():
        clips = json.loads(CLIPS_FILE.read_text(encoding="utf-8"))
        for clip_id, ann in clips.items():
            quality_map[clip_id] = int(ann.get("quality", 10))
        print(f"质量评分: {len(quality_map)} 条（阈值 Q>={min_quality}）")
    else:
        print("质量评分: 无（跳过质量过滤）")

    lines: list[str] = []
    copied = 0
    skipped_no_text = 0
    skipped_short = 0
    skipped_quality = 0

    for subdir in subdirs:
        wavs = sorted(subdir.glob("*.wav"))
        sub_count = 0

        for wav_path in wavs:
            # 构造 ASR 标注中的相对路径键
            rel_key = f"output/slicer_opt/{subdir.name}/{wav_path.name}"
            text = asr_map.get(rel_key, "")

            if not text:
                skipped_no_text += 1
                continue
            if len(text) < MIN_TEXT_LEN:
                skipped_short += 1
                continue

            # 质量过滤（clip_id = 子目录/stem）
            if quality_map:
                clip_id = f"{subdir.name}/{wav_path.stem}"
                quality = quality_map.get(clip_id, 10)  # 无评分默认保留
                if quality < min_quality:
                    skipped_quality += 1
                    continue

            # 用序号命名避免超长文件名
            dst_name = f"{copied + 1:04d}.wav"
            dst = RAW_DIR / dst_name
            if not dst.exists():
                shutil.copy2(wav_path, dst)

            # esd.list 格式: Data/eris/wavs/文件名|说话人|语言|台词
            lines.append(f"Data/eris/wavs/{dst_name}|{SPEAKER}|{LANG}|{text}")
            copied += 1
            sub_count += 1

        print(f"  {subdir.name}/: {sub_count} 条")

    ESD_FILE.write_text("\n".join(lines), encoding="utf-8")

    print(f"\n数据准备完成：")
    print(f"  音频文件: {copied} 个 -> {RAW_DIR}")
    print(f"  台词列表: {ESD_FILE}")
    print(f"  跳过（无台词）: {skipped_no_text}")
    print(f"  跳过（台词<{MIN_TEXT_LEN}字）: {skipped_short}")
    if quality_map:
        print(f"  跳过（质量<{min_quality}）: {skipped_quality}")

    # 显示前 3 行样例
    print(f"\nesd.list 样例：")
    for line in lines[:3]:
        print(f"  {line}")


if __name__ == "__main__":
    main()
