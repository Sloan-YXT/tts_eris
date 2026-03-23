"""
Gemini 批量情绪标注脚本。

对 GPT-SoVITS/output/slicer_opt/ 下所有切片调用 Gemini（语音 + 台词文本同时输入），输出：
  classification/emotion_clips.json   每个切片的标注结果（支持断点续跑）
  classification/emotion_map.json     按情绪分组（GSV 用）

用法：
  python class_annotate.py
  python class_annotate.py --force       # 忽略已有结果，全部重新标注
"""
from __future__ import annotations

import ast
import json
import re
import shutil
import time
import argparse
from pathlib import Path

# ── 情绪分类定义（修改此处即可调整分类体系）─────────────────────────────────────
# 标签含义：
#   neutral    - 平静、日常对话
#   gentle     - 温柔、体贴、感激
#   serious    - 严肃、认真、沉稳
#   confident  - 自信、果断、傲娇式强势
#   surprised  - 惊讶、困惑、疑问
#   happy      - 开心、兴奋、雀跃
#   sad        - 悲伤、遗憾、低落

EMOTION_LABELS: list[str] = [
    "neutral",
    "gentle",
    "serious",
    "confident",
    "surprised",
    "happy",
    "sad",
]

# 每类保留的音色数量上限（断点续跑后重新聚合时也使用此值）
DEFAULT_TOP_N = 3

# Gemini 请求间隔（秒），避免触发速率限制
REQUEST_INTERVAL = 0.5

# ── 路径配置 ──────────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent
SLICER_DIR = ROOT / "GPT-SoVITS" / "output" / "slicer_opt"
ASR_LIST = ROOT / "GPT-SoVITS" / "output" / "asr_opt" / "combined.list"
CLASSIFICATION_DIR = ROOT / "classification"
CLIPS_FILE = CLASSIFICATION_DIR / "emotion_clips.json"
MAP_FILE = CLASSIFICATION_DIR / "emotion_map.json"
LOWCUT_DIR = CLASSIFICATION_DIR / "lowcut"
CREDENTIALS_FILE = ROOT / "credentials.txt"

SKIP_DIRS = {"instruments", "skipped_empty_phoneme"}
MIN_TEXT_LEN = 3


# ── 工具函数 ──────────────────────────────────────────────────────────────────

def load_credentials() -> dict:
    if not CREDENTIALS_FILE.exists():
        raise FileNotFoundError(f"credentials.txt 不存在：{CREDENTIALS_FILE}")
    return ast.literal_eval(CREDENTIALS_FILE.read_text(encoding="utf-8"))


def parse_json_response(text: str) -> dict:
    """从 Gemini 返回文本中提取 JSON，兼容带 markdown 代码块的情况。"""
    text = re.sub(r"^```[a-z]*\n?", "", text.strip(), flags=re.MULTILINE)
    text = re.sub(r"\n?```$", "", text.strip(), flags=re.MULTILINE)
    text = text.strip()
    return json.loads(text)


def make_prompt(transcript: str) -> str:
    labels_str = ", ".join(EMOTION_LABELS)
    return (
        f"Listen to this Japanese audio clip from an anime character (エリス/Eris). "
        f"The transcript of the audio is: 「{transcript}」\n\n"
        f"Do TWO things:\n\n"
        f"1. EMOTION: Using both audio prosody and transcript, "
        f"classify the emotion into exactly one of: {labels_str}.\n\n"
        f"2. QUALITY: Rate the audio quality from 1-10 for TTS training use:\n"
        f"   10 = perfect: clean voice, no background noise, matches transcript exactly\n"
        f"   7-9 = good: minor issues but usable (slight noise, fast speech)\n"
        f"   4-6 = mediocre: noticeable issues (background music, partial mismatch)\n"
        f"   1-3 = bad: wrong speaker, truncated, heavy noise, transcript completely wrong\n\n"
        f"Reply with ONLY valid JSON (no markdown, no explanation outside JSON):\n"
        f'{{"emotion": "<label>", "confidence": <0.0-1.0>, '
        f'"quality": <1-10>, "quality_reason": "<one sentence in Chinese>", '
        f'"reason": "<emotion reason in Chinese>"}}'
    )


def annotate_clip(wav_path: Path, transcript: str, client, model: str) -> dict:
    """调用 Gemini 对单个音频文件做情绪标注（语音 + 文字）。"""
    from google.genai import types

    audio_bytes = wav_path.read_bytes()
    prompt = make_prompt(transcript)

    response = client.models.generate_content(
        model=model,
        contents=[
            types.Part.from_bytes(data=audio_bytes, mime_type="audio/wav"),
            prompt,
        ],
    )

    raw = response.text.strip()
    result = parse_json_response(raw)

    # 校验 emotion 是否在允许列表内
    if result.get("emotion") not in EMOTION_LABELS:
        result["emotion"] = "neutral"
        result["confidence"] = 0.0
        result["parse_fallback"] = True

    # 校验 quality
    try:
        result["quality"] = int(result.get("quality", 5))
    except (ValueError, TypeError):
        result["quality"] = 5

    result["raw"] = raw
    return result


def build_emotion_map(clips: dict, top_n: int) -> dict[str, list[str]]:
    """
    从 clips 字典聚合 emotion_map。
    每类按置信度降序，保留前 top_n 个 voice_id。
    """
    buckets: dict[str, list[tuple[float, str]]] = {e: [] for e in EMOTION_LABELS}

    for voice_id, ann in clips.items():
        emotion = ann.get("emotion", "neutral")
        confidence = float(ann.get("confidence", 0.0))
        if emotion in buckets:
            buckets[emotion].append((confidence, voice_id))

    emotion_map: dict[str, list[str]] = {}
    for emotion, items in buckets.items():
        items.sort(key=lambda x: x[0], reverse=True)
        emotion_map[emotion] = [vid for _, vid in items[:top_n]]

    return emotion_map


# ── 主流程 ────────────────────────────────────────────────────────────────────

def load_asr_map() -> dict[str, str]:
    """从 combined.list 构建 wav 路径 → 台词 映射。"""
    asr_map: dict[str, str] = {}
    if not ASR_LIST.exists():
        return asr_map
    for line in ASR_LIST.read_text(encoding="utf-8").splitlines():
        parts = line.split("|")
        if len(parts) >= 4:
            wav_rel = parts[0].strip().replace("\\", "/")
            text = parts[3].strip()
            if text and len(text) >= MIN_TEXT_LEN:
                asr_map[wav_rel] = text
    return asr_map


def collect_clips(asr_map: dict[str, str]) -> list[tuple[str, Path, str]]:
    """收集所有待标注切片，返回 (clip_id, wav_path, transcript) 列表。"""
    clips_list: list[tuple[str, Path, str]] = []
    for subdir in sorted(SLICER_DIR.iterdir()):
        if not subdir.is_dir() or subdir.name in SKIP_DIRS:
            continue
        for wav_path in sorted(subdir.glob("*.wav")):
            rel_key = f"output/slicer_opt/{subdir.name}/{wav_path.name}"
            transcript = asr_map.get(rel_key, "")
            if not transcript:
                continue
            # clip_id 用 子目录/stem 格式，和 audit 系统一致
            clip_id = f"{subdir.name}/{wav_path.stem}"
            clips_list.append((clip_id, wav_path, transcript))
    return clips_list


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--force", action="store_true",
                        help="忽略已有标注，全部重新请求 Gemini")
    args = parser.parse_args()

    # 加载凭证
    creds = load_credentials()
    gemini_cfg = creds["gemini"]
    min_quality = int(creds.get("min_quality", 6))

    from google import genai
    client = genai.Client(api_key=gemini_cfg["api_key"])
    model = gemini_cfg["model"]

    # 收集切片
    asr_map = load_asr_map()
    if not asr_map:
        print(f"[ERROR] 找不到 ASR 标注: {ASR_LIST}")
        return

    all_clips = collect_clips(asr_map)
    if not all_clips:
        print("未找到有效切片")
        return

    print(f"有效切片：{len(all_clips)} 个")
    print(f"情绪分类：{EMOTION_LABELS}\n")

    # 加载已有标注（支持断点续跑）
    clips: dict = {}
    if CLIPS_FILE.exists() and not args.force:
        clips = json.loads(CLIPS_FILE.read_text(encoding="utf-8"))
        print(f"已加载已有标注：{len(clips)} 条，跳过已完成项\n")

    CLASSIFICATION_DIR.mkdir(parents=True, exist_ok=True)

    total = len(all_clips)
    done = 0
    skipped = 0
    errors = 0

    for clip_id, wav_path, transcript in all_clips:
        if clip_id in clips and not args.force:
            # 自动重试之前失败的（quality=0）
            if clips[clip_id].get("quality", 0) > 0:
                skipped += 1
                continue

        max_retries = 3
        for attempt in range(max_retries):
            try:
                result = annotate_clip(wav_path, transcript, client, model)
                clips[clip_id] = {
                    "emotion":    result["emotion"],
                    "confidence": result["confidence"],
                    "quality":    result["quality"],
                    "quality_reason": result.get("quality_reason", ""),
                    "reason":     result.get("reason", ""),
                    "transcript": transcript,
                }
                done += 1
                q = result["quality"]
                q_mark = "x" if q < min_quality else " "
                print(f"  [{skipped + done}/{total}] {clip_id}: "
                      f"{result['emotion']} ({result['confidence']:.2f}) "
                      f"Q={q:2d} [{q_mark}]  {result.get('quality_reason', '')}")
                break

            except Exception as e:
                if attempt < max_retries - 1:
                    wait = (attempt + 1) * 10
                    print(f"  [{skipped + done + 1}/{total}] {clip_id}: retry {attempt + 1}/{max_retries} ({e}), {wait}s 后重试...")
                    time.sleep(wait)
                else:
                    errors += 1
                    done += 1
                    print(f"  [{skipped + done}/{total}] {clip_id}: ERROR - {e}")
                    clips[clip_id] = {
                        "emotion": "neutral", "confidence": 0.0,
                        "quality": 0, "quality_reason": f"error: {e}",
                        "reason": f"error: {e}", "transcript": transcript,
                    }

        # 每条保存一次，防止中断丢失进度
        CLIPS_FILE.write_text(
            json.dumps(clips, ensure_ascii=False, indent=2), encoding="utf-8"
        )

        time.sleep(REQUEST_INTERVAL)

    print(f"\n标注完成：新增 {done} 条，跳过 {skipped} 条，错误 {errors} 条")

    # 情绪分布
    print("\n情绪分布：")
    emotion_counts: dict[str, int] = {e: 0 for e in EMOTION_LABELS}
    for ann in clips.values():
        e = ann.get("emotion", "neutral")
        if e in emotion_counts:
            emotion_counts[e] += 1
    for emotion, n in emotion_counts.items():
        print(f"  {emotion:12s}: {n}")

    # 质量分布
    print("\n质量分布：")
    q_counts: dict[int, int] = {i: 0 for i in range(1, 11)}
    for ann in clips.values():
        q = ann.get("quality", 0)
        if 1 <= q <= 10:
            q_counts[q] += 1
        else:
            q_counts[1] = q_counts.get(1, 0) + 1
    for score in range(10, 0, -1):
        bar = "#" * q_counts[score]
        print(f"  Q={score:2d}: {q_counts[score]:4d}  {bar}")

    # 得分区间统计
    print("\n得分区间：")
    range_labels = [("1-3 (差)", 1, 3), ("4-5 (一般)", 4, 5),
                    ("6-7 (可用)", 6, 7), ("8-10 (优)", 8, 10)]
    for label, lo, hi in range_labels:
        n = sum(q_counts[s] for s in range(lo, hi + 1))
        print(f"  {label:12s}: {n}")

    # 按阈值统计
    for threshold in [5, 6, 7]:
        kept = sum(1 for ann in clips.values() if ann.get("quality", 0) >= threshold)
        print(f"  Q>={threshold}: {kept}/{len(clips)} 条保留")

    # 复制低质量音频到 lowcut/
    low_clips = {cid: ann for cid, ann in clips.items()
                 if ann.get("quality", 0) < min_quality}
    if low_clips:
        if LOWCUT_DIR.exists():
            shutil.rmtree(LOWCUT_DIR)
        LOWCUT_DIR.mkdir(parents=True)
        low_copied = 0
        for clip_id, ann in low_clips.items():
            # clip_id = 子目录/stem
            parts = clip_id.split("/", 1)
            if len(parts) != 2:
                continue
            subdir_name, stem = parts
            src = SLICER_DIR / subdir_name / f"{stem}.wav"
            if src.exists():
                dst_dir = LOWCUT_DIR / subdir_name
                dst_dir.mkdir(exist_ok=True)
                shutil.copy2(src, dst_dir / f"{stem}.wav")
                low_copied += 1
        print(f"\n低质量音频 (Q<{min_quality}): {low_copied} 个 → {LOWCUT_DIR}")
    else:
        print(f"\n无低质量音频 (Q<{min_quality})")

    print(f"\n输出文件：{CLIPS_FILE}")


if __name__ == "__main__":
    main()
