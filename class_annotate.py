"""
Gemini 批量情绪标注脚本。

对 voices/eris_avl_* 下所有参考音频调用 Gemini（语音 + 台词文本同时输入），输出：
  classification/emotion_clips.json   每个音色的原始标注结果（支持断点续跑）
  classification/emotion_map.json     按情绪分组，每类保留置信度最高的前 N 个

用法：
  python class_annotate.py
  python class_annotate.py --top-n 5     # 每类保留前5个（默认3）
  python class_annotate.py --force       # 忽略已有结果，全部重新标注
"""
from __future__ import annotations

import ast
import json
import re
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
VOICES_DIR = ROOT / "voices"
CLASSIFICATION_DIR = ROOT / "classification"
CLIPS_FILE = CLASSIFICATION_DIR / "emotion_clips.json"
MAP_FILE = CLASSIFICATION_DIR / "emotion_map.json"
CREDENTIALS_FILE = ROOT / "credentials.txt"


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
        f"Listen to this Japanese audio clip from an anime character. "
        f"The transcript of the audio is: 「{transcript}」\n\n"
        f"Using both the audio prosody and the transcript, "
        f"classify the speaker's emotion into exactly one of these labels: {labels_str}.\n\n"
        f"Reply with ONLY valid JSON (no markdown, no explanation outside JSON):\n"
        f'{{ "emotion": "<label>", "confidence": <0.0-1.0>, "reason": "<one sentence in Chinese>" }}'
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

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--top-n", type=int, default=DEFAULT_TOP_N,
                        help=f"每类情绪保留的最大音色数（默认 {DEFAULT_TOP_N}）")
    parser.add_argument("--force", action="store_true",
                        help="忽略已有标注，全部重新请求 Gemini")
    args = parser.parse_args()

    # 加载凭证
    creds = load_credentials()
    gemini_cfg = creds["gemini"]

    from google import genai
    client = genai.Client(api_key=gemini_cfg["api_key"])
    model = gemini_cfg["model"]

    # 找到所有 eris_avl_* 音色目录
    voice_dirs = sorted(
        d for d in VOICES_DIR.iterdir()
        if d.is_dir() and d.name.startswith("eris_avl_") and (d / "reference.wav").exists()
    )

    if not voice_dirs:
        print("未找到 eris_avl_* 音色目录")
        return

    print(f"找到音色目录：{len(voice_dirs)} 个")
    print(f"情绪分类：{EMOTION_LABELS}")
    print(f"每类保留前 {args.top_n} 个\n")

    # 加载已有标注（支持断点续跑）
    clips: dict = {}
    if CLIPS_FILE.exists() and not args.force:
        clips = json.loads(CLIPS_FILE.read_text(encoding="utf-8"))
        print(f"已加载已有标注：{len(clips)} 条，跳过已完成项\n")

    CLASSIFICATION_DIR.mkdir(parents=True, exist_ok=True)

    total = len(voice_dirs)
    done = 0
    errors = 0

    for voice_dir in voice_dirs:
        voice_id = voice_dir.name
        wav_path = voice_dir / "reference.wav"

        if voice_id in clips and not args.force:
            done += 1
            continue

        # 读取台词文本（来自 config.json 的 prompt_text）
        cfg_path = voice_dir / "config.json"
        transcript = ""
        if cfg_path.exists():
            try:
                transcript = json.loads(cfg_path.read_text(encoding="utf-8")).get("prompt_text", "")
            except Exception:
                pass

        try:
            result = annotate_clip(wav_path, transcript, client, model)
            clips[voice_id] = {
                "emotion":    result["emotion"],
                "confidence": result["confidence"],
                "reason":     result.get("reason", ""),
                "transcript": transcript,
            }
            done += 1
            print(f"  [{done}/{total}] {voice_id}: {result['emotion']} ({result['confidence']:.2f})  {result.get('reason', '')}")

        except Exception as e:
            errors += 1
            print(f"  [{done}/{total}] {voice_id}: ERROR - {e}")
            clips[voice_id] = {"emotion": "neutral", "confidence": 0.0, "reason": f"error: {e}", "transcript": transcript}

        # 每条保存一次，防止中断丢失进度
        CLIPS_FILE.write_text(
            json.dumps(clips, ensure_ascii=False, indent=2), encoding="utf-8"
        )

        time.sleep(REQUEST_INTERVAL)

    print(f"\n标注完成：{done} 条，错误：{errors}")

    # 生成 emotion_map
    emotion_map = build_emotion_map(clips, args.top_n)
    MAP_FILE.write_text(
        json.dumps(emotion_map, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    print("\nemotion_map 统计：")
    for emotion, ids in emotion_map.items():
        print(f"  {emotion:12s}: {len(ids)} 个  {ids}")

    print(f"\n输出文件：")
    print(f"  {CLIPS_FILE}")
    print(f"  {MAP_FILE}")


if __name__ == "__main__":
    main()
