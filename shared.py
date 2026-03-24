"""跨脚本共享的常量和工具函数。"""
from __future__ import annotations

import ast
from pathlib import Path

ROOT = Path(__file__).resolve().parent

# ── 情绪标签（唯一定义，所有脚本从这里读）──────────────────────────────────────
EMOTION_LABELS: list[str] = [
    "neutral",
    "gentle",
    "serious",
    "confident",
    "surprised",
    "happy",
    "sad",
]

# ── 数据过滤 ────────────────────────────────────────────────────────────────────
SKIP_DIRS = {"instruments", "skipped_empty_phoneme"}
MIN_TEXT_LEN = 3

# ── 路径 ────────────────────────────────────────────────────────────────────────
CREDENTIALS_FILE = ROOT / "credentials.txt"
SLICER_DIR = ROOT / "GPT-SoVITS" / "output" / "slicer_opt"
ASR_LIST = ROOT / "GPT-SoVITS" / "output" / "asr_opt" / "combined.list"
CLASSIFICATION_DIR = ROOT / "classification"
CLIPS_FILE = CLASSIFICATION_DIR / "emotion_clips.json"
SBV2_DIR = ROOT / "Style-BERT-VITS2"


def load_credentials() -> dict:
    """读取 credentials.txt，返回 dict。"""
    if not CREDENTIALS_FILE.exists():
        raise FileNotFoundError(f"credentials.txt 不存在：{CREDENTIALS_FILE}")
    return ast.literal_eval(CREDENTIALS_FILE.read_text(encoding="utf-8"))


def get_min_quality(default: int = 6) -> int:
    """从 credentials.txt 读取 min_quality 阈值。"""
    try:
        creds = load_credentials()
        return int(creds.get("min_quality", default))
    except (FileNotFoundError, SyntaxError):
        return default
