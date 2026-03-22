"""参考音频预处理：静音切除、音量标准化。"""
from __future__ import annotations

from pathlib import Path
from typing import Tuple

import librosa
import numpy as np
import soundfile as sf


def trim_silence(
    y: np.ndarray, sr: int, top_db: float = 25.0
) -> Tuple[np.ndarray, int]:
    """切除首尾静音。"""
    y_trimmed, _ = librosa.effects.trim(y, top_db=top_db)
    return y_trimmed, sr


def normalize_volume(y: np.ndarray, target_db: float = -20.0) -> np.ndarray:
    """音量标准化到目标 dB。"""
    rms = np.sqrt(np.mean(y**2))
    if rms < 1e-8:
        return y
    current_db = 20 * np.log10(rms + 1e-8)
    gain_db = target_db - current_db
    gain = 10 ** (gain_db / 20)
    return (y * gain).astype(np.float32)


def preprocess_reference(
    input_path: str | Path,
    output_path: str | Path | None = None,
    *,
    top_db: float = 25.0,
    target_db: float = -20.0,
    sr: int | None = None,
) -> np.ndarray:
    """
    预处理参考音频：加载、静音切除、音量标准化。
    若指定 output_path 则保存处理后的 WAV。
    """
    path = Path(input_path)
    y, orig_sr = librosa.load(path, sr=sr, mono=True)

    y, _ = trim_silence(y, orig_sr, top_db=top_db)
    y = normalize_volume(y, target_db=target_db)

    if output_path is not None:
        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        sf.write(out, y, orig_sr, subtype="PCM_16")

    return y
