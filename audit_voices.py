"""
音频质量审计：自动标记可疑的 voice 片段。

检查项：
  1. 时长异常（< 1s 或 > 15s）
  2. SNR 偏低（可能有背景噪音）
  3. 语音占比低（静音多或音乐残留）
  4. 音量过小（RMS 过低）

不会删除任何文件，只输出可疑列表供人工复查。

用法：python audit_voices.py
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
_venv_py = ROOT / ".venv" / "Scripts" / "python.exe"
if _venv_py.exists() and Path(sys.prefix).resolve() != _venv_py.parent.parent.resolve():
    os.execv(str(_venv_py), [str(_venv_py), __file__] + sys.argv[1:])

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
sys.stderr.reconfigure(encoding="utf-8", errors="replace")

import json
import numpy as np
import librosa
import torch

SLICER_DIR = ROOT / "GPT-SoVITS" / "output" / "slicer_opt"
ASR_LIST = ROOT / "GPT-SoVITS" / "output" / "asr_opt" / "combined.list"
SKIP_DIRS = {"instruments", "skipped_empty_phoneme"}

# ── 阈值（宽松标准，只抓明显有问题的） ─────────────────────────────
MIN_DURATION = 0.8    # 秒，太短的片段
MAX_DURATION = 18.0   # 秒，太长的片段
MIN_SNR = 5.0         # dB，低于此视为噪音大
MIN_SPEECH_RATIO = 0.3  # 语音占比低于 30% 视为可疑
MIN_RMS_DB = -40.0    # dB，音量过小


def compute_snr(y: np.ndarray, sr: int) -> float:
    """简易 SNR 估算：用 top 10% RMS 作信号，bottom 10% 作噪音。"""
    frame_length = int(0.025 * sr)
    hop_length = int(0.010 * sr)
    rms = librosa.feature.rms(y=y, frame_length=frame_length, hop_length=hop_length)[0]
    if len(rms) < 4:
        return 0.0
    sorted_rms = np.sort(rms)
    n = len(sorted_rms)
    noise_rms = np.mean(sorted_rms[:max(1, n // 10)]) + 1e-10
    signal_rms = np.mean(sorted_rms[-(max(1, n // 10)):]) + 1e-10
    return 20 * np.log10(signal_rms / noise_rms)


def compute_rms_db(y: np.ndarray) -> float:
    """整体 RMS（dB）。"""
    rms = np.sqrt(np.mean(y ** 2)) + 1e-10
    return 20 * np.log10(rms)


def compute_speech_ratio(y: np.ndarray, sr: int, vad_model, get_stamps) -> float:
    """用 Silero VAD 计算语音占总时长的比例。"""
    # Silero 要求 16kHz mono
    if sr != 16000:
        y_16k = librosa.resample(y, orig_sr=sr, target_sr=16000)
    else:
        y_16k = y
    wav_tensor = torch.FloatTensor(y_16k)

    stamps = get_stamps(wav_tensor, vad_model, sampling_rate=16000,
                        threshold=0.3, min_speech_duration_ms=100)
    if not stamps:
        return 0.0
    speech_dur = sum(s["end"] - s["start"] for s in stamps)
    total_dur = len(y_16k) / 16000
    return speech_dur / total_dur if total_dur > 0 else 0.0


def main() -> None:
    # 读取 ASR 标注
    asr_map: dict[str, str] = {}
    if ASR_LIST.exists():
        for line in ASR_LIST.read_text(encoding="utf-8").splitlines():
            parts = line.split("|")
            if len(parts) >= 4:
                wav_rel = parts[0].strip().replace("\\", "/")
                asr_map[wav_rel] = parts[3].strip()

    # 收集 slicer_opt 所有子目录的 wav
    wav_files: list[Path] = []
    for subdir in sorted(SLICER_DIR.iterdir()):
        if not subdir.is_dir() or subdir.name in SKIP_DIRS:
            continue
        wav_files.extend(sorted(subdir.glob("*.wav")))

    if not wav_files:
        print(f"未找到切片: {SLICER_DIR}")
        sys.exit(1)

    print(f"扫描 {len(wav_files)} 个切片...\n")

    # 加载 Silero VAD
    print("加载 Silero VAD...")
    vad_model, utils = torch.hub.load("snakers4/silero-vad", "silero_vad", trust_repo=True)
    get_stamps = utils[0]  # get_speech_timestamps
    print("VAD 加载完成\n")

    suspects: list[dict] = []
    stats = {"total": 0, "ok": 0, "suspect": 0}

    for wav_path in wav_files:
        stats["total"] += 1
        # 用 子目录/文件名 作为标识
        name = f"{wav_path.parent.name}/{wav_path.stem}"
        reasons: list[str] = []

        try:
            y, sr = librosa.load(str(wav_path), sr=None, mono=True)
        except Exception as e:
            reasons.append(f"加载失败: {e}")
            suspects.append({"name": name, "reasons": reasons})
            stats["suspect"] += 1
            continue

        duration = len(y) / sr
        snr = compute_snr(y, sr)
        rms_db = compute_rms_db(y)

        # VAD
        try:
            speech_ratio = compute_speech_ratio(y, sr, vad_model, get_stamps)
        except Exception:
            speech_ratio = 1.0  # VAD 失败不标记

        # 判定
        if duration < MIN_DURATION:
            reasons.append(f"太短 {duration:.1f}s")
        if duration > MAX_DURATION:
            reasons.append(f"太长 {duration:.1f}s")
        if snr < MIN_SNR:
            reasons.append(f"SNR低 {snr:.1f}dB")
        if speech_ratio < MIN_SPEECH_RATIO:
            reasons.append(f"语音占比低 {speech_ratio:.0%}")
        if rms_db < MIN_RMS_DB:
            reasons.append(f"音量小 {rms_db:.1f}dB")

        if reasons:
            rel_key = f"output/slicer_opt/{wav_path.parent.name}/{wav_path.name}"
            prompt = asr_map.get(rel_key, "")[:50]
            suspects.append({
                "name": name,
                "duration": f"{duration:.1f}s",
                "snr": f"{snr:.1f}dB",
                "rms": f"{rms_db:.1f}dB",
                "speech": f"{speech_ratio:.0%}",
                "reasons": reasons,
                "prompt": prompt,
            })
            stats["suspect"] += 1
        else:
            stats["ok"] += 1

    # 输出结果
    print(f"{'=' * 70}")
    print(f"  扫描完成: {stats['total']} 个片段")
    print(f"  正常: {stats['ok']}  |  可疑: {stats['suspect']}")
    print(f"{'=' * 70}\n")

    if not suspects:
        print("没有发现可疑片段！")
        return

    print(f"{'编号':<16} {'时长':>5} {'SNR':>7} {'RMS':>7} {'语音%':>6}  问题             台词")
    print("-" * 100)
    for s in suspects:
        reasons_str = ", ".join(s.get("reasons", []))
        print(f"{s['name']:<16} {s.get('duration','?'):>5} {s.get('snr','?'):>7} "
              f"{s.get('rms','?'):>7} {s.get('speech','?'):>6}  {reasons_str:<16} {s.get('prompt','')}")

    # 保存到文件
    out_path = ROOT / "audit_suspects.txt"
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(f"可疑片段: {stats['suspect']}/{stats['total']}\n")
        f.write(f"阈值: 时长 {MIN_DURATION}-{MAX_DURATION}s, SNR>{MIN_SNR}dB, "
                f"语音>{MIN_SPEECH_RATIO:.0%}, RMS>{MIN_RMS_DB}dB\n\n")
        for s in suspects:
            reasons_str = ", ".join(s.get("reasons", []))
            f.write(f"{s['name']}  |  {reasons_str}  |  {s.get('prompt','')}\n")
    print(f"\n详细列表已保存到: {out_path}")


if __name__ == "__main__":
    main()
