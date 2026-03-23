"""
Step 6: Style-BERT-VITS2 推理测试。

使用训练好的 eris 模型合成语音。
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SBV2_DIR = ROOT / "Style-BERT-VITS2"

# 需要从 SBV2 目录运行以解析相对路径 (BERT 模型等)
import os
os.chdir(SBV2_DIR)
sys.path.insert(0, str(SBV2_DIR))

from style_bert_vits2.tts_model import TTSModel
from style_bert_vits2.constants import Languages
import scipy.io.wavfile as wavfile
import numpy as np


def find_latest_model(assets_dir: Path) -> Path:
    """找到最新的 safetensors 模型文件 (按修改时间排序)。"""
    models = sorted(assets_dir.glob("eris_e*_s*.safetensors"), key=lambda p: p.stat().st_mtime)
    if not models:
        raise FileNotFoundError(f"未找到训练模型: {assets_dir}")
    return models[-1]


def main() -> None:
    assets_dir = SBV2_DIR / "model_assets" / "eris"
    output_dir = ROOT / "sbv2_test_output"
    output_dir.mkdir(exist_ok=True)

    model_path = find_latest_model(assets_dir)
    config_path = assets_dir / "config.json"
    style_vec_path = assets_dir / "style_vectors.npy"

    print(f"模型: {model_path.name}")
    print(f"配置: {config_path}")
    print(f"风格: {style_vec_path}")
    print()

    model = TTSModel(
        model_path=model_path,
        config_path=config_path,
        style_vec_path=style_vec_path,
        device="cuda",
    )

    # 测试句子 (日文 - エリス风格)
    test_sentences = [
        "ルーデウス、あなたは本当にすごいわね！",
        "わたしはエリスよ。グレイラット家の娘なの。",
        "どうして私を置いていくの？",
        "一緒に冒険に行きましょう！",
        "ふん、そんなの簡単よ。",
    ]

    print(f"{'='*60}")
    print(f"  推理测试 ({len(test_sentences)} 句)")
    print(f"{'='*60}")

    for i, text in enumerate(test_sentences, 1):
        print(f"\n[{i}] {text}")
        sr, audio = model.infer(
            text=text,
            language=Languages.JP,
            speaker_id=0,
            style="Neutral",
            style_weight=1.0,
            sdp_ratio=0.2,
            noise=0.6,
            noise_w=0.8,
            length=1.0,
        )

        out_path = output_dir / f"eris_test_{i:03d}.wav"
        wavfile.write(str(out_path), sr, audio)
        duration = len(audio) / sr
        print(f"    -> {out_path.name} ({duration:.1f}s, {sr}Hz)")

    print(f"\n{'='*60}")
    print(f"  测试完成! 输出目录: {output_dir}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
