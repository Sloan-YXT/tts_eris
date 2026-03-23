"""
多风格训练 Step 4: 多风格推理测试。

用训练好的多风格模型，对每种风格生成测试音频。
输出目录: sbv2_multistyle_test_output/
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SBV2_DIR = ROOT / "Style-BERT-VITS2"
ASSETS_DIR = SBV2_DIR / "model_assets" / "eris"

import os
os.chdir(SBV2_DIR)
sys.path.insert(0, str(SBV2_DIR))


def find_latest_model(assets_dir: Path) -> Path:
    """按修改时间找到最新的 safetensors 模型。"""
    models = sorted(assets_dir.glob("eris_e*_s*.safetensors"), key=lambda p: p.stat().st_mtime)
    if not models:
        raise FileNotFoundError(f"未找到训练模型: {assets_dir}")
    return models[-1]


def main() -> None:
    output_dir = ROOT / "sbv2_multistyle_test_output"
    output_dir.mkdir(exist_ok=True)

    # 加载配置，获取 style2id
    config_path = ASSETS_DIR / "config.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    style2id = config["data"]["style2id"]
    num_styles = config["data"]["num_styles"]

    print(f"{'='*60}")
    print(f"  多风格推理测试")
    print(f"{'='*60}")
    print(f"  num_styles: {num_styles}")
    print(f"  style2id: {style2id}")

    # 加载模型
    model_path = find_latest_model(ASSETS_DIR)
    style_vec_path = ASSETS_DIR / "style_vectors.npy"

    print(f"  模型: {model_path.name}")
    print(f"  style_vectors: {style_vec_path.name}")
    print()

    from style_bert_vits2.tts_model import TTSModel
    from style_bert_vits2.constants import Languages
    import scipy.io.wavfile as wavfile

    print("加载模型...")
    t0 = time.time()
    model = TTSModel(
        model_path=model_path,
        config_path=config_path,
        style_vec_path=style_vec_path,
        device="cuda",
    )
    print(f"模型加载完成 ({time.time()-t0:.1f}s)")

    # 每种风格 × 多个测试句 → 对比音频
    test_sentences = [
        "ルーデウス、あなたは本当にすごいわね！",
        "どうして私を置いていくの？",
        "一緒に冒険に行きましょう！",
    ]

    styles = list(style2id.keys())
    total = len(styles) * len(test_sentences)
    idx = 0

    print(f"\n生成 {total} 个测试音频 ({len(styles)} 风格 × {len(test_sentences)} 句)...")
    print()

    for style_name in styles:
        for i, text in enumerate(test_sentences, 1):
            idx += 1
            t0 = time.time()

            sr, audio = model.infer(
                text=text,
                language=Languages.JP,
                speaker_id=0,
                style=style_name,
                style_weight=1.0,
                sdp_ratio=0.2,
                noise=0.6,
                noise_w=0.8,
                length=1.0,
            )

            dt = time.time() - t0
            safe_style = style_name.lower().replace(" ", "_")
            out_path = output_dir / f"{safe_style}_sent{i}.wav"
            wavfile.write(str(out_path), sr, audio)
            duration = len(audio) / sr
            print(f"  [{idx:2d}/{total}] {style_name:12s} | {text[:20]:20s} → {out_path.name} ({duration:.1f}s, {dt:.1f}s)")

    # 额外测试: style_weight 对比 (用一个明显情绪的句子)
    print(f"\n{'='*60}")
    print(f"  Style Weight 对比测试")
    print(f"{'='*60}")

    test_text = "どうして私を置いていくの？"
    # 只选几个有代表性的风格
    contrast_styles = [s for s in ["sad", "Sad", "happy", "Happy"] if s in style2id]
    if not contrast_styles:
        contrast_styles = [s for s in styles if s.lower() != "neutral"][:2]

    for style_name in contrast_styles:
        for weight in [0.5, 1.0, 1.5, 2.0]:
            sr, audio = model.infer(
                text=test_text,
                language=Languages.JP,
                speaker_id=0,
                style=style_name,
                style_weight=weight,
                sdp_ratio=0.2,
                noise=0.6,
                noise_w=0.8,
                length=1.0,
            )
            safe_style = style_name.lower().replace(" ", "_")
            out_path = output_dir / f"weight_{safe_style}_w{weight:.1f}.wav"
            wavfile.write(str(out_path), sr, audio)
            duration = len(audio) / sr
            print(f"  {style_name} weight={weight:.1f} → {out_path.name} ({duration:.1f}s)")

    print(f"\n{'='*60}")
    print(f"  测试完成! 输出目录: {output_dir}")
    print(f"  共 {len(list(output_dir.glob('*.wav')))} 个音频文件")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
