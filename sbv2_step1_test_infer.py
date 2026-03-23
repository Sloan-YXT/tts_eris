"""
Step 1: 验证 Style-BERT-VITS2 推理环境。
用内置 jvnv-F1-jp 音色生成一句日语，确认环境正常。
"""
from pathlib import Path

SBV2_DIR = Path(__file__).resolve().parent / "Style-BERT-VITS2"

import sys, os
os.chdir(SBV2_DIR)
sys.path.insert(0, str(SBV2_DIR))

import soundfile as sf

from style_bert_vits2.tts_model import TTSModel
from style_bert_vits2.nlp import bert_models
from style_bert_vits2.constants import Languages

print("加载 BERT 模型（首次较慢）...")
bert_models.load_model(Languages.JP, "ku-nlp/deberta-v2-large-japanese-char-wwm")
bert_models.load_tokenizer(Languages.JP, "ku-nlp/deberta-v2-large-japanese-char-wwm")

print("加载 TTS 模型...")
assets = SBV2_DIR / "model_assets"
model = TTSModel(
    model_path=assets / "jvnv-F1-jp" / "jvnv-F1-jp_e160_s14000.safetensors",
    config_path=assets / "jvnv-F1-jp" / "config.json",
    style_vec_path=assets / "jvnv-F1-jp" / "style_vectors.npy",
    device="cuda",
)

text = "こんにちは、スタイルバートビッツ2の推理テストです。"
styles = ["Neutral", "Happy", "Sad"]
out_dir = Path(__file__).resolve().parent / "sbv2_test_output"
out_dir.mkdir(exist_ok=True)

for style in styles:
    print(f"生成: {style}...")
    sr, audio = model.infer(
        text=text,
        language=Languages.JP,
        style=style,
        style_weight=1.0,
    )
    out_path = out_dir / f"test_{style.lower()}.wav"
    sf.write(str(out_path), audio, sr)
    print(f"  -> {out_path} ({len(audio)/sr:.1f}s)")

print(f"\n验证完成，输出在 {out_dir}")
