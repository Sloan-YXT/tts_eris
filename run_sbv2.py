"""
Style-BERT-VITS2 TTS API 服务。

直接加载训练好的 eris 模型，无需外部后端进程。

用法：
  python run_sbv2.py

端口：8010（可通过 --port 修改）
"""
from __future__ import annotations

import asyncio
import io
import os
import sys
from pathlib import Path

# 自动切换到 .venv
_venv_py = Path(__file__).resolve().parent / ".venv" / "Scripts" / "python.exe"
if _venv_py.exists() and Path(sys.prefix).resolve() != _venv_py.parent.parent.resolve():
    os.execv(str(_venv_py), [str(_venv_py), __file__] + sys.argv[1:])

ROOT = Path(__file__).resolve().parent
SBV2_DIR = ROOT / "Style-BERT-VITS2"
ASSETS_DIR = SBV2_DIR / "model_assets" / "eris"

# SBV2 需要从其目录运行以找到 BERT 模型等
os.chdir(SBV2_DIR)
sys.path.insert(0, str(SBV2_DIR))

import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.responses import Response
from pydantic import BaseModel, Field
import scipy.io.wavfile as wavfile
import numpy as np


_sbv2_model_name: str | None = None  # 从配置读取，None 则自动选最新


def _find_model(assets_dir: Path) -> Path:
    if _sbv2_model_name:
        p = assets_dir / _sbv2_model_name
        if p.exists():
            return p
        raise FileNotFoundError(f"指定模型不存在: {p}")
    models = sorted(assets_dir.glob("eris_e*_s*.safetensors"), key=lambda p: p.stat().st_mtime)
    if not models:
        raise FileNotFoundError(f"未找到训练模型: {assets_dir}")
    return models[-1]


# ── 全局模型（启动时加载）──────────────────────────────────────────────────────

_tts_model = None
_multi_style: bool = False


def get_model():
    global _tts_model, _multi_style
    if _tts_model is None:
        import json as _json
        from style_bert_vits2.tts_model import TTSModel
        model_path = _find_model(ASSETS_DIR)
        config_path = ASSETS_DIR / "config.json"
        style_vec_path = ASSETS_DIR / "style_vectors.npy"
        cfg = _json.loads(config_path.read_text(encoding="utf-8"))
        num_styles = cfg.get("data", {}).get("num_styles", 1)
        _multi_style = num_styles > 1
        print(f"加载模型: {model_path.name} ({'多风格' if _multi_style else '单风格'})")
        _tts_model = TTSModel(
            model_path=model_path,
            config_path=config_path,
            style_vec_path=style_vec_path,
            device="cuda",
        )
    return _tts_model


_SHORT_TEXT_THRESHOLD = 4


def _synthesize(text: str, style: str, style_weight: float, language: str,
                sdp_ratio: float, noise: float, noise_w: float, length: float) -> bytes:
    """同步合成，返回 WAV 字节。"""
    from style_bert_vits2.constants import Languages

    lang_map = {"ja": Languages.JP, "jp": Languages.JP, "en": Languages.EN, "zh": Languages.ZH}
    lang = lang_map.get(language.lower(), Languages.JP)

    model = get_model()

    # 极短文本降低随机性，让生成更稳定
    if len(text) <= _SHORT_TEXT_THRESHOLD:
        sdp_ratio = min(sdp_ratio, 0.1)
        noise = min(noise, 0.3)
        noise_w = min(noise_w, 0.3)

    style_name = style if _multi_style else "Neutral"
    sr, audio = model.infer(
        text=text, language=lang, speaker_id=0, style=style_name,
        style_weight=style_weight, sdp_ratio=sdp_ratio,
        noise=noise, noise_w=noise_w, length=length,
    )

    # 末尾淡出 + 补静音，消除截断尖锐音（短音频跳过淡出）
    audio = audio.astype(np.float32)
    fade_ms, silence_ms = 80, 50
    fade_samples = int(sr * fade_ms / 1000)
    if len(audio) > fade_samples:
        fade = np.exp(-np.linspace(0, 5, fade_samples)).astype(np.float32)
        audio[-fade_samples:] *= fade
    silence = np.zeros(int(sr * silence_ms / 1000), dtype=np.float32)
    audio = np.concatenate([audio, silence])
    audio = np.clip(audio, -32768, 32767).astype(np.int16)

    buf = io.BytesIO()
    wavfile.write(buf, sr, audio)
    return buf.getvalue()


# ── FastAPI ──────────────────────────────────────────────────────────────────

from contextlib import asynccontextmanager

_pure_mode = False


def _load_config():
    global _sbv2_model_name, _pure_mode
    cfg_path = ROOT / "run_with_class_config.txt"
    if cfg_path.exists():
        for line in cfg_path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line.startswith("#") or "=" not in line:
                continue
            k, _, v = line.partition("=")
            k, v = k.strip(), v.strip()
            if k == "sbv2_model" and v:
                _sbv2_model_name = v
            elif k == "pure_mode":
                _pure_mode = v.lower() == "true"


@asynccontextmanager
async def lifespan(app: FastAPI):
    _load_config()
    print(f"纯净模式: {'开' if _pure_mode else '关'}")
    print("预加载 SBV2 模型...")
    loop = asyncio.get_event_loop()
    await loop.run_in_executor(None, get_model)
    print("模型加载完成")
    yield


app = FastAPI(title="SBV2 TTS API", lifespan=lifespan)


class TTSRequest(BaseModel):
    text: str = Field(..., min_length=1, description="要合成的文本")
    language: str = Field("ja", description="语言: ja/en/zh")
    style: str = Field("Neutral", description="风格")
    style_weight: float = Field(1.0, ge=0, le=20, description="风格权重")
    sdp_ratio: float = Field(0.2, ge=0, le=1, description="SDP 比率")
    noise: float = Field(0.6, ge=0, le=2, description="噪声")
    noise_w: float = Field(0.8, ge=0, le=2, description="SDP 噪声")
    length: float = Field(1.0, ge=0.5, le=2.0, description="语速 (1.0=正常)")
    speed: float | None = Field(None, ge=0.5, le=2.0, description="语速别名，优先于 length")


@app.post("/tts")
async def tts_post(request: TTSRequest):
    text = request.text.strip()
    if not text:
        raise HTTPException(status_code=400, detail="text 不能为空")

    if _pure_mode:
        # 纯净模式：忽略所有调音参数
        style, style_weight, language = "Neutral", 1.0, request.language
        sdp_ratio, noise, noise_w, length = 0.2, 0.6, 0.8, 1.0
        print(f"[pure] {text!r}")
    else:
        style, style_weight, language = request.style, request.style_weight, request.language
        sdp_ratio, noise, noise_w = request.sdp_ratio, request.noise, request.noise_w
        length = request.speed if request.speed is not None else request.length
        print(f"[tts] {text!r} ({language}, {style})")

    loop = asyncio.get_event_loop()
    try:
        wav_bytes = await loop.run_in_executor(
            None,
            _synthesize,
            text, style, style_weight, language,
            sdp_ratio, noise, noise_w, length,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"合成失败: {e}")

    return Response(content=wav_bytes, media_type="audio/wav")


@app.get("/tts")
async def tts_get(text: str = "", language: str = "ja", style: str = "Neutral",
                  speed: float | None = None, style_weight: float | None = None):
    if not text:
        raise HTTPException(status_code=400, detail="需要 query 参数: text")
    kwargs: dict = {"text": text, "language": language, "style": style}
    if speed is not None:
        kwargs["length"] = max(0.5, min(2.0, speed))
    if style_weight is not None:
        kwargs["style_weight"] = max(0.0, min(20.0, style_weight))
    return await tts_post(TTSRequest(**kwargs))


@app.get("/styles")
async def styles():
    import json
    config_path = ASSETS_DIR / "config.json"
    cfg = json.loads(config_path.read_text(encoding="utf-8"))
    style2id = cfg.get("data", {}).get("style2id", {})
    return {"styles": list(style2id.keys()), "style2id": style2id}


@app.get("/health")
async def health():
    model_loaded = _tts_model is not None
    return {"status": "ok", "model_loaded": model_loaded, "model": "eris (SBV2 JP-Extra multi-style)"}


# ── 启动 ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8010)
    parser.add_argument("--host", default="0.0.0.0")
    args = parser.parse_args()

    uvicorn.run("run_sbv2:app", host=args.host, port=args.port, reload=False)
