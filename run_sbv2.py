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


def get_model():
    global _tts_model
    if _tts_model is None:
        from style_bert_vits2.tts_model import TTSModel
        model_path = _find_model(ASSETS_DIR)
        config_path = ASSETS_DIR / "config.json"
        style_vec_path = ASSETS_DIR / "style_vectors.npy"
        print(f"加载模型: {model_path.name}")
        _tts_model = TTSModel(
            model_path=model_path,
            config_path=config_path,
            style_vec_path=style_vec_path,
            device="cuda",
        )
    return _tts_model


def _synthesize(text: str, style: str, style_weight: float, language: str,
                sdp_ratio: float, noise: float, noise_w: float, length: float) -> bytes:
    """同步合成，返回 WAV 字节。"""
    from style_bert_vits2.constants import Languages

    lang_map = {"ja": Languages.JP, "jp": Languages.JP, "en": Languages.EN, "zh": Languages.ZH}
    lang = lang_map.get(language.lower(), Languages.JP)

    model = get_model()
    sr, audio = model.infer(
        text=text,
        language=lang,
        speaker_id=0,
        style=style,
        style_weight=style_weight,
        sdp_ratio=sdp_ratio,
        noise=noise,
        noise_w=noise_w,
        length=length,
    )

    buf = io.BytesIO()
    wavfile.write(buf, sr, audio)
    return buf.getvalue()


# ── FastAPI ──────────────────────────────────────────────────────────────────

from contextlib import asynccontextmanager

def _load_config_model_name():
    global _sbv2_model_name
    cfg_path = ROOT / "run_with_class_config.txt"
    if cfg_path.exists():
        for line in cfg_path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line.startswith("#") or "=" not in line:
                continue
            k, _, v = line.partition("=")
            if k.strip() == "sbv2_model" and v.strip():
                _sbv2_model_name = v.strip()
                break


@asynccontextmanager
async def lifespan(app: FastAPI):
    _load_config_model_name()
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


@app.post("/tts")
async def tts_post(request: TTSRequest):
    text = request.text.strip()
    if not text:
        raise HTTPException(status_code=400, detail="text 不能为空")

    loop = asyncio.get_event_loop()
    try:
        wav_bytes = await loop.run_in_executor(
            None,
            _synthesize,
            text, request.style, request.style_weight, request.language,
            request.sdp_ratio, request.noise, request.noise_w, request.length,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"合成失败: {e}")

    print(f"[tts] {text!r} ({request.language}, {request.style})")
    return Response(content=wav_bytes, media_type="audio/wav")


@app.get("/tts")
async def tts_get(text: str = "", language: str = "ja", style: str = "Neutral"):
    if not text:
        raise HTTPException(status_code=400, detail="需要 query 参数: text")
    return await tts_post(TTSRequest(text=text, language=language, style=style))


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
