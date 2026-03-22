"""TTS API 服务入口。"""
from __future__ import annotations

import os
from contextlib import asynccontextmanager

import httpx
from fastapi import FastAPI, HTTPException
from fastapi.responses import Response
from pydantic import BaseModel, Field

from .gpt_sovits_client import synthesize as gpt_synthesize
from .voice_catalog import load_catalog


@asynccontextmanager
async def lifespan(app: FastAPI):
    """启动时加载音色目录。"""
    app.state.catalog = load_catalog()
    yield
    app.state.catalog = {}


app = FastAPI(title="TTS API", lifespan=lifespan)


class TTSRequest(BaseModel):
    """TTS 请求体。"""

    text: str = Field(..., min_length=1, description="要合成的文本")
    voice_id: str = Field(..., min_length=1, description="音色 ID")


@app.post("/tts")
async def tts_post(request: TTSRequest):
    """
    文本转语音。返回 WAV 音频。
    """
    catalog = app.state.catalog
    voice_id = request.voice_id
    text = request.text.strip()

    if not text:
        raise HTTPException(status_code=400, detail="text 不能为空")

    if voice_id not in catalog:
        raise HTTPException(
            status_code=404,
            detail=f"音色不存在: {voice_id}。可用音色: {list(catalog.keys())}",
        )

    voice = catalog[voice_id]
    ref_path = voice["reference_path"]
    cfg = voice["config"]

    prompt_text = cfg.get("prompt_text", "")
    prompt_lang = cfg.get("prompt_language") or cfg.get("prompt_lang", "zh")
    text_lang = cfg.get("text_language", "zh")
    speed = float(cfg.get("speed", 1.0))

    api_url = os.environ.get("GPT_SOVITS_API_URL", "http://127.0.0.1:9880")

    try:
        wav_bytes = await gpt_synthesize(
            ref_audio_path=ref_path,
            prompt_text=prompt_text,
            prompt_lang=prompt_lang,
            text=text,
            text_lang=text_lang,
            speed_factor=speed,
            batch_size=1,
            api_url=api_url,
        )
    except FileNotFoundError as e:
        raise HTTPException(status_code=500, detail=str(e))
    except httpx.HTTPStatusError as e:
        raise HTTPException(
            status_code=502,
            detail=f"GPT-SoVITS 服务错误: {e.response.text}",
        )
    except httpx.RequestError as e:
        raise HTTPException(
            status_code=503,
            detail=f"无法连接 GPT-SoVITS 服务 ({api_url})，请确认已启动: {e!s}",
        )

    return Response(
        content=wav_bytes,
        media_type="audio/wav",
    )


@app.get("/tts")
async def tts_get(text: str = "", voice_id: str = ""):
    """GET 方式调用 TTS，便于浏览器测试。"""
    if not text or not voice_id:
        raise HTTPException(
            status_code=400,
            detail="需要 query 参数: text, voice_id",
        )
    return await tts_post(TTSRequest(text=text, voice_id=voice_id))


@app.get("/voices")
async def list_voices():
    """列出可用音色。"""
    return {"voices": list(app.state.catalog.keys())}


@app.get("/health")
async def health():
    """健康检查。"""
    return {"status": "ok"}
