"""GPT-SoVITS API 客户端。调用官方 api_v2 服务。"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import httpx

# 默认 GPT-SoVITS API 地址（官方 api_v2.py 默认 9880）
DEFAULT_API_URL = "http://127.0.0.1:9880"


async def switch_models(
    *,
    gpt_model: str | None,
    sovits_model: str | None,
    api_url: str = DEFAULT_API_URL,
    timeout: float = 30.0,
) -> None:
    """切换 GPT/SoVITS 权重（调用 api_v2 的 set_gpt_weights / set_sovits_weights）。"""
    base = api_url.rstrip("/")
    async with httpx.AsyncClient(timeout=timeout) as client:
        if gpt_model:
            resp = await client.get(f"{base}/set_gpt_weights", params={"weights_path": gpt_model})
            resp.raise_for_status()
        if sovits_model:
            resp = await client.get(f"{base}/set_sovits_weights", params={"weights_path": sovits_model})
            resp.raise_for_status()


async def synthesize(
    *,
    ref_audio_path: str | Path,
    prompt_text: str,
    prompt_lang: str,
    text: str,
    text_lang: str = "zh",
    speed_factor: float = 1.0,
    batch_size: int = 1,
    api_url: str = DEFAULT_API_URL,
    timeout: float = 120.0,
) -> bytes:
    """
    调用 GPT-SoVITS api_v2 /tts 合成语音，返回 WAV 字节。
    4GB 显存：batch_size=1，半精度由 GPT-SoVITS 配置决定。
    """
    path = Path(ref_audio_path)
    if not path.exists():
        raise FileNotFoundError(f"参考音频不存在: {ref_audio_path}")

    payload: dict[str, Any] = {
        "ref_audio_path": str(path.resolve()),
        "prompt_text": prompt_text or "",
        "prompt_lang": prompt_lang,
        "text": text,
        "text_lang": text_lang,
        "batch_size": batch_size,
        "speed_factor": speed_factor,
        "media_type": "wav",
        "streaming_mode": False,
    }

    url = api_url.rstrip("/") + "/tts"
    async with httpx.AsyncClient(timeout=timeout) as client:
        resp = await client.post(url, json=payload)
        resp.raise_for_status()
        return resp.content
