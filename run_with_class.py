"""
情绪路由 TTS 服务（端口 8092）。

DeepSeek 检测文本情感 → 自动选择风格/音色合成。

用法：
  python run_with_class.py          # SBV2 后端（直接加载模型）
  python run_with_class.py gsv      # GPT-SoVITS 后端（需先启动 api_v2.py 9880）
"""
from __future__ import annotations

import asyncio
import io
import json
import os
import random
import sys
from contextlib import asynccontextmanager
from pathlib import Path

ROOT = Path(__file__).resolve().parent
_venv_py = ROOT / ".venv" / "Scripts" / "python.exe"
if _venv_py.exists() and Path(sys.prefix).resolve() != _venv_py.parent.parent.resolve():
    os.execv(str(_venv_py), [str(_venv_py), __file__] + sys.argv[1:])

from shared import EMOTION_LABELS

# SBV2 需要从其目录运行
SBV2_DIR = ROOT / "Style-BERT-VITS2"
ASSETS_DIR = SBV2_DIR / "model_assets" / "eris"

_BACKEND_MODE = "gsv" if len(sys.argv) > 1 and sys.argv[1] == "gsv" else "sbv2"

if _BACKEND_MODE == "sbv2":
    os.chdir(SBV2_DIR)
    sys.path.insert(0, str(SBV2_DIR))

import httpx
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.responses import Response
from openai import OpenAI
from pydantic import BaseModel, Field

EMOTION_PARAMS: dict[str, dict] = {
    "neutral":   {"sdp_ratio": 0.2, "noise": 0.6, "noise_w": 0.8, "length": 1.1,  "style_weight": 1.0},
    "gentle":    {"sdp_ratio": 0.2, "noise": 0.4, "noise_w": 0.6, "length": 1.2,  "style_weight": 1.0},
    "serious":   {"sdp_ratio": 0.1, "noise": 0.4, "noise_w": 0.4, "length": 1.1,  "style_weight": 1.0},
    "confident": {"sdp_ratio": 0.3, "noise": 0.5, "noise_w": 0.7, "length": 1.05, "style_weight": 1.0},
    "surprised": {"sdp_ratio": 0.5, "noise": 0.8, "noise_w": 1.0, "length": 1.0,  "style_weight": 1.0},
    "happy":     {"sdp_ratio": 0.4, "noise": 0.7, "noise_w": 0.9, "length": 1.05, "style_weight": 1.0},
    "sad":       {"sdp_ratio": 0.3, "noise": 0.5, "noise_w": 0.6, "length": 1.25, "style_weight": 1.0},
}


# ── 配置 ──────────────────────────────────────────────────────────────────────

def load_config(path: Path) -> dict[str, str]:
    cfg: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if "=" in line:
            k, _, v = line.partition("=")
            cfg[k.strip()] = v.strip()
    return cfg


# ── DeepSeek 情绪检测 ────────────────────────────────────────────────────────

_SYSTEM_PROMPT = (
    "你是一个日语文本情绪分析助手。"
    "根据用户提供的文本，判断其情感倾向，从以下标签中选择最匹配的一个：\n"
    + ", ".join(EMOTION_LABELS) + "\n"
    "只输出标签名，不要输出任何其他文字。"
)


def detect_emotion(text: str, client: OpenAI, model: str) -> str:
    resp = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user", "content": f"「{text}」"},
        ],
        max_tokens=16,
        temperature=0.0,
    )
    emotion = resp.choices[0].message.content.strip().lower()
    if emotion not in EMOTION_PARAMS:
        emotion = "neutral"
    return emotion


# ── SBV2 直接合成 ────────────────────────────────────────────────────────────

_tts_model = None
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


def _get_model():
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


_SHORT_TEXT_THRESHOLD = 4  # 字符数 ≤ 此值视为极短文本


def _synthesize_sbv2(text: str, language: str, emotion: str,
                     speed: float | None = None,
                     style_weight: float | None = None) -> bytes:
    from style_bert_vits2.constants import Languages
    import numpy as np
    import scipy.io.wavfile as wavfile

    lang_map = {"ja": Languages.JP, "jp": Languages.JP, "en": Languages.EN, "zh": Languages.ZH}
    lang = lang_map.get(language.lower(), Languages.JP)
    params = EMOTION_PARAMS.get(emotion, EMOTION_PARAMS["neutral"])
    infer_length = speed if speed is not None else params["length"]
    infer_sw = style_weight if style_weight is not None else params["style_weight"]

    model = _get_model()

    # 极短文本降低随机性，让生成更稳定
    sdp = params["sdp_ratio"]
    noise = params["noise"]
    noise_w = params["noise_w"]
    if len(text) <= _SHORT_TEXT_THRESHOLD:
        sdp = min(sdp, 0.1)
        noise = min(noise, 0.3)
        noise_w = min(noise_w, 0.3)

    sr, audio = model.infer(
        text=text, language=lang, speaker_id=0, style=emotion,
        style_weight=infer_sw, sdp_ratio=sdp,
        noise=noise, noise_w=noise_w, length=infer_length,
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


# ── GPT-SoVITS 合成 ──────────────────────────────────────────────────────────

async def _synthesize_gsv(text: str, language: str, emotion: str,
                          gsv_url: str, emotion_map: dict, catalog: dict,
                          speed: float | None = None) -> bytes:
    candidates = emotion_map.get(emotion, emotion_map.get("neutral", []))
    if not candidates:
        raise ValueError(f"emotion_map 中无 {emotion} 对应的 voice_id")
    voice_id = random.choice(candidates)

    voice = catalog.get(voice_id)
    if not voice:
        raise ValueError(f"voice_id {voice_id} 不在 voices/ 中")

    cfg = voice["config"]
    payload = {
        "ref_audio_path": voice["reference_path"],
        "prompt_text": cfg.get("prompt_text", ""),
        "prompt_lang": cfg.get("prompt_language", "ja"),
        "text": text,
        "text_lang": language,
        "speed_factor": (1.0 / speed) if speed is not None else float(cfg.get("speed", 1.0)),
        "batch_size": 1,
        "media_type": "wav",
        "streaming_mode": False,
    }

    # 切换模型（如果 config 指定了权重）
    async with httpx.AsyncClient(timeout=120.0) as client:
        gpt_model = cfg.get("gpt_model")
        sovits_model = cfg.get("sovits_model")
        if gpt_model:
            await client.get(f"{gsv_url}/set_gpt_weights", params={"weights_path": gpt_model})
        if sovits_model:
            await client.get(f"{gsv_url}/set_sovits_weights", params={"weights_path": sovits_model})

        resp = await client.post(f"{gsv_url}/tts", json=payload)
        resp.raise_for_status()
        return resp.content


# ── FastAPI ──────────────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    cfg = load_config(ROOT / "run_with_class_config.txt")
    app.state.cfg = cfg
    app.state.backend = _BACKEND_MODE

    ds_key = cfg.get("deepseek_api_key")
    if not ds_key:
        print("[ERROR] run_with_class_config.txt 缺少 deepseek_api_key")
        sys.exit(1)
    app.state.ds_client = OpenAI(
        api_key=ds_key,
        base_url="https://api.deepseek.com",
    )

    if _BACKEND_MODE == "sbv2":
        global _sbv2_model_name
        _sbv2_model_name = cfg.get("sbv2_model") or None
        print("后端: SBV2 (直接加载模型)")
        print("预加载模型...")
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, _get_model)
        print("模型加载完成，预热 BERT...")
        await loop.run_in_executor(None, _synthesize_sbv2, "テスト", "ja", "neutral")
        print("BERT 预热完成")
    else:
        from src.voice_catalog import load_catalog
        app.state.catalog = load_catalog()
        map_path = ROOT / "classification" / "emotion_map.json"
        if not map_path.exists():
            print(f"[ERROR] {map_path} 不存在，请先运行 class_annotate.py")
            sys.exit(1)
        app.state.emotion_map = json.loads(map_path.read_text(encoding="utf-8"))
        gsv_url = cfg.get("gsv_api_url", "http://127.0.0.1:9880")
        print(f"后端: GPT-SoVITS ({gsv_url})")
        print(f"音色: {len(app.state.catalog)} 个")

    print(f"情绪标签: {EMOTION_LABELS}")
    yield


app = FastAPI(title="Emotion-Routed TTS", lifespan=lifespan)


class TTSRequest(BaseModel):
    text: str = Field(..., min_length=1)
    language: str = Field("ja", description="语言: ja/en/zh")
    emotion: str | None = Field(None, description="情绪标签；不填则由 DeepSeek 自动检测")
    speed: float | None = Field(None, ge=0.5, le=2.0, description="语速（越大越慢，默认按情绪预设，参考 0.8-1.5）")
    style_weight: float | None = Field(None, ge=0.0, le=5.0, description="风格强度（越大越夸张，默认 1.0，推荐 0.5-1.5）")


@app.post("/tts")
async def tts_post(request: TTSRequest):
    return await _handle(request.text, request.language, request.emotion,
                         request.speed, request.style_weight)


@app.get("/tts")
async def tts_get(text: str = "", language: str = "ja", emotion: str = "",
                  speed: float | None = None, style_weight: float | None = None):
    if not text:
        raise HTTPException(status_code=400, detail="需要 query 参数: text")
    if speed is not None:
        speed = max(0.5, min(2.0, speed))
    if style_weight is not None:
        style_weight = max(0.0, min(5.0, style_weight))
    return await _handle(text, language, emotion or None, speed, style_weight)


async def _handle(text: str, language: str, emotion: str | None,
                  speed: float | None = None, style_weight: float | None = None) -> Response:
    text = text.strip()
    if not text:
        raise HTTPException(status_code=400, detail="text 不能为空")

    cfg: dict = app.state.cfg
    ds_client: OpenAI = app.state.ds_client

    # 情绪检测
    if emotion:
        emotion = emotion.lower()
        if emotion not in EMOTION_PARAMS:
            raise HTTPException(status_code=400, detail=f"无效情绪: {emotion}，可选: {EMOTION_LABELS}")
        print(f"[manual] {text!r} → {emotion}")
    else:
        try:
            emotion = detect_emotion(text, ds_client, cfg.get("deepseek_model", "deepseek-chat"))
        except Exception as e:
            print(f"[warn] DeepSeek 失败，使用 neutral: {e}")
            emotion = "neutral"
        print(f"[auto]  {text!r} → {emotion}")

    # 合成
    try:
        if app.state.backend == "sbv2":
            loop = asyncio.get_event_loop()
            wav = await loop.run_in_executor(
                None, _synthesize_sbv2, text, language, emotion, speed, style_weight)
        else:
            gsv_url = cfg.get("gsv_api_url", "http://127.0.0.1:9880")
            wav = await _synthesize_gsv(text, language, emotion,
                                        gsv_url, app.state.emotion_map, app.state.catalog,
                                        speed)
    except httpx.ConnectError:
        raise HTTPException(status_code=503, detail="无法连接 GPT-SoVITS，请确认 api_v2.py 已启动")
    except httpx.HTTPStatusError as e:
        raise HTTPException(status_code=502, detail=f"后端错误: {e.response.text}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"合成失败: {e}")

    return Response(content=wav, media_type="audio/wav")


@app.get("/emotions")
async def list_emotions():
    return {"emotions": EMOTION_LABELS, "backend": app.state.backend}


@app.get("/health")
async def health():
    return {"status": "ok", "backend": app.state.backend}


# ── 启动 ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    cfg = load_config(ROOT / "run_with_class_config.txt")
    port = int(cfg.get("port", 8092))
    uvicorn.run("run_with_class:app", host="0.0.0.0", port=port, reload=False)
