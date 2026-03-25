# Eris TTS — 多风格语音合成

支持两种后端：**Style-BERT-VITS2 (SBV2)** 和 **GPT-SoVITS**。

## 前提

- Windows 10/11, Python 3.10, NVIDIA GPU (VRAM >= 8GB)
- [Visual C++ Redistributable 2015–2022 x64](https://aka.ms/vs/17/release/vc_redist.x64.exe)
- Conda（pyopenjtalk 编译需要 MinGW 工具链）

## 1. 安装

```bash
git clone https://github.com/litagin02/Style-BERT-VITS2.git
git clone https://github.com/RVC-Boss/GPT-SoVITS.git
python install.py
```

`install.py` 一次性安装**两种模式的全部依赖**：

1. 创建 `.venv`（自动定位 Python 3.10）
2. PyTorch 2.10 + CUDA 12.8
3. GPT-SoVITS 全部依赖（含 pyopenjtalk MinGW 编译、ffmpeg DLL、预训练模型）
4. SBV2 全部依赖（BERT 模型、JP-Extra 底模、SLM 模型）

首次运行总下载约 8-10GB。重复运行安全（已有文件跳过）。

## 2. 数据准备

用 GPT-SoVITS WebUI 完成音频切片和 ASR 标注：

```bash
cd GPT-SoVITS
..\.venv\Scripts\python webui.py
```

在 WebUI 中依次完成：

1. **UVR5 人声分离** — 模型 `HP5_only_main_vocal` 去背景音乐，再用 `VR-DeEchoAggressive` 去混响
2. **音频切片** — 阈值 -34dB，最小片段 4000ms，输出到 `output/slicer_opt/`
3. **ASR 标注** — `faster-whisper` + `large-v3`，语言 `ja`，输出 `output/asr_opt/*.list`
4. **清理标注** — 删除 `instrument_` 开头的行（乐器轨道）

切片结果存入 `voices/eris_avl_*/`，每个目录包含 `reference.wav` + `config.json`（含 `prompt_text` 台词）。

---

## SBV2 模式

### 训练

两种训练模式：

```bash
python train_all.py           # 多风格训练（7 种情绪，需要 Gemini API）
python train_all.py --basic   # 单风格训练（音色更自然，推荐首次使用）
```

**训练标志：**

| 标志 | 作用 |
|------|------|
| `--basic` | 单风格训练（所有数据训一个风格，音色更自然） |
| `--force` | 清理已有 checkpoint，从头训练 |
| `--reannotate` | 重新进行 Gemini 质量/情绪标注（忽略已有结果） |

组合示例：

```bash
python train_all.py --basic              # 单风格，续训已有 checkpoint
python train_all.py --basic --force      # 单风格，从头训练
python train_all.py --force              # 多风格，从头训练
python train_all.py --force --reannotate # 多风格，从头训练 + 重新标注
```

无 `--force` 时，若存在已有 checkpoint 则自动从断点续训（适用于训练中断恢复）。

**质量过滤：** 所有模式都会调用 Gemini 对音频质量评分（1-10），低于阈值（`credentials.txt` 中 `min_quality`，默认 8）的切片自动排除。已评分的切片不会重复评分。

**多风格训练**额外需要 `credentials.txt` 配置 Gemini API Key（自动情绪标注）：

```json
{"gemini": {"api_key": "AIza...", "model": "models/gemini-2.5-flash-lite"}, "min_quality": 8}
```

多风格自动完成 6 步：Gemini 标注 → 数据整理 → 预处理（重采样 + BERT 特征）→ 按情绪分组 → 多风格预处理 → 训练（~40 分钟）。

单风格（`--basic`）4 步：Gemini 质量评分 → 数据整理 → 预处理 → 训练。

### 部署

```bash
python run.py                      # 手动指定风格，端口 8010
python run_with_class.py           # 自动情绪检测，端口 8092
```

两个服务独立运行，各自直接加载模型，无外部依赖。首次启动加载 BERT 模型约 2 分钟。

`run_with_class.py` 需在 `run_with_class_config.txt` 中配置 `deepseek_api_key`。

### 配置

`run_with_class_config.txt`：

```ini
deepseek_api_key = sk-xxxxxxxx
deepseek_model = deepseek-chat
port = 8092
sbv2_api_url = http://127.0.0.1:8010
gsv_api_url = http://127.0.0.1:9880
# SBV2 模型文件名（留空则自动选最新的）
sbv2_model = eris_e60_s5880.safetensors
# 纯净模式：忽略所有调音参数，只用模型默认值（单风格推荐开启）
pure_mode = false
```

### 请求

```bash
# run.py — 手动指定风格
curl -X POST http://localhost:8010/tts \
  -H "Content-Type: application/json" \
  -d '{"text": "どうして私を置いていくの？", "style": "sad"}' \
  -o output.wav

# run_with_class.py — 自动检测情绪
curl -X POST http://localhost:8092/tts \
  -H "Content-Type: application/json" \
  -d '{"text": "やったー！"}' \
  -o output.wav

# 手动指定情绪 + 可选参数
curl -X POST http://localhost:8092/tts \
  -H "Content-Type: application/json" \
  -d '{"text": "やったー！", "emotion": "happy", "speed": 1.0, "style_weight": 1.0}' \
  -o output.wav

# GET 方式
curl "http://localhost:8092/tts?text=hello&emotion=neutral" -o output.wav
```

**API 参数：**

| 参数 | 说明 | 默认 | 范围 |
|------|------|------|------|
| `text` | 合成文本（必填） | - | - |
| `language` | 语言 | `ja` | `ja`/`en`/`zh` |
| `emotion` | 情绪标签（不填则自动检测） | auto | `neutral`/`gentle`/`serious`/`confident`/`surprised`/`happy`/`sad` |
| `speed` | 语速（越大越慢） | 按情绪预设 | 0.5 - 2.0 |
| `style_weight` | 风格强度（越大越夸张） | 1.0 | 0.0 - 5.0 |

> **纯净模式**（`pure_mode = true`）：忽略 emotion/speed/style_weight，只用模型默认参数。单风格训练后推荐开启，保持音色最自然。

---

## GPT-SoVITS 模式

### Windows 训练补丁

GPT-SoVITS 在 Windows 单 GPU 下训练需要手动修改以下文件（gloo 后端不可用）：

**`GPT_SoVITS/s2_train.py`** — gloo 无法解析主机名 + 单 GPU DDP 问题：
- `MASTER_ADDR` 改为 `"127.0.0.1"`
- 单 GPU (`n_gpus == 1`) 时直接调用 `run(0, 1, hps)` 跳过 `mp.spawn`
- `dist.init_process_group()` 仅在 `n_gpus > 1` 时调用
- DDP 包装仅在 `n_gpus > 1` 时启用
- `generator.module.infer(...)` 改用 `hasattr` 判断是否被 DDP 包装

**`GPT_SoVITS/s1_train.py`** — gloo + DDP 策略：
- `MASTER_ADDR` 改为 `"127.0.0.1"`
- Trainer 改为 `devices=1, strategy="auto"`

**`GPT_SoVITS/AR/data/bucket_sampler.py`** — dist 未初始化：
- `dist.get_world_size()` 和 `dist.get_rank()` 前加 `dist.is_initialized()` 判断

### 训练

```bash
python train.py
```

自动完成：合并标注 → 文本分词 → HuBERT 特征 → 语义 Token → SoVITS 训练 → GPT 训练。

训练完成后在 `voices/eris/config.json` 中填入模型路径：

```json
{
  "prompt_text": "台词原文",
  "prompt_language": "ja",
  "text_language": "ja",
  "speed": 1.0,
  "gpt_model": "GPT_weights_v2Pro/eris-e50.ckpt",
  "sovits_model": "SoVITS_weights_v2Pro/eris_e20_s1520.pth"
}
```

### 部署

先启动 GPT-SoVITS 推理后端，再启动本项目服务：

```bash
# 终端 1 — GPT-SoVITS 推理引擎
cd GPT-SoVITS
..\.venv\Scripts\python api_v2.py -a 127.0.0.1 -p 9880

# 终端 2 — 本项目服务
python run.py gsv                  # 手动指定音色，端口 8010
python run_with_class.py gsv       # 自动情绪检测，端口 8092
```

`run_with_class.py gsv` 通过 `classification/emotion_map.json` 将情绪映射到对应的 voice_id。

### 请求

```bash
# run.py gsv — 指定音色
curl -X POST http://localhost:8010/tts \
  -H "Content-Type: application/json" \
  -d '{"text": "どうして私を置いていくの？", "voice_id": "eris_avl_006"}' \
  -o output.wav

# run_with_class.py gsv — 自动检测情绪 → 选择音色
curl -X POST http://localhost:8092/tts \
  -H "Content-Type: application/json" \
  -d '{"text": "やったー！"}' \
  -o output.wav
```
