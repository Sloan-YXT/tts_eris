# GPT-SoVITS 音色克隆 TTS API

基于 GPT-SoVITS 的文本转语音 API，支持零样本音色克隆，可扩展多角色音色。

## 环境要求

- Python 3.10+
- **ffmpeg**：视频转音频需安装。可运行 `python scripts/setup_ffmpeg.py` 下载到项目 `ffmpeg/` 目录
- **GPT-SoVITS**：需单独安装并启动 API 服务（见下方）
- 推荐 GPU：6GB+ 显存；4GB 需按说明优化

### 安装 GPT-SoVITS

1. 克隆官方仓库：
   ```bash
   git clone https://github.com/RVC-Boss/GPT-SoVITS.git
   cd GPT-SoVITS
   ```

2. 安装依赖（含 PyTorch CUDA 12.x）：
   ```bash
   pip install -r requirements.txt
   # 或按官方文档安装 PyTorch
   ```

3. 下载预训练模型到 `GPT_SoVITS/pretrained_models/`：
   ```bash
   # 在 tts 项目下运行（将路径改为你的 GPT-SoVITS 克隆位置）
   python scripts/setup_gpt_sovits_models.py D:\GPT-SoVITS
   ```
   或手动从 [HuggingFace](https://huggingface.co/lj1995/GPT-SoVITS) 下载以下内容到 `GPT_SoVITS/pretrained_models/`：
   - `chinese-roberta-wwm-ext-large`（BERT 中文文本编码，约 620MB）
   - `chinese-hubert-base`（HuBERT 语音编码，约 180MB）
   - `gsv-v2final-pretrained`（v2 预训练，含 s1/s2 模型）

4. 4GB 显存优化：在 `configs/tts_infer.yaml` 中设置：
   - `batch_size: 1`
   - `is_half: true`
   - `text_split_method: cut5`

5. 启动 GPT-SoVITS 推理 API（二选一）：
   - **本仓库内一键安装**：进入 `GPT-SoVITS` 目录，双击 `run_api.bat`（会自动用 conda 环境并监听 9880）
   - **手动**：在 GPT-SoVITS 根目录下，用其 conda 环境执行：
     ```bash
     python api_v2.py -a 127.0.0.1 -p 9880 -c GPT_SoVITS/configs/tts_infer.yaml
     ```

### 安装本 TTS API

```bash
cd tts
pip install -r requirements.txt
python scripts/setup_ffmpeg.py   # 下载 ffmpeg 到项目 ffmpeg/ 目录
```

## 启动

1. 确保 GPT-SoVITS API 已在 9880 端口运行
2. 启动本服务：
   ```bash
   python run.py
   ```
   或：`uvicorn src.main:app --host 0.0.0.0 --port 8000`

3. 访问 http://localhost:8000/docs 查看 API 文档

## 用「配置文件 + 推理 API」按请求获取语音

按下面四步即可：填好配置、启动推理服务，通过 HTTP 请求拿语音。

1. **新建一个音色目录**（例如 `voices/my_voice`），放入：
   - **reference.wav**：约 5–10 秒的参考语音（WAV，16kHz/24kHz、16bit，尽量安静）
   - **config.json**：内容示例：
     ```json
     {
       "prompt_text": "这里填参考音频里说的那句话的原文",
       "prompt_language": "zh",
       "text_language": "zh",
       "speed": 1.0
     }
     ```
     - `prompt_text`：参考音频的转写/原文（需与 reference.wav 内容一致）
     - `prompt_language` / `text_language`：`zh` 中文、`ja` 日语、`en` 英语等

2. **启动 GPT-SoVITS 推理服务**（9880）  
   进入 `GPT-SoVITS` 目录，双击 `run_api.bat`；或在该目录下用其 conda 环境执行：
   ```bash
   python api_v2.py -a 127.0.0.1 -p 9880 -c GPT_SoVITS/configs/tts_infer.yaml
   ```

3. **启动本 TTS 服务**（8000）  
   在**本仓库根目录**（tts）执行：
   ```bash
   python run.py
   ```

4. **按请求获取语音**  
   - 浏览器或 GET：`http://localhost:8000/tts?text=要合成的内容&voice_id=my_voice`  
   - 或 POST：`POST /tts`，Body：`{"text": "要合成的内容", "voice_id": "my_voice"}`  
   返回为 WAV 音频流。

## 素材规范（艾莉丝等角色）

### 参考音频要求

- 格式：WAV
- 采样率：16kHz 或 24kHz
- 位深：16bit
- 时长：5–60 秒（零样本推荐 5–10 秒）
- 环境：尽量安静，背景噪音小

### 视频转音频 Helper

从链接或本地视频提取音频，生成 MP3 和 GPT-SoVITS 所需的 WAV：

```bash
# 首次使用可运行: python scripts/setup_ffmpeg.py 下载 ffmpeg 到项目目录
# 从 URL 下载并提取
python scripts/video_to_audio.py "https://www.bilibili.com/video/xxx" -o output/

# 从本地视频提取
python scripts/video_to_audio.py path/to/video.mp4 -o output/

# 仅生成 WAV，并截取前 10 秒（适合做参考音频）
python scripts/video_to_audio.py video.mp4 --wav-only --max-duration 10 -o voices/eris/
```

输出 WAV 为 24kHz、16bit、单声道，符合 GPT-SoVITS 参考音频要求。

### WAV 按时间范围剪辑

指定起止时间截取音频片段（支持 WAV/MP4/MP3 输入）：

```bash
python scripts/clip_wav.py 输入.wav -s 1 -e 10 -o voices/eris/reference.wav
# -s/--start: 起始秒数
# -e/--end: 结束秒数
# -o/--output: 输出路径
```

### 艾莉丝音色准备

1. 从动画《无职转生》中提取艾莉丝·伯雷亚斯·格雷拉特的对白片段（可用上述 video_to_audio 从视频提取）
2. 使用剪辑软件或 ffmpeg 导出为 WAV
3. （可选）预处理：静音切除、音量标准化：
   ```bash
   python scripts/preprocess_audio.py path/to/raw.wav -o voices/eris/reference.wav
   ```
4. 将文件保存为 `voices/eris/reference.wav`
5. 编辑 `voices/eris/config.json`，填写 `prompt_text`（参考音频的转写文本）。艾莉丝为日语角色，`prompt_language` 与 `text_language` 已设为 `ja`

### 预处理脚本用法

```bash
python scripts/preprocess_audio.py 输入.wav -o 输出.wav
# 可选参数：
#   --top-db 25    静音切除阈值（默认 25）
#   --target-db -20  目标音量 dB（默认 -20）
```

## API 使用

### POST /tts

```json
{
  "text": "要合成的文本（日语角色请用日文）",
  "voice_id": "eris"
}
```

返回：WAV 音频流（`Content-Type: audio/wav`）

### GET /tts

```
GET /tts?text=要合成的文本&voice_id=eris
```

### 环境变量

- `GPT_SOVITS_API_URL`：GPT-SoVITS API 地址，默认 `http://127.0.0.1:9880`

## 新增音色

在 `voices/` 下新建目录，例如 `voices/roxy/`：

1. 放入 `reference.wav`
2. 创建 `config.json`：
   ```json
   {
     "prompt_text": "参考音频的转写文本",
     "prompt_language": "ja",
     "text_language": "ja",
     "speed": 1.0
   }
   ```
   （日语角色用 `ja`，中文用 `zh`，英文用 `en`）
3. 重启服务或重载目录

## 目录结构

```
tts/
├── src/
│   ├── main.py           # FastAPI 入口
│   ├── voice_catalog.py  # 音色目录加载
│   ├── gpt_sovits_client.py
│   └── audio_preprocess.py
├── voices/
│   ├── eris/
│   │   ├── reference.wav
│   │   └── config.json
│   └── ...
├── scripts/
│   └── preprocess_audio.py
├── requirements.txt
└── run.py
```
