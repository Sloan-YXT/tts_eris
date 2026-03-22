# Proposal: GPT-SoVITS 音色克隆与 TTS API

## Why

需要一套本地可用的语音克隆与合成方案：以动漫角色（如无职转生中的艾莉丝）为目标音色，通过 GPT-SoVITS 进行零样本克隆，并提供 HTTP API 将任意文本转换为该音色的语音。同时希望流程可标准化，便于后续添加更多角色音色。

## What Changes

- **音色克隆**：基于 GPT-SoVITS 零样本能力，从参考音频克隆目标音色（首例：艾莉丝·伯雷亚斯·格雷拉特）
- **TTS API 服务**：提供 HTTP 接口，接收文本和音色标识，返回对应音色的语音音频
- **音色目录**：定义统一的音色配置结构（参考音频、参数等），支持多音色切换与扩展
- **部署与运行**：支持本地（RTX 3050 4GB）或服务器部署，含环境与依赖说明

## Capabilities

### New Capabilities

- `voice-cloning`: 参考音频准备、预处理与 GPT-SoVITS 零样本克隆流程
- `tts-api`: 文本转语音的 HTTP API，支持音色选择与音频返回
- `voice-catalog`: 标准化音色目录结构，支持多音色注册与配置管理

### Modified Capabilities

- （无）

## Impact

- **依赖**：GPT-SoVITS、Python 3.10+、PyTorch（CUDA 12.x）、NVIDIA GPU（推荐 6GB+，4GB 需优化）
- **新增代码**：API 服务、音色配置加载、与 GPT-SoVITS 的集成逻辑
- **部署**：本地或服务器运行，需考虑显存与推理延迟（约 10–20 秒/10 秒语音）
