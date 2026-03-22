# Tasks: GPT-SoVITS 音色克隆与 TTS API

## 1. 环境与依赖

- [x] 1.1 创建项目结构（src/、voices/、requirements.txt）
- [x] 1.2 安装 GPT-SoVITS 及 PyTorch（CUDA 12.x）
- [x] 1.3 添加 FastAPI、uvicorn 等 API 依赖

## 2. 音色目录

- [x] 2.1 定义 voices/<voice_id>/ 目录结构及 config.json schema
- [x] 2.2 实现音色目录扫描与加载逻辑（voice-catalog）
- [x] 2.3 创建 voices/eris 示例目录及占位 config.json

## 3. GPT-SoVITS 集成

- [x] 3.1 集成 GPT-SoVITS 推理（调用官方 API 或嵌入）
- [x] 3.2 实现参考音频预处理（静音切除、音量标准化）
- [x] 3.3 配置 4GB 显存优化（batch_size=1、is_half=true、分段生成）

## 4. TTS API

- [x] 4.1 创建 FastAPI 应用及 POST /tts 端点
- [x] 4.2 实现 text、voice_id 入参校验及 4xx 错误返回
- [x] 4.3 实现合成调用与 WAV 音频返回

## 5. 部署与文档

- [x] 5.1 编写 README（环境要求、启动命令、素材规范）
- [x] 5.2 提供艾莉丝参考音频准备说明及预处理脚本用法
