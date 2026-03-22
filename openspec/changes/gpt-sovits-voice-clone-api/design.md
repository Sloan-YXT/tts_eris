# Design: GPT-SoVITS 音色克隆与 TTS API

## Context

- **目标**：基于 GPT-SoVITS 实现动漫角色音色克隆，提供 text → 语音的 HTTP API，并支持多音色扩展
- **首例音色**：艾莉丝·伯雷亚斯·格雷拉特（无职转生）
- **硬件约束**：RTX 3050 4GB 显存，CUDA 12.x；推理需优化（batch_size=1、半精度等）
- **现状**：项目为空，需从零搭建

## Goals / Non-Goals

**Goals:**
- 集成 GPT-SoVITS，支持零样本音色克隆
- 提供 REST API：`POST /tts`，输入 text + voice_id，返回音频
- 定义音色目录结构，便于新增角色
- 支持本地与服务器部署，含 4GB 显存优化说明

**Non-Goals:**
- 不实现微调训练（4GB 显存不足）
- 不实现流式输出（可后续扩展）
- 不提供 Web UI（仅 API）

## Decisions

### 1. GPT-SoVITS 集成方式

**选择**：使用官方 Python API（`api_v2.py` 或等价接口），以子进程或 HTTP 代理方式调用。

**备选**：直接 fork 官方仓库并嵌入；Rust EchoKit 服务。  
**理由**：官方 API 成熟、文档全，便于维护；Rust 方案需额外构建链。

### 2. API 框架

**选择**：FastAPI，提供 `/tts` 等端点。

**备选**：Flask、纯 asyncio。  
**理由**：FastAPI 异步友好、自动 OpenAPI、类型提示好，适合 TTS 这类 I/O 密集场景。

### 3. 音色目录结构

**选择**：按音色 ID 分目录，每目录含 `reference.wav` 与 `config.json`：

```
voices/
  eris/
    reference.wav
    config.json   # 语速、参考文本等
  roxy/
    ...
```

**备选**：单 YAML 配置 + 音频路径列表。  
**理由**：目录结构直观，新增音色只需新增目录，便于脚本化。

### 4. 4GB 显存优化策略

**选择**：
- `batch_size=1`
- `is_half=true`（半精度）
- 禁用 BERT 或使用轻量文本前端（若支持）
- 长文本分段生成

**理由**：社区反馈 4GB 可跑推理，但需上述优化；微调暂不实现。

### 5. 部署形态

**选择**：单进程，FastAPI 直接调用 GPT-SoVITS 推理逻辑（或通过本地 HTTP 代理）。

**备选**：Docker 多容器。  
**理由**：首版简化部署；Docker 可后续补充。

## Risks / Trade-offs

| 风险 | 缓解 |
|------|------|
| 4GB 显存 OOM | 半精度、batch=1、分段生成；必要时提供 CPU 推理说明 |
| BERT 提取慢（3–10 秒） | 文档说明预期延迟；可选禁用 BERT（若模型支持） |
| 参考音频质量影响效果 | 提供素材规范（格式、时长、降噪）与预处理脚本 |
| 多音色并发显存压力 | 单请求串行；多实例需更多显存 |

## Migration Plan

1. 安装 GPT-SoVITS 及依赖
2. 准备艾莉丝参考音频，放入 `voices/eris/`
3. 启动 API 服务，验证 `/tts?voice=eris&text=...`
4. 按需新增音色目录

**回滚**：停止服务即可，无持久状态。

## Open Questions

- 是否支持跨语言（如日文参考 → 中文合成）？GPT-SoVITS 支持，可留作配置项
- 音频格式：WAV / MP3 / 其他？建议默认 WAV，可加格式参数
