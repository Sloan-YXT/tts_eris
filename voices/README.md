# 音色目录

每个音色对应一个子目录 `voices/<voice_id>/`，需包含：

- `reference.wav`：参考音频（WAV，16kHz 或 24kHz，16bit，5–60 秒）
- `config.json`：音色配置

## config.json schema

```json
{
  "prompt_text": "参考音频对应的文本（与 reference.wav 内容一致）",
  "prompt_language": "zh",
  "text_language": "zh",
  "speed": 1.0
}
```

| 字段 | 类型 | 说明 |
|------|------|------|
| prompt_text | string | 参考音频的转写文本 |
| prompt_language | string | 参考音频语言：zh, ja, en 等 |
| text_language | string | 合成目标语言 |
| speed | number | 语速，默认 1.0 |
