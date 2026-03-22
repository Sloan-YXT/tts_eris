# Spec: TTS API

## ADDED Requirements

### Requirement: 文本转语音端点

系统 SHALL 提供 HTTP 端点，接收文本与音色标识，返回对应音色的语音音频。

#### Scenario: 成功合成并返回音频

- **WHEN** 客户端发送有效请求（text + voice_id）
- **THEN** 系统返回 200 及音频数据（Content-Type 为 audio/*）

#### Scenario: 无效音色返回错误

- **WHEN** 客户端指定不存在的 voice_id
- **THEN** 系统 SHALL 返回 4xx 错误及明确错误信息

#### Scenario: 空文本返回错误

- **WHEN** 客户端发送空文本
- **THEN** 系统 SHALL 返回 4xx 错误

### Requirement: 音色选择

系统 SHALL 支持通过 voice_id 参数选择已注册音色。

#### Scenario: 指定音色合成

- **WHEN** 请求包含 voice_id=eris
- **THEN** 系统使用 eris 对应参考音频进行合成

### Requirement: 音频格式

系统 SHALL 默认返回 WAV 格式音频；SHALL 支持通过参数指定输出格式（若实现）。

#### Scenario: 默认 WAV 返回

- **WHEN** 客户端未指定格式
- **THEN** 系统返回 WAV 格式音频
