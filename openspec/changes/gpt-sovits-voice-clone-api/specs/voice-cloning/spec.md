# Spec: Voice Cloning

## ADDED Requirements

### Requirement: 参考音频格式规范

系统 SHALL 支持符合以下规范的参考音频：WAV 格式，16kHz 或 24kHz 采样率，16bit 位深，时长 5–60 秒。

#### Scenario: 有效参考音频被接受

- **WHEN** 用户提供符合规范的 WAV 文件作为参考
- **THEN** 系统接受该文件并可用于零样本克隆

#### Scenario: 过短音频被拒绝

- **WHEN** 参考音频时长少于 5 秒
- **THEN** 系统 SHALL 拒绝或提示用户补充素材

### Requirement: 零样本克隆流程

系统 SHALL 基于 GPT-SoVITS 零样本能力，从参考音频提取音色特征并用于合成，无需微调训练。

#### Scenario: 零样本克隆成功

- **WHEN** 用户提供有效参考音频并请求合成
- **THEN** 系统使用该音频作为音色参考，生成目标文本的语音

#### Scenario: 无微调训练

- **WHEN** 用户仅提供参考音频
- **THEN** 系统 SHALL 不执行模型微调，仅使用零样本推理

### Requirement: 参考音频预处理

系统 SHALL 支持对参考音频进行预处理：静音切除、音量标准化；可选降噪。

#### Scenario: 预处理后音频可用

- **WHEN** 用户提供含静音或音量不均的参考音频
- **THEN** 系统 SHALL 提供预处理能力，使输出符合 GPT-SoVITS 输入要求
