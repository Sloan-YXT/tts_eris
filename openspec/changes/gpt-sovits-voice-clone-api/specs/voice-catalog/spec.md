# Spec: Voice Catalog

## ADDED Requirements

### Requirement: 音色目录结构

系统 SHALL 使用按音色 ID 分目录的结构：每个音色对应 `voices/<voice_id>/`，内含 `reference.wav` 与 `config.json`。

#### Scenario: 有效音色目录被加载

- **WHEN** `voices/eris/` 存在且含 `reference.wav` 与 `config.json`
- **THEN** 系统将 eris 注册为可用音色

#### Scenario: 缺少必要文件被跳过

- **WHEN** 某音色目录缺少 reference.wav
- **THEN** 系统 SHALL 跳过该音色或报错，不将其注册为可用

### Requirement: 音色配置

每个音色的 config.json SHALL 支持语速、参考文本等可调参数；具体字段由实现定义。

#### Scenario: 配置覆盖默认参数

- **WHEN** config.json 中指定语速
- **THEN** 合成时使用该语速

### Requirement: 多音色扩展

新增音色 SHALL 仅需在 voices 下添加新目录（含 reference.wav 与 config.json），无需修改代码。

#### Scenario: 新增音色生效

- **WHEN** 用户在 voices/ 下新增 roxy/ 目录并放入必要文件
- **THEN** 系统在启动或重载后识别 roxy 为可用音色
