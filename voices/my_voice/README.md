# 自定义音色示例（my_voice）

1. 将约 5–10 秒的参考语音放到本目录，命名为 **reference.wav**（WAV，16kHz 或 24kHz、16bit，环境尽量安静）。
2. 编辑 **config.json**，把 `prompt_text` 改成 reference.wav 里说的那句话的原文；按需修改 `prompt_language`、`text_language`。
3. 先启动 GPT-SoVITS 推理服务（见项目根目录 README），再在本项目根目录执行 `python run.py`。
4. 请求：`GET http://localhost:8000/tts?text=要合成的内容&voice_id=my_voice` 即可获取语音。
