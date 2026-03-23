import os
cache = r"F:\programs\python\play\python\tts\tts_eris\GPT-SoVITS\GPT_SoVITS\pretrained_models\fast_langdetect"
os.makedirs(cache, exist_ok=True)
os.environ["FTLANG_CACHE"] = cache
from fast_langdetect import detect
print("ok:", detect("hello world"))
