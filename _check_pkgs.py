import importlib
pkgs = [
    'torchcodec', 'faster_whisper', 'einops', 'gruut', 'jamo',
    'ko_pron', 'inflect', 'auraloss', 'ToJyutping', 'g2p_en',
    'g2pk2', 'pandas', 'wordsegment', 'fast_langdetect',
    'split_lang', 'rotary_embedding_torch', 'x_transformers',
    'peft', 'funasr', 'cn2an', 'pypinyin', 'jieba_fast',
    'sentencepiece', 'ctranslate2', 'av', 'pyopenjtalk',
    'gradio', 'opencc', 'psutil', 'yaml', 'scipy', 'librosa',
    'soundfile', 'tqdm', 'numpy', 'torch', 'torchaudio',
]
for p in pkgs:
    try:
        m = importlib.import_module(p.replace('-','_'))
        v = getattr(m, '__version__', 'ok')
        print(f'  OK   {p} ({v})')
    except ImportError as e:
        print(f'  MISS {p}')
