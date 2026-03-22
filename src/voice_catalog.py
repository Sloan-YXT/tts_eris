"""音色目录扫描与加载。"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

VOICES_DIR = Path(__file__).resolve().parent.parent / "voices"
CONFIG_FILE = "config.json"
REFERENCE_FILE = "reference.wav"


def load_catalog(voices_dir: Path | None = None) -> dict[str, dict[str, Any]]:
    """
    扫描 voices 目录，加载所有有效音色。
    有效音色：目录内存在 reference.wav 与 config.json。
    """
    base = voices_dir or VOICES_DIR
    catalog: dict[str, dict[str, Any]] = {}

    if not base.exists():
        return catalog

    for item in base.iterdir():
        if not item.is_dir():
            continue
        voice_id = item.name
        ref_path = item / REFERENCE_FILE
        config_path = item / CONFIG_FILE

        if not ref_path.exists():
            continue  # 缺少 reference.wav，跳过
        if not config_path.exists():
            continue  # 缺少 config.json，跳过

        try:
            with open(config_path, encoding="utf-8") as f:
                config = json.load(f)
        except (json.JSONDecodeError, OSError):
            continue

        catalog[voice_id] = {
            "voice_id": voice_id,
            "reference_path": str(ref_path),
            "config": config,
        }

    return catalog
