#!/usr/bin/env python3
"""启动 TTS API 服务。"""
import uvicorn

if __name__ == "__main__":
    uvicorn.run(
        "src.main:app",
        host="0.0.0.0",
        port=8010,
        reload=True,
    )
