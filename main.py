#!/usr/bin/env python3
"""
DocFlow 入口。

用法：
  # 启动 Web 服务（含文件夹监控）
  python main.py serve

  # 手动 ingest 单个文件
  python main.py ingest /path/to/file.pdf

  # 扫描所有 watch_dirs（config.yaml）
  python main.py scan
"""

import sys
import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s  %(name)s  %(message)s",
    datefmt="%H:%M:%S",
)


def serve():
    import uvicorn
    uvicorn.run("src.api.app:app", host="0.0.0.0", port=8000, reload=False)


def ingest(path: str):
    from src.ingest.pipeline import IngestPipeline
    pipeline = IngestPipeline.from_config("config.yaml")
    result = pipeline.ingest(path)
    print(result)


def scan():
    import yaml
    from src.ingest.pipeline import IngestPipeline
    from src.api.app import _parse_watch_dirs
    with open("config.yaml") as f:
        cfg = yaml.safe_load(f)
    pipeline = IngestPipeline.from_config("config.yaml")
    for wd in _parse_watch_dirs(cfg):
        exts = wd.extensions if wd.extensions else pipeline.registry.supported_extensions
        for ext in exts:
            pattern = f"**/*{ext}" if wd.recursive else f"*{ext}"
            for path in wd.path.glob(pattern):
                print(pipeline.ingest(path))


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "serve"
    if cmd == "serve":
        serve()
    elif cmd == "ingest":
        if len(sys.argv) < 3:
            print("Usage: python main.py ingest <path>")
            sys.exit(1)
        ingest(sys.argv[2])
    elif cmd == "scan":
        scan()
    else:
        print(f"Unknown command: {cmd}")
        sys.exit(1)
