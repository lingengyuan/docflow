#!/usr/bin/env python3
"""
DocFlow 入口。

用法：
  # 启动 Web 服务（含文件夹监控）
  python main.py serve

  # 手动 ingest 单个文件
  python main.py ingest /path/to/file.pdf

  # dry-run benchmark 一个或多个文件
  python main.py benchmark /path/to/file1.md /path/to/file2.pdf

  # 扫描所有 watch_dirs（config.yaml）
  python main.py scan
"""

import json
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
    from src.ingest.watcher import _is_excluded
    with open("config.yaml") as f:
        cfg = yaml.safe_load(f)
    pipeline = IngestPipeline.from_config("config.yaml")
    for wd in _parse_watch_dirs(cfg):
        exts = wd.extensions if wd.extensions else pipeline.registry.supported_extensions
        for ext in exts:
            pattern = f"**/*{ext}" if wd.recursive else f"*{ext}"
            for path in wd.path.glob(pattern):
                if _is_excluded(path):
                    continue
                print(pipeline.ingest(path))


def benchmark(paths: list[str]):
    from src.ingest.pipeline import IngestPipeline

    pipeline = IngestPipeline.from_config("config.yaml")
    results = [pipeline.benchmark_file(path) for path in paths]
    print(json.dumps(results, ensure_ascii=False, indent=2))


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
    elif cmd == "benchmark":
        if len(sys.argv) < 3:
            print("Usage: python main.py benchmark <path> [<path> ...]")
            sys.exit(1)
        benchmark(sys.argv[2:])
    else:
        print(f"Unknown command: {cmd}")
        sys.exit(1)
