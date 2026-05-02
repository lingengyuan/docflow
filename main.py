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

  # 运行固定检索评估集（不调用回答 LLM）
  python main.py eval

  # 检查 SQLite 与 Qdrant 是否一致
  python main.py check

  # 从原始文件重建索引，或只重建 Qdrant
  python main.py rebuild [--qdrant-only] [--dry-run]

  # 备份、导出 chunk，或查看恢复步骤
  python main.py backup [--dry-run] [--output backups] [--keep 5]
  python main.py export-chunks [--output backups/chunks.jsonl]
  python main.py restore-plan <backup.tar.gz>

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
logging.getLogger("httpx").setLevel(logging.WARNING)


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


def eval_retrieval(args: list[str]):
    from scripts.run_eval import main as run_eval_main

    sys.argv = [sys.argv[0], *args]
    return run_eval_main()


def check_index(args: list[str]):
    from src.maintenance.consistency import check_consistency, print_report

    as_json = "--json" in args
    report = check_consistency("config.yaml")
    print_report(report, as_json=as_json)
    return 0 if report.ok else 1


def rebuild_command(args: list[str]):
    from src.maintenance.consistency import rebuild_index, rebuild_qdrant_only

    dry_run = "--dry-run" in args
    qdrant_only = "--qdrant-only" in args
    if qdrant_only:
        result = rebuild_qdrant_only("config.yaml", dry_run=dry_run)
    else:
        result = rebuild_index("config.yaml", dry_run=dry_run)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


def backup_command(args: list[str]):
    from src.maintenance.backup import create_backup

    dry_run = "--dry-run" in args
    output = _arg_value(args, "--output", "backups")
    keep = int(_arg_value(args, "--keep", "5"))
    result = create_backup("config.yaml", output_dir=output, keep=keep, dry_run=dry_run)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


def export_chunks_command(args: list[str]):
    from src.maintenance.backup import export_chunks_jsonl

    output = _arg_value(args, "--output")
    result = export_chunks_jsonl("config.yaml", output_path=output)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


def restore_plan_command(args: list[str]):
    from src.maintenance.backup import restore_plan

    archive_args = [arg for arg in args if not arg.startswith("--")]
    if not archive_args:
        print("Usage: python main.py restore-plan <backup.tar.gz>")
        return 1
    result = restore_plan(archive_args[0])
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if result["status"] in {"ok", "incomplete"} else 1


def _arg_value(args: list[str], name: str, default: str | None = None) -> str | None:
    prefix = f"{name}="
    for arg in args:
        if arg.startswith(prefix):
            return arg[len(prefix):]
    if name in args:
        index = args.index(name)
        if index + 1 < len(args):
            return args[index + 1]
    return default


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
    elif cmd == "eval":
        sys.exit(eval_retrieval(sys.argv[2:]))
    elif cmd == "check":
        sys.exit(check_index(sys.argv[2:]))
    elif cmd == "rebuild":
        sys.exit(rebuild_command(sys.argv[2:]))
    elif cmd == "backup":
        sys.exit(backup_command(sys.argv[2:]))
    elif cmd == "export-chunks":
        sys.exit(export_chunks_command(sys.argv[2:]))
    elif cmd == "restore-plan":
        sys.exit(restore_plan_command(sys.argv[2:]))
    else:
        print(f"Unknown command: {cmd}")
        sys.exit(1)
