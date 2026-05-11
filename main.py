#!/usr/bin/env python3
"""
DocFlow 入口。

用法：
  # 启动 Web 服务（含文件夹监控）
  python main.py serve

  # 启动前检查，或检查后启动 Web 服务
  python main.py doctor [--json] [--strict] [--port 8000]
  python main.py start [--host 0.0.0.0] [--port 8000] [--check-only] [--json]

  # macOS 后台服务（launchd）
  python main.py install-local [--apply] [--with-service] [--skip-deps] [--host 127.0.0.1] [--port 8000]
  python main.py service install [--dry-run] [--host 127.0.0.1] [--port 8000] [--python /path/to/python]
  python main.py service status
  python main.py service uninstall [--dry-run]

  # 手动 ingest 单个文件
  python main.py ingest /path/to/file.pdf
  python main.py demo [--create-only] [--json]

  # dry-run benchmark 一个或多个文件
  python main.py benchmark /path/to/file1.md /path/to/file2.pdf

  # 运行固定评估集（不调用回答 LLM）
  python main.py eval retrieval
  python main.py eval parsing

  # 生成 Phase 11 的 9 分成熟度评分报告
  python main.py maturity-eval

  # 生成并验证真实样本套件
  python main.py sample-suite

  # 运行浏览器验收检查（需要 Web 服务已启动）
  python main.py browser-acceptance [--base-url http://127.0.0.1:8000] [--json] [--no-screenshots] [--headed]

  # 检查 SQLite 与 Qdrant 是否一致
  python main.py check

  # 从原始文件重建索引，或只重建 Qdrant
  python main.py rebuild [--qdrant-only] [--dry-run]
  python main.py repair-ids [--dry-run]

  # 备份、导出 chunk，或查看恢复步骤
  python main.py backup [--dry-run] [--output backups] [--keep 5]
  python main.py export-chunks [--output backups/chunks.jsonl]
  python main.py restore-plan <backup.tar.gz>
  python main.py restore-drill [--output-dir /tmp/docflow-phase22-restore-drill] [--json]

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
    _config_path()
    import uvicorn
    uvicorn.run("src.api.app:app", host="0.0.0.0", port=8000, reload=False)


def doctor_command(args: list[str]):
    from src.maintenance.startup import (
        doctor_command as run_doctor,
        offline_doctor_command as run_offline_doctor,
    )

    port = int(_arg_value(args, "--port", "8000"))
    if "--offline" in args:
        return run_offline_doctor(
            _config_path(),
            app_port=port,
            as_json="--json" in args,
        )
    return run_doctor(
        _config_path(),
        app_port=port,
        as_json="--json" in args,
        strict="--strict" in args,
    )


def start_command(args: list[str]):
    from src.maintenance.startup import start_command as run_start

    host = _arg_value(args, "--host", "0.0.0.0")
    port = int(_arg_value(args, "--port", "8000"))
    return run_start(
        _config_path(),
        host=host,
        port=port,
        as_json="--json" in args,
        check_only="--check-only" in args,
    )


def service_command(args: list[str]):
    from pathlib import Path

    from src.maintenance.launchd import (
        install_service,
        print_result,
        service_status,
        uninstall_service,
    )

    action = args[0] if args else "status"
    if action == "install":
        host = _arg_value(args, "--host", "127.0.0.1")
        port = int(_arg_value(args, "--port", "8000"))
        python_arg = _arg_value(args, "--python")
        result = install_service(
            host=host,
            port=port,
            python_bin=Path(python_arg).expanduser() if python_arg else None,
            dry_run="--dry-run" in args,
        )
    elif action == "uninstall":
        result = uninstall_service(dry_run="--dry-run" in args)
    elif action == "status":
        result = service_status()
    else:
        print("Usage: python main.py service install|status|uninstall [--dry-run]")
        return 1
    print_result(result)
    return 0 if result["status"] in {"ok", "dry_run", "loaded", "not_loaded"} else 1


def install_local_command(args: list[str]):
    from src.maintenance.local_install import install_local, print_result

    host = _arg_value(args, "--host", "127.0.0.1")
    port = int(_arg_value(args, "--port", "8000"))
    result = install_local(
        dry_run="--apply" not in args,
        with_service="--with-service" in args,
        host=host,
        port=port,
        skip_deps="--skip-deps" in args,
    )
    print_result(result)
    return 0 if result["status"] in {"ok", "dry_run"} else 1


def ingest(path: str):
    from src.ingest.pipeline import IngestPipeline
    pipeline = IngestPipeline.from_config(_config_path())
    result = pipeline.ingest(path)
    print(result)


def demo_command(args: list[str]):
    from src.maintenance.demo import demo_command as run_demo

    return run_demo(args)


def scan():
    import yaml
    from src.ingest.pipeline import IngestPipeline
    from src.api.app import _parse_watch_dirs
    from src.ingest.watcher import _is_excluded
    config_path = _config_path()
    with open(config_path) as f:
        cfg = yaml.safe_load(f)
    pipeline = IngestPipeline.from_config(config_path)
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

    pipeline = IngestPipeline.from_config(_config_path())
    results = [pipeline.benchmark_file(path) for path in paths]
    print(json.dumps(results, ensure_ascii=False, indent=2))


def eval_command(args: list[str]):
    if args and args[0] == "parsing":
        from scripts.run_parsing_eval import main as run_parsing_eval_main

        sys.argv = [sys.argv[0], *args[1:]]
        return run_parsing_eval_main()
    if args and args[0] == "retrieval":
        args = args[1:]
    from scripts.run_eval import main as run_eval_main

    sys.argv = [sys.argv[0], *args]
    return run_eval_main()


def maturity_eval(args: list[str]):
    from scripts.run_maturity_eval import main as run_maturity_main

    sys.argv = [sys.argv[0], *args]
    return run_maturity_main()


def sample_suite(args: list[str]):
    from scripts.run_sample_suite import main as run_sample_suite_main

    sys.argv = [sys.argv[0], *args]
    return run_sample_suite_main()


def browser_acceptance(args: list[str]):
    from scripts.run_browser_acceptance import main as run_browser_acceptance_main

    sys.argv = [sys.argv[0], *args]
    return run_browser_acceptance_main()


def restore_drill_command(args: list[str]):
    from scripts.run_restore_drill import main as run_restore_drill_main

    sys.argv = [sys.argv[0], *args]
    return run_restore_drill_main()


def check_index(args: list[str]):
    from src.maintenance.consistency import check_consistency, print_report

    as_json = "--json" in args
    report = check_consistency(_config_path())
    print_report(report, as_json=as_json)
    return 0 if report.ok else 1


def rebuild_command(args: list[str]):
    from src.maintenance.consistency import rebuild_index, rebuild_qdrant_only

    dry_run = "--dry-run" in args
    qdrant_only = "--qdrant-only" in args
    if qdrant_only:
        result = rebuild_qdrant_only(_config_path(), dry_run=dry_run)
    else:
        result = rebuild_index(_config_path(), dry_run=dry_run)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


def repair_ids_command(args: list[str]):
    from src.maintenance.consistency import repair_index_ids

    dry_run = "--dry-run" in args
    result = repair_index_ids(_config_path(), dry_run=dry_run)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if result["status"] in {"done", "dry_run"} else 1


def backup_command(args: list[str]):
    from src.maintenance.backup import create_backup

    dry_run = "--dry-run" in args
    output = _arg_value(args, "--output", "backups")
    keep = int(_arg_value(args, "--keep", "5"))
    result = create_backup(_config_path(), output_dir=output, keep=keep, dry_run=dry_run)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


def export_chunks_command(args: list[str]):
    from src.maintenance.backup import export_chunks_jsonl

    output = _arg_value(args, "--output")
    result = export_chunks_jsonl(_config_path(), output_path=output)
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


def _config_path() -> str:
    from src.maintenance.startup import ensure_config_file

    return str(ensure_config_file("config.yaml"))


def cli() -> int:
    cmd = sys.argv[1] if len(sys.argv) > 1 else "serve"
    if cmd == "serve":
        serve()
        return 0
    elif cmd == "doctor":
        return doctor_command(sys.argv[2:])
    elif cmd == "start":
        return start_command(sys.argv[2:])
    elif cmd == "service":
        return service_command(sys.argv[2:])
    elif cmd == "install-local":
        return install_local_command(sys.argv[2:])
    elif cmd == "ingest":
        if len(sys.argv) < 3:
            print("Usage: python main.py ingest <path>")
            return 1
        ingest(sys.argv[2])
        return 0
    elif cmd == "demo":
        return demo_command(sys.argv[2:])
    elif cmd == "scan":
        scan()
        return 0
    elif cmd == "benchmark":
        if len(sys.argv) < 3:
            print("Usage: python main.py benchmark <path> [<path> ...]")
            return 1
        benchmark(sys.argv[2:])
        return 0
    elif cmd == "eval":
        return eval_command(sys.argv[2:])
    elif cmd == "maturity-eval":
        return maturity_eval(sys.argv[2:])
    elif cmd == "sample-suite":
        return sample_suite(sys.argv[2:])
    elif cmd == "browser-acceptance":
        return browser_acceptance(sys.argv[2:])
    elif cmd == "restore-drill":
        return restore_drill_command(sys.argv[2:])
    elif cmd == "check":
        return check_index(sys.argv[2:])
    elif cmd == "rebuild":
        return rebuild_command(sys.argv[2:])
    elif cmd == "repair-ids":
        return repair_ids_command(sys.argv[2:])
    elif cmd == "backup":
        return backup_command(sys.argv[2:])
    elif cmd == "export-chunks":
        return export_chunks_command(sys.argv[2:])
    elif cmd == "restore-plan":
        return restore_plan_command(sys.argv[2:])
    else:
        print(f"Unknown command: {cmd}")
        return 1


if __name__ == "__main__":
    sys.exit(cli())
