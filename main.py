#!/usr/bin/env python3
"""
DocFlow 入口。

用法：
  docflow serve
  docflow start
  docflow doctor --offline
  docflow scan
  docflow ingest /path/to/file.pdf
  docflow admin check
  docflow admin platform
  docflow dev eval public
  docflow dev browser-acceptance
"""

import json
import logging
import os
import sys

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s  %(name)s  %(message)s",
    datefmt="%H:%M:%S",
)
logging.getLogger("httpx").setLevel(logging.WARNING)


USER_COMMANDS = {
    "serve": "Start the local browser app.",
    "start": "Check local requirements, then start the browser app.",
    "doctor": "Check local requirements and offline privacy coverage.",
    "status": "Show the same local readiness check without starting the app.",
    "demo": "Create or ingest the bundled demo library.",
    "scan": "Scan configured watched folders.",
    "ingest": "Ingest one file.",
}

ADMIN_COMMANDS = {
    "platform": "Show local runtime capability details.",
    "check": "Check SQLite and vector-store consistency.",
    "rebuild": "Rebuild SQLite/Qdrant indexes.",
    "repair-ids": "Repair vector-store ID counters.",
    "backup": "Create or preview a local backup.",
    "export-chunks": "Export indexed chunks as JSONL.",
    "restore-plan": "Inspect a backup archive before restoring.",
    "restore-drill": "Run a disposable backup/restore drill.",
    "service": "Manage the optional local background service.",
    "install-local": "Prepare a local source checkout installation.",
}

DEV_COMMANDS = {
    "eval": "Run retrieval, parsing, performance, faithfulness, or external checks.",
    "browser-acceptance": "Run browser acceptance checks against a running app.",
    "sample-suite": "Generate and validate the sample-suite fixtures.",
    "maturity-eval": "Run the internal planning scorecard.",
    "dead-code-audit": "Audit the command and release surface for stale entries.",
}

RETIRED_TOP_LEVEL_COMMANDS = {
    "browser-acceptance": "docflow dev browser-acceptance",
    "sample-suite": "docflow dev sample-suite",
    "maturity-eval": "docflow dev maturity-eval",
    "restore-drill": "docflow admin restore-drill",
    "check": "docflow admin check",
    "rebuild": "docflow admin rebuild",
    "repair-ids": "docflow admin repair-ids",
    "backup": "docflow admin backup",
    "export-chunks": "docflow admin export-chunks",
    "restore-plan": "docflow admin restore-plan",
    "service": "docflow admin service",
    "install-local": "docflow admin install-local",
    "platform": "docflow admin platform",
    "eval": "docflow dev eval",
    "benchmark": "docflow dev eval performance",
}


def serve():
    _config_path()
    import uvicorn

    uvicorn.run("src.api.app:app", host="0.0.0.0", port=8000, reload=False)


def doctor_command(args: list[str]):
    from src.maintenance.startup import (
        doctor_command as run_doctor,
    )
    from src.maintenance.startup import (
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


def platform_command(args: list[str]):
    from src.maintenance.platform import platform_command as run_platform

    return run_platform(args)


def status_command(args: list[str]):
    return doctor_command(args)


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
        print("Usage: docflow admin service install|status|uninstall [--dry-run]")
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

    from src.api.app import _parse_watch_dirs
    from src.ingest.pipeline import IngestPipeline
    from src.ingest.watcher import _is_excluded

    config_path = _config_path()
    with open(config_path) as f:
        cfg = yaml.safe_load(f)
    pipeline = IngestPipeline.from_config(config_path)
    for wd in _parse_watch_dirs(cfg, config_path=config_path):
        exts = wd.extensions if wd.extensions else pipeline.registry.supported_extensions
        for ext in exts:
            pattern = f"**/*{ext}" if wd.recursive else f"*{ext}"
            for path in wd.path.glob(pattern):
                if _is_excluded(path):
                    continue
                print(pipeline.ingest(path))


def eval_command(args: list[str]):
    if _is_help(args):
        _print_eval_help()
        return 0
    if args and args[0] == "parsing":
        from scripts.run_parsing_eval import main as run_parsing_eval_main

        sys.argv = ["docflow dev eval parsing", *args[1:]]
        return run_parsing_eval_main()
    if args and args[0] == "public":
        from scripts.run_public_eval import main as run_public_eval_main

        sys.argv = ["docflow dev eval public", *args[1:]]
        return run_public_eval_main()
    if args and args[0] == "performance":
        from scripts.run_performance_smoke import main as run_performance_smoke_main

        sys.argv = ["docflow dev eval performance", *args[1:]]
        return run_performance_smoke_main()
    if args and args[0] == "faithfulness":
        from scripts.run_faithfulness_eval import main as run_faithfulness_eval_main

        sys.argv = ["docflow dev eval faithfulness", *args[1:]]
        return run_faithfulness_eval_main()
    if args and args[0] == "large-library":
        from scripts.run_large_library_benchmark import main as run_large_library_benchmark_main

        sys.argv = ["docflow dev eval large-library", *args[1:]]
        return run_large_library_benchmark_main()
    if args and args[0] == "external":
        if len(args) > 1 and args[1] == "run":
            from scripts.run_external_retrieval_eval import main as run_external_retrieval_main

            sys.argv = ["docflow dev eval external run", *args[2:]]
            return run_external_retrieval_main()
        from scripts.run_external_benchmark_status import main as run_external_benchmark_status_main

        sys.argv = ["docflow dev eval external", *args[1:]]
        return run_external_benchmark_status_main()
    if args and args[0] == "retrieval":
        args = args[1:]
    elif args and not args[0].startswith("-"):
        print(f"Unknown eval command: {args[0]}")
        _print_eval_help()
        return 1
    from scripts.run_eval import main as run_eval_main

    sys.argv = ["docflow dev eval retrieval", *args]
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


def dead_code_audit(args: list[str]):
    from scripts.run_dead_code_audit import main as run_dead_code_main

    sys.argv = [sys.argv[0], *args]
    return run_dead_code_main()


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
        print("Usage: docflow admin restore-plan <backup.tar.gz>")
        return 1
    result = restore_plan(archive_args[0])
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if result["status"] in {"ok", "incomplete"} else 1


def _arg_value(args: list[str], name: str, default: str | None = None) -> str | None:
    prefix = f"{name}="
    for arg in args:
        if arg.startswith(prefix):
            return arg[len(prefix) :]
    if name in args:
        index = args.index(name)
        if index + 1 < len(args):
            return args[index + 1]
    return default


def _config_path() -> str:
    from src.maintenance.startup import ensure_config_file

    return str(ensure_config_file(os.getenv("DOCFLOW_CONFIG", "config.yaml")))


def _print_command_group(title: str, commands: dict[str, str], prefix: str = "docflow") -> None:
    print(title)
    width = max(len(command) for command in commands)
    for command, description in commands.items():
        print(f"  {prefix} {command:<{width}}  {description}")


def _print_help() -> None:
    print("DocFlow local personal knowledge workspace")
    print()
    _print_command_group("Daily commands:", USER_COMMANDS)
    print()
    print("Maintenance and contributor commands are grouped:")
    print("  docflow admin <command>  Backups, restore drills, index checks, service install")
    print("  docflow dev <command>    Browser acceptance, sample fixtures, internal audits")
    print()
    print("Run `docflow admin --help` or `docflow dev --help` for details.")


def _print_group_help(group: str, commands: dict[str, str]) -> None:
    _print_command_group(f"{group.title()} commands:", commands, prefix=f"docflow {group}")


def _print_eval_help() -> None:
    print("Eval commands:")
    print("  docflow dev eval public       Run public-domain retrieval checks.")
    print("  docflow dev eval retrieval    Run internal retrieval regression checks.")
    print("  docflow dev eval parsing      Run parsing fixture checks.")
    print("  docflow dev eval performance  Run parser/chunker performance smoke checks.")
    print("  docflow dev eval faithfulness Run deterministic answer-grounding checks.")
    print("  docflow dev eval large-library Run desktop large-library smoke benchmark.")
    print("  docflow dev eval external     Show external benchmark claim status.")
    print("  docflow dev eval external run Run BEIR SciFact-lite retrieval benchmark.")


def _is_help(args: list[str]) -> bool:
    return not args or args[0] in {"--help", "-h", "help"}


def admin_command(args: list[str]) -> int:
    if _is_help(args):
        _print_group_help("admin", ADMIN_COMMANDS)
        return 0
    action = args[0]
    rest = args[1:]
    if action == "platform":
        return platform_command(rest)
    if action == "check":
        return check_index(rest)
    if action == "rebuild":
        return rebuild_command(rest)
    if action == "repair-ids":
        return repair_ids_command(rest)
    if action == "backup":
        return backup_command(rest)
    if action == "export-chunks":
        return export_chunks_command(rest)
    if action == "restore-plan":
        return restore_plan_command(rest)
    if action == "restore-drill":
        return restore_drill_command(rest)
    if action == "service":
        return service_command(rest)
    if action == "install-local":
        return install_local_command(rest)
    print(f"Unknown admin command: {action}")
    _print_group_help("admin", ADMIN_COMMANDS)
    return 1


def dev_command(args: list[str]) -> int:
    if _is_help(args):
        _print_group_help("dev", DEV_COMMANDS)
        return 0
    action = args[0]
    rest = args[1:]
    if action == "eval":
        return eval_command(rest)
    if action == "browser-acceptance":
        return browser_acceptance(rest)
    if action == "sample-suite":
        return sample_suite(rest)
    if action == "maturity-eval":
        return maturity_eval(rest)
    if action == "dead-code-audit":
        return dead_code_audit(rest)
    print(f"Unknown dev command: {action}")
    _print_group_help("dev", DEV_COMMANDS)
    return 1


def _retired_command(cmd: str) -> int:
    replacement = RETIRED_TOP_LEVEL_COMMANDS[cmd]
    print(f"`docflow {cmd}` has moved. Use `{replacement}`.")
    return 1


def cli() -> int:
    cmd = sys.argv[1] if len(sys.argv) > 1 else "serve"
    if cmd in {"--help", "-h", "help"}:
        _print_help()
        return 0
    if cmd == "serve":
        serve()
        return 0
    elif cmd == "doctor":
        return doctor_command(sys.argv[2:])
    elif cmd == "start":
        return start_command(sys.argv[2:])
    elif cmd == "status":
        return status_command(sys.argv[2:])
    elif cmd == "admin":
        return admin_command(sys.argv[2:])
    elif cmd == "dev":
        return dev_command(sys.argv[2:])
    elif cmd == "ingest":
        if len(sys.argv) < 3:
            print("Usage: docflow ingest <path>")
            return 1
        ingest(sys.argv[2])
        return 0
    elif cmd == "demo":
        return demo_command(sys.argv[2:])
    elif cmd == "scan":
        scan()
        return 0
    elif cmd in RETIRED_TOP_LEVEL_COMMANDS:
        return _retired_command(cmd)
    else:
        print(f"Unknown command: {cmd}")
        return 1


if __name__ == "__main__":
    sys.exit(cli())
