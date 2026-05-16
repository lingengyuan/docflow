"""Startup preflight checks and one-command service launcher."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import yaml

from src.maintenance.offline_doctor import (
    _run_offline_runtime_checks as _run_offline_runtime_checks,
)
from src.maintenance.offline_doctor import (
    build_offline_report as build_offline_report,
)
from src.maintenance.offline_doctor import (
    format_offline_report as format_offline_report,
)
from src.maintenance.offline_doctor import (
    offline_doctor_command as offline_doctor_command,
)
from src.maintenance.startup_checks import (
    CORE_MODULES as CORE_MODULES,
)
from src.maintenance.startup_checks import (
    STARTUP_BLOCKERS as STARTUP_BLOCKERS,
)
from src.maintenance.startup_checks import (
    STATUS_ORDER as STATUS_ORDER,
)
from src.maintenance.startup_checks import (
    _run_command as _run_command,
)
from src.maintenance.startup_checks import (
    aggregate_status,
    check_app_port,
    check_ollama,
    check_python_dependencies,
    check_qdrant,
    check_sqlite,
    startup_blockers,
)
from src.maintenance.startup_checks import (
    ensure_qdrant as _ensure_qdrant,
)
from src.maintenance.startup_checks import (
    qdrant_run_command as qdrant_run_command,
)
from src.resources import resource_path


def ensure_config_file(config_path: str | Path, example_path: str | Path | None = None) -> Path:
    path = Path(config_path).expanduser()
    if path.exists():
        return path

    example = Path(example_path).expanduser() if example_path else _default_example_path(path)
    if not example.exists():
        raise FileNotFoundError(f"Config file not found: {path}. Missing example: {example}")

    path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(example, path)
    with path.open(encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}
    _ensure_config_directories(cfg, base_dir=path.resolve().parent)
    return path


def load_config(config_path: str | Path) -> tuple[dict, Path]:
    path = ensure_config_file(config_path)
    with path.open(encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}
    return cfg, path


def _default_example_path(config_path: Path) -> Path:
    sibling = config_path.with_name("config.example.yaml")
    if sibling.exists():
        return sibling
    return resource_path("config.example.yaml")


def _ensure_config_directories(cfg: dict, base_dir: Path) -> None:
    paths_cfg = cfg.get("paths", {})
    for key in ("db_path", "id_counter"):
        value = paths_cfg.get(key)
        if value:
            _resolve_config_path(value, base_dir).parent.mkdir(parents=True, exist_ok=True)

    raw_watch_dirs = paths_cfg.get("watch_dirs", paths_cfg.get("watch_dir", []))
    if isinstance(raw_watch_dirs, (str, Path)):
        raw_watch_dirs = [{"path": str(raw_watch_dirs)}]
    for entry in raw_watch_dirs or []:
        raw_path = entry.get("path") if isinstance(entry, dict) else entry
        if raw_path:
            _resolve_config_path(raw_path, base_dir).mkdir(parents=True, exist_ok=True)


def _resolve_config_path(value: str | Path, base_dir: Path) -> Path:
    path = Path(value).expanduser()
    return path if path.is_absolute() else base_dir / path


def check_config(config_path: str | Path) -> dict:
    try:
        cfg, path = load_config(config_path)
    except Exception as exc:
        return {
            "status": "unavailable",
            "path": str(Path(config_path).expanduser()),
            "error": str(exc),
            "actions": ["Fix config.yaml before starting DocFlow."],
        }

    missing = []
    for dotted in ("paths.db_path", "qdrant.host", "qdrant.port"):
        current = cfg
        for part in dotted.split("."):
            if not isinstance(current, dict) or part not in current:
                missing.append(dotted)
                break
            current = current[part]

    status = "ok" if not missing else "unavailable"
    return {
        "status": status,
        "path": str(path),
        "missing": missing,
        "actions": ["Restore required config.yaml keys."] if missing else [],
    }


def ensure_qdrant(cfg: dict, runner=_run_command) -> dict:
    return _ensure_qdrant(
        cfg,
        runner=runner,
        check_qdrant_fn=check_qdrant,
        docker_which=shutil.which,
    )


def build_startup_report(
    config_path: str | Path = "config.yaml",
    app_host: str = "127.0.0.1",
    app_port: int = 8000,
    try_start_qdrant: bool = False,
) -> dict:
    checks: dict[str, dict] = {
        "python": check_python_dependencies(),
        "config": check_config(config_path),
    }

    if checks["config"]["status"] == "ok":
        cfg, _ = load_config(config_path)
        qdrant_result = ensure_qdrant(cfg)["result"] if try_start_qdrant else check_qdrant(cfg)
        checks.update(
            {
                "sqlite": check_sqlite(cfg),
                "qdrant": qdrant_result,
                "ollama": check_ollama(cfg),
                "port": check_app_port(app_port, host="127.0.0.1"),
            }
        )
    else:
        checks.update(
            {
                "sqlite": {"status": "unavailable", "actions": ["Fix config.yaml first."]},
                "qdrant": {"status": "unavailable", "actions": ["Fix config.yaml first."]},
                "ollama": {
                    "status": "degraded",
                    "optional": True,
                    "actions": ["Fix config.yaml first."],
                },
                "port": check_app_port(app_port, host="127.0.0.1"),
            }
        )

    actions = []
    for check in checks.values():
        actions.extend(check.get("actions", []))

    blockers = startup_blockers(checks)
    return {
        "status": aggregate_status(checks),
        "can_start": not blockers,
        "startup_blockers": blockers,
        "url": f"http://localhost:{int(app_port)}",
        "host": app_host,
        "port": int(app_port),
        "checks": checks,
        "actions": list(dict.fromkeys(action for action in actions if action)),
    }


def format_report(report: dict) -> str:
    lines = [
        f"DocFlow startup check: {report['status']}",
        f"Local URL: {report['url']}",
    ]
    for name, check in report["checks"].items():
        status = check.get("status", "unavailable")
        marker = "ok" if status == "ok" else "warn" if status == "degraded" else "fail"
        detail = check.get("error") or check.get("quick_check") or ""
        if detail and detail != "ok":
            lines.append(f"- [{marker}] {name}: {status} ({detail})")
        else:
            lines.append(f"- [{marker}] {name}: {status}")

    if report["startup_blockers"]:
        lines.append(f"Startup blockers: {', '.join(report['startup_blockers'])}")
    if report["actions"]:
        lines.append("Next actions:")
        for action in report["actions"]:
            lines.append(f"- {action}")
    return "\n".join(lines)


def doctor_command(
    config_path: str | Path = "config.yaml",
    app_port: int = 8000,
    as_json: bool = False,
    strict: bool = False,
) -> int:
    report = build_startup_report(config_path=config_path, app_port=app_port)
    print(json.dumps(report, ensure_ascii=False, indent=2) if as_json else format_report(report))
    if strict:
        return 0 if report["status"] == "ok" else 1
    return 0 if report["can_start"] else 1


def start_command(
    config_path: str | Path = "config.yaml",
    host: str = "0.0.0.0",
    port: int = 8000,
    as_json: bool = False,
    check_only: bool = False,
) -> int:
    report = build_startup_report(
        config_path=config_path,
        app_host=host,
        app_port=port,
        try_start_qdrant=True,
    )
    print(json.dumps(report, ensure_ascii=False, indent=2) if as_json else format_report(report))
    if not report["can_start"]:
        return 1
    if check_only:
        return 0

    print(f"Starting DocFlow at {report['url']}")
    import uvicorn

    uvicorn.run("src.api.app:app", host=host, port=int(port), reload=False)
    return 0
