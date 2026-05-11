"""Startup preflight checks and one-command service launcher."""

from __future__ import annotations

import importlib.util
import json
import shutil
import socket
import sqlite3
import subprocess
import sys
import time
from pathlib import Path
from typing import Callable

import httpx
import yaml

STATUS_ORDER = {"ok": 0, "degraded": 1, "unavailable": 2}
STARTUP_BLOCKERS = {"python", "config", "qdrant", "port"}

CORE_MODULES = {
    "fastapi": "fastapi",
    "uvicorn": "uvicorn",
    "yaml": "PyYAML",
    "qdrant_client": "qdrant-client",
    "sentence_transformers": "sentence-transformers",
    "torch": "torch",
    "watchdog": "watchdog",
}


def load_config(config_path: str | Path) -> tuple[dict, Path]:
    path = Path(config_path).expanduser()
    with path.open() as f:
        cfg = yaml.safe_load(f) or {}
    return cfg, path


def aggregate_status(checks: dict[str, dict]) -> str:
    worst = "ok"
    for check in checks.values():
        status = check.get("status", "unavailable")
        if STATUS_ORDER[status] > STATUS_ORDER[worst]:
            worst = status
    return worst


def startup_blockers(checks: dict[str, dict]) -> list[str]:
    return [
        name
        for name, check in checks.items()
        if name in STARTUP_BLOCKERS and check.get("status") == "unavailable"
    ]


def check_python_dependencies() -> dict:
    missing = [
        package_name
        for module_name, package_name in CORE_MODULES.items()
        if importlib.util.find_spec(module_name) is None
    ]
    version = ".".join(str(part) for part in sys.version_info[:3])
    too_old = sys.version_info < (3, 11)
    status = "ok" if not missing and not too_old else "unavailable"
    actions = []
    if too_old:
        actions.append("Use Python 3.11 or newer.")
    if missing:
        actions.append("Run: pip install -r requirements.txt")
    return {
        "status": status,
        "python": version,
        "missing": missing,
        "actions": actions,
    }


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


def check_sqlite(cfg: dict) -> dict:
    db_path = Path(cfg["paths"]["db_path"]).expanduser()
    parent = db_path.parent
    if not parent.exists():
        return {
            "status": "unavailable",
            "path": str(db_path),
            "error": f"Parent directory does not exist: {parent}",
            "actions": [f"Create the database directory: mkdir -p {parent}"],
        }
    if not db_path.exists():
        return {
            "status": "degraded",
            "path": str(db_path),
            "quick_check": "database file does not exist yet",
            "actions": ["Start DocFlow once to create the local database."],
        }

    try:
        conn = sqlite3.connect(db_path)
        try:
            conn.execute("SELECT 1").fetchone()
            quick_check = conn.execute("PRAGMA quick_check").fetchone()[0]
        finally:
            conn.close()
    except Exception as exc:
        return {
            "status": "unavailable",
            "path": str(db_path),
            "error": str(exc),
            "actions": ["Repair or rebuild the local SQLite index before relying on search."],
        }

    status = "ok" if quick_check == "ok" else "unavailable"
    actions = []
    if status != "ok":
        actions.append("Repair or rebuild the local SQLite index before relying on search.")
    return {
        "status": status,
        "path": str(db_path),
        "quick_check": quick_check,
        "actions": actions,
    }


def _http_json(url: str, timeout: float = 2.0) -> tuple[int, dict]:
    response = httpx.get(
        url,
        headers={"Accept": "application/json"},
        timeout=httpx.Timeout(timeout, connect=min(timeout, 1.0)),
    )
    response.raise_for_status()
    return response.status_code, response.json()


def check_qdrant(cfg: dict) -> dict:
    host = cfg["qdrant"]["host"]
    port = int(cfg["qdrant"]["port"])
    collection = cfg.get("qdrant", {}).get("collection", "docflow")
    base_url = f"http://{host}:{port}"
    try:
        _, data = _http_json(f"{base_url}/collections/{collection}", timeout=2)
    except httpx.HTTPStatusError as exc:
        if exc.response.status_code == 404:
            return {
                "status": "degraded",
                "host": host,
                "port": port,
                "collection": collection,
                "error": "Collection does not exist yet.",
                "actions": ["Start DocFlow; the collection can be created during indexing."],
            }
        return {
            "status": "unavailable",
            "host": host,
            "port": port,
            "collection": collection,
            "error": str(exc),
            "actions": [qdrant_run_command(port)],
        }
    except Exception as exc:
        return {
            "status": "unavailable",
            "host": host,
            "port": port,
            "collection": collection,
            "error": str(exc),
            "actions": [qdrant_run_command(port)],
        }

    result = data.get("result", {})
    return {
        "status": "ok",
        "host": host,
        "port": port,
        "collection": collection,
        "points_count": result.get("points_count"),
        "vectors_count": result.get("vectors_count"),
        "actions": [],
    }


def check_ollama(cfg: dict) -> dict:
    base_url = cfg.get("ollama", {}).get("base_url", "http://localhost:11434").rstrip("/")
    try:
        _, data = _http_json(f"{base_url}/api/tags", timeout=2)
    except Exception as exc:
        return {
            "status": "degraded",
            "base_url": base_url,
            "error": str(exc),
            "optional": True,
            "actions": ["Open Ollama when OCR or Ollama-backed features are needed."],
        }

    installed = set()
    for item in data.get("models", []):
        name = item.get("name", "")
        if not name:
            continue
        installed.add(name)
        installed.add(name.split(":", 1)[0])

    required = []
    ocr_model = cfg.get("ollama", {}).get("ocr_model")
    if ocr_model:
        required.append(ocr_model)
    if cfg.get("llm", {}).get("backend") == "local":
        model = cfg.get("llm", {}).get("ollama_model") or cfg.get("ollama", {}).get("llm_model")
        if model:
            required.append(model)
    if cfg.get("ingest", {}).get("contextual_prefix_mode") == "ollama":
        model = cfg.get("ingest", {}).get("contextual_prefix_model")
        if model:
            required.append(model)

    missing = [
        model
        for model in required
        if model not in installed and model.split(":", 1)[0] not in installed
    ]
    return {
        "status": "ok" if not missing else "degraded",
        "base_url": base_url,
        "missing_models": missing,
        "optional": True,
        "actions": [f"Run: ollama pull {model}" for model in missing],
    }


def check_app_port(port: int, host: str = "127.0.0.1") -> dict:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.settimeout(0.5)
        in_use = sock.connect_ex((host, int(port))) == 0
    return {
        "status": "unavailable" if in_use else "ok",
        "host": host,
        "port": int(port),
        "actions": [f"Use another port, for example: python main.py start --port {int(port) + 1}"] if in_use else [],
    }


def qdrant_run_command(port: int) -> str:
    return f"Run: docker run -d --name qdrant -p {port}:6333 qdrant/qdrant"


def _run_command(args: list[str], timeout: float = 10.0) -> subprocess.CompletedProcess:
    return subprocess.run(args, capture_output=True, text=True, timeout=timeout, check=False)


def ensure_qdrant(cfg: dict, runner: Callable[[list[str], float], subprocess.CompletedProcess] = _run_command) -> dict:
    current = check_qdrant(cfg)
    if current["status"] != "unavailable":
        return {"attempted": False, "result": current, "actions": current.get("actions", [])}

    port = int(cfg["qdrant"]["port"])
    if shutil.which("docker") is None:
        return {
            "attempted": False,
            "result": current,
            "actions": ["Install or open Docker Desktop.", qdrant_run_command(port)],
        }

    inspect = runner(["docker", "inspect", "qdrant"], 10.0)
    if inspect.returncode != 0:
        return {
            "attempted": False,
            "result": current,
            "actions": [qdrant_run_command(port)],
        }

    started = runner(["docker", "start", "qdrant"], 20.0)
    if started.returncode != 0:
        return {
            "attempted": True,
            "result": current,
            "error": started.stderr.strip() or started.stdout.strip(),
            "actions": ["Open Docker Desktop, then run: docker start qdrant"],
        }

    for _ in range(20):
        time.sleep(0.5)
        checked = check_qdrant(cfg)
        if checked["status"] != "unavailable":
            return {"attempted": True, "result": checked, "actions": checked.get("actions", [])}

    return {
        "attempted": True,
        "result": current,
        "actions": ["Qdrant container started, but the service did not answer within 10 seconds."],
    }


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
        checks.update({
            "sqlite": check_sqlite(cfg),
            "qdrant": qdrant_result,
            "ollama": check_ollama(cfg),
            "port": check_app_port(app_port, host="127.0.0.1"),
        })
    else:
        checks.update({
            "sqlite": {"status": "unavailable", "actions": ["Fix config.yaml first."]},
            "qdrant": {"status": "unavailable", "actions": ["Fix config.yaml first."]},
            "ollama": {"status": "degraded", "optional": True, "actions": ["Fix config.yaml first."]},
            "port": check_app_port(app_port, host="127.0.0.1"),
        })

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
