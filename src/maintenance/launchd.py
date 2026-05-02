"""macOS launchd helper for running DocFlow in the background."""

from __future__ import annotations

import json
import os
import plistlib
import subprocess
import sys
from pathlib import Path

LABEL = "com.docflow.local"


def repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def default_python(root: Path) -> Path:
    venv_python = root / ".venv" / "bin" / "python"
    if venv_python.exists():
        return venv_python
    return Path(sys.executable)


def launch_agents_dir(home: Path | None = None) -> Path:
    base = home or Path.home()
    return base / "Library" / "LaunchAgents"


def plist_path(home: Path | None = None) -> Path:
    return launch_agents_dir(home) / f"{LABEL}.plist"


def logs_dir(home: Path | None = None) -> Path:
    base = home or Path.home()
    return base / "Library" / "Logs" / "docflow"


def service_target() -> str:
    return f"gui/{os.getuid()}/{LABEL}"


def service_domain() -> str:
    return f"gui/{os.getuid()}"


def build_plist(
    root: Path,
    python_bin: Path,
    host: str = "127.0.0.1",
    port: int = 8000,
    log_dir: Path | None = None,
) -> dict:
    log_root = log_dir or logs_dir()
    return {
        "Label": LABEL,
        "ProgramArguments": [
            str(python_bin),
            "main.py",
            "start",
            "--host",
            host,
            "--port",
            str(int(port)),
        ],
        "WorkingDirectory": str(root),
        "RunAtLoad": True,
        "KeepAlive": True,
        "StandardOutPath": str(log_root / "docflow.out.log"),
        "StandardErrorPath": str(log_root / "docflow.err.log"),
        "EnvironmentVariables": {
            "PYTHONUNBUFFERED": "1",
        },
    }


def write_plist(
    root: Path | None = None,
    python_bin: Path | None = None,
    host: str = "127.0.0.1",
    port: int = 8000,
    dry_run: bool = False,
) -> dict:
    root = root or repo_root()
    python_bin = python_bin or default_python(root)
    path = plist_path()
    log_root = logs_dir()
    plist = build_plist(root=root, python_bin=python_bin, host=host, port=port, log_dir=log_root)
    data = plistlib.dumps(plist, sort_keys=False)
    if not dry_run:
        path.parent.mkdir(parents=True, exist_ok=True)
        log_root.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
    return {
        "label": LABEL,
        "plist_path": str(path),
        "logs_dir": str(log_root),
        "program": plist["ProgramArguments"],
        "working_directory": str(root),
        "dry_run": dry_run,
        "plist": plist,
    }


def _run(args: list[str]) -> subprocess.CompletedProcess:
    return subprocess.run(args, capture_output=True, text=True, check=False)


def install_service(
    root: Path | None = None,
    python_bin: Path | None = None,
    host: str = "127.0.0.1",
    port: int = 8000,
    dry_run: bool = False,
) -> dict:
    written = write_plist(root=root, python_bin=python_bin, host=host, port=port, dry_run=dry_run)
    path = written["plist_path"]
    commands = [
        ["launchctl", "bootout", service_domain(), path],
        ["launchctl", "bootstrap", service_domain(), path],
        ["launchctl", "kickstart", "-k", service_target()],
    ]
    if dry_run:
        return {**written, "status": "dry_run", "commands": commands}

    results = []
    for args in commands:
        result = _run(args)
        results.append({
            "command": args,
            "returncode": result.returncode,
            "stdout": result.stdout.strip(),
            "stderr": result.stderr.strip(),
        })
        if args[1] != "bootout" and result.returncode != 0:
            return {**written, "status": "error", "commands": commands, "results": results}
    return {**written, "status": "ok", "commands": commands, "results": results}


def uninstall_service(dry_run: bool = False) -> dict:
    path = plist_path()
    commands = [["launchctl", "bootout", service_domain(), str(path)]]
    if dry_run:
        return {"status": "dry_run", "label": LABEL, "plist_path": str(path), "commands": commands}

    result = _run(commands[0])
    if path.exists():
        path.unlink()
    return {
        "status": "ok" if result.returncode in {0, 3, 36} else "error",
        "label": LABEL,
        "plist_path": str(path),
        "command": commands[0],
        "returncode": result.returncode,
        "stdout": result.stdout.strip(),
        "stderr": result.stderr.strip(),
    }


def service_status() -> dict:
    path = plist_path()
    result = _run(["launchctl", "print", service_target()])
    return {
        "status": "loaded" if result.returncode == 0 else "not_loaded",
        "label": LABEL,
        "plist_path": str(path),
        "plist_exists": path.exists(),
        "returncode": result.returncode,
        "stdout": result.stdout.strip(),
        "stderr": result.stderr.strip(),
    }


def print_result(result: dict) -> None:
    print(json.dumps(result, ensure_ascii=False, indent=2))
