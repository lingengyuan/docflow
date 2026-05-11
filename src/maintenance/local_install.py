"""Local installer plan for DocFlow."""

from __future__ import annotations

import json
import subprocess
import sys
from collections.abc import Callable
from pathlib import Path

Runner = Callable[[list[str], Path], subprocess.CompletedProcess]


def repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def build_install_plan(
    root: Path | None = None,
    python_bin: Path | None = None,
    with_service: bool = False,
    host: str = "127.0.0.1",
    port: int = 8000,
    skip_deps: bool = False,
) -> dict:
    root = (root or repo_root()).resolve()
    bootstrap_python = python_bin or Path(sys.executable)
    venv_dir = root / ".venv"
    venv_python = venv_dir / "bin" / "python"
    app_python = venv_python if venv_python.exists() else bootstrap_python

    steps: list[dict] = []
    if not venv_dir.exists():
        steps.append(
            _step(
                "create_venv",
                "Create Python virtual environment",
                [str(bootstrap_python), "-m", "venv", ".venv"],
            )
        )
        app_python = venv_python

    if not skip_deps:
        steps.append(
            _step(
                "install_python_deps",
                "Install Python dependencies",
                [str(app_python), "-m", "pip", "install", "-r", "requirements.txt"],
            )
        )

    steps.extend(
        [
            _step(
                "startup_check",
                "Check local startup requirements",
                [
                    str(app_python),
                    "main.py",
                    "start",
                    "--check-only",
                    "--host",
                    host,
                    "--port",
                    str(port),
                ],
            ),
            _step(
                "restore_drill",
                "Run disposable restore drill",
                [str(app_python), "main.py", "restore-drill", "--json"],
            ),
            _step(
                "repair_ids_preview",
                "Preview vector ID repairs",
                [str(app_python), "main.py", "repair-ids", "--dry-run"],
            ),
        ]
    )

    if with_service:
        steps.append(
            _step(
                "install_service",
                "Install launchd background service",
                [
                    str(app_python),
                    "main.py",
                    "service",
                    "install",
                    "--host",
                    host,
                    "--port",
                    str(port),
                    "--python",
                    str(app_python),
                ],
            )
        )
    else:
        steps.append(
            _step(
                "service_dry_run",
                "Preview launchd service install",
                [
                    str(app_python),
                    "main.py",
                    "service",
                    "install",
                    "--dry-run",
                    "--host",
                    host,
                    "--port",
                    str(port),
                    "--python",
                    str(app_python),
                ],
            )
        )

    return {
        "schema": "docflow.local_install_plan.v1",
        "root": str(root),
        "python": str(app_python),
        "with_service": with_service,
        "skip_deps": skip_deps,
        "steps": steps,
    }


def install_local(
    root: Path | None = None,
    python_bin: Path | None = None,
    dry_run: bool = True,
    with_service: bool = False,
    host: str = "127.0.0.1",
    port: int = 8000,
    skip_deps: bool = False,
    runner: Runner | None = None,
) -> dict:
    plan = build_install_plan(
        root=root,
        python_bin=python_bin,
        with_service=with_service,
        host=host,
        port=port,
        skip_deps=skip_deps,
    )
    if dry_run:
        return {**plan, "status": "dry_run", "results": []}

    cwd = Path(plan["root"])
    active_runner = runner or _run
    results = []
    for step in plan["steps"]:
        result = active_runner(step["command"], cwd)
        item = {
            "id": step["id"],
            "returncode": result.returncode,
            "stdout": result.stdout.strip(),
            "stderr": result.stderr.strip(),
        }
        results.append(item)
        if result.returncode != 0:
            return {**plan, "status": "error", "results": results}
    return {**plan, "status": "ok", "results": results}


def print_result(result: dict) -> None:
    print(json.dumps(result, ensure_ascii=False, indent=2))


def _step(step_id: str, title: str, command: list[str]) -> dict:
    return {
        "id": step_id,
        "title": title,
        "command": command,
    }


def _run(command: list[str], cwd: Path) -> subprocess.CompletedProcess:
    return subprocess.run(command, cwd=cwd, capture_output=True, text=True, check=False)
