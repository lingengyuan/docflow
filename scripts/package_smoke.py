#!/usr/bin/env python3
from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path


def _run(args: list[str], *, cwd: Path | None = None, env: dict[str, str] | None = None) -> None:
    subprocess.run(args, cwd=cwd, env=env, check=True)


def _latest_wheel(dist_dir: Path) -> Path:
    wheels = sorted(dist_dir.glob("docflow-*.whl"), key=lambda path: path.stat().st_mtime)
    if not wheels:
        raise RuntimeError(f"No wheel built in {dist_dir}")
    return wheels[-1]


def main() -> int:
    repo = Path(__file__).resolve().parents[1]
    with tempfile.TemporaryDirectory(prefix="docflow-package-smoke-") as tmp_raw:
        tmp = Path(tmp_raw)
        dist_dir = tmp / "dist"
        install_dir = tmp / "install"
        work_dir = tmp / "work"
        dist_dir.mkdir()
        install_dir.mkdir()
        work_dir.mkdir()

        _run([sys.executable, "-m", "build", "--wheel", "--outdir", str(dist_dir)], cwd=repo)
        wheel = _latest_wheel(dist_dir)
        _run(
            [
                sys.executable,
                "-m",
                "pip",
                "install",
                "--quiet",
                "--no-deps",
                "--target",
                str(install_dir),
                str(wheel),
            ]
        )

        check = """
import json
from pathlib import Path

from src.resources import resource_path
from src.maintenance.startup import ensure_config_file

frontend = resource_path("frontend")
example = resource_path("config.example.yaml")
assert (frontend / "index.html").is_file(), frontend
assert (frontend / "partials" / "app.html").is_file(), frontend
assert (frontend / "js" / "bootstrap.js").is_file(), frontend
assert example.is_file(), example

config = ensure_config_file("config.yaml")
assert Path(config).is_file(), config

import src.api.app as api_app
from src.api import app_impl

assert app_impl.STATIC_DIR == frontend
assert app_impl.CONFIG_PATH.name == "config.yaml"
assert api_app.app is not None

print(json.dumps({
    "frontend": str(frontend),
    "config": str(config),
    "app": True,
}, ensure_ascii=False))
"""
        env = os.environ.copy()
        env["PYTHONPATH"] = str(install_dir)
        env["DOCFLOW_CONFIG"] = str(work_dir / "config.yaml")
        result = subprocess.run(
            [sys.executable, "-c", check],
            cwd=work_dir,
            env=env,
            check=True,
            text=True,
            stdout=subprocess.PIPE,
        )
        payload = json.loads(result.stdout.strip().splitlines()[-1])
        print(f"Package smoke passed: {wheel.name}")
        print(f"Frontend: {payload['frontend']}")
        print(f"Config: {payload['config']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
