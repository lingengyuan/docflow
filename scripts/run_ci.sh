#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

PYTHON_BIN="${PYTHON_BIN:-.venv/bin/python}"

"$PYTHON_BIN" -m compileall src tests
"$PYTHON_BIN" -m ruff check .
"$PYTHON_BIN" -m mypy
"$PYTHON_BIN" -m pytest -q
"$PYTHON_BIN" scripts/run_performance_smoke.py --json > /tmp/docflow-performance-smoke.json
"$PYTHON_BIN" scripts/run_release_surface_check.py
npm run check:frontend
npm run test:frontend
npm run build:frontend
npm run audit:frontend
"$PYTHON_BIN" scripts/package_smoke.py
