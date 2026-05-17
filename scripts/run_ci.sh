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
"$PYTHON_BIN" main.py dev eval faithfulness --json > /tmp/docflow-faithfulness-eval.json
"$PYTHON_BIN" main.py dev eval large-library --documents 200 --queries 5 --json \
  > /tmp/docflow-large-library-smoke.json
"$PYTHON_BIN" scripts/run_external_benchmark_status.py --json > /tmp/docflow-external-benchmarks.json
"$PYTHON_BIN" scripts/run_dead_code_audit.py --json > /tmp/docflow-dead-code-audit.json
"$PYTHON_BIN" scripts/run_release_surface_check.py
"$PYTHON_BIN" scripts/run_phase110_readiness_check.py --json > /tmp/docflow-phase110-readiness.json
npm run check:frontend
npm run test:frontend
npm run build:frontend
npm run audit:frontend
"$PYTHON_BIN" scripts/package_smoke.py
