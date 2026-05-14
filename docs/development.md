# Development

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements-dev.txt
pip install -e .
```

Install optional image understanding support only when needed:

```bash
pip install -r requirements-vision.txt
```

Install optional Apple Silicon MLX support only when you want MLX-backed local answers or reranking:

```bash
pip install -r requirements-mac-mlx.txt
```

## Run Tests

```bash
scripts/run_ci.sh
```

This runs compile checks, ruff, mypy, pytest, frontend syntax checks, frontend unit tests, the frontend build, and dependency audit. Use `docs/release.md` before publishing a release.

Frontend and dependency checks:

```bash
npm run check:frontend
npm run test:frontend
npm run build:frontend
npm run audit:frontend
.venv/bin/python -m pip_audit -r requirements.txt -r requirements-dev.txt -r requirements-vision.txt
```

CI runs the same quality gate on Ubuntu and macOS, plus a Windows smoke matrix and an offline doctor job.

## Run the App

```bash
docflow serve
```

## Frontend Styles

The browser shell is split into a small `frontend/index.html`, reusable markup in `frontend/partials/`, product styles in `frontend/app.css`, and focused JavaScript files in `frontend/js/`. Stream parsing is built from TypeScript in `frontend/src/` and tested with Vitest.

Committed CSS and generated frontend assets are enough to run the app. Rebuild styles or frontend generated assets only after related changes:

```bash
npm install
npm run build:css
npm run build:frontend
```

## Project Rule

Normal browser UI must feel like a finished personal knowledge product. Developer-only commands, repair instructions, scripts, and recovery details belong in docs, CLI, tests, or internal implementation, not in the user-facing app.

## Configuration And Migration

- Runtime dependencies must live in the API application context; do not add new module-level state for the same object.
- New query behavior should be configurable through the `query:` section when it affects relevance, answer limits, or refusal thresholds.
- SQLite schema changes belong in store migration code and need tests that prove an existing database can open after the change.
- Config changes should keep old defaults working. If a rebuild or reindex is unavoidable, document that clearly before release.
