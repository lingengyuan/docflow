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

This runs compile checks, ruff, mypy, and pytest. Use `docs/release.md` before publishing a release.

Frontend and dependency checks:

```bash
npm run check:frontend
npm run audit:frontend
.venv/bin/python -m pip_audit -r requirements.txt -r requirements-dev.txt -r requirements-vision.txt
```

CI runs the same quality gate on Ubuntu and macOS, plus a Windows smoke matrix and an offline doctor job.

## Run the App

```bash
docflow serve
```

## Frontend Styles

Committed CSS is enough to run the app. Rebuild styles only after frontend style changes:

```bash
npm install
npm run build:css
```

## Project Rule

Normal browser UI must feel like a finished personal knowledge product. Developer-only commands, repair instructions, scripts, and recovery details belong in docs, CLI, tests, or internal implementation, not in the user-facing app.
