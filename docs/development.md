# Development

## Setup

For daily use, prefer one of these supported paths:

- Source checkout: clone the repository, create a virtual environment, install with `pip install -e .`, start Qdrant, and run `docflow serve`.
- Docker from source: run `docker compose up --build` to build the app container and start Qdrant together.
- Docker image release: after a tagged image is published, run `docker compose -f docker-compose.image.yml up` to use `ghcr.io/lingengyuan/docflow` without rebuilding locally.

DocFlow is not on PyPI yet. The wheel now packages browser assets, configuration templates, and public docs, and `scripts/package_smoke.py` verifies an installed wheel can find them. Until PyPI publishing is enabled, use source checkout or Docker for daily use.

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

First-run storage expectations:

- App and Python runtime: roughly 0.5 GB for the current Docker image build on the validation machine; local virtual environments vary by platform and dependency cache.
- Qdrant and SQLite: grow with your indexed files and chunks.
- Local answer models: managed by Ollama, LM Studio, or your chosen local model tool; a common 7B model is usually 4-5 GB.
- Optional image understanding and Apple Silicon MLX packages increase install size and should be installed only when needed.

## Run Tests

```bash
scripts/run_ci.sh
```

This runs compile checks, ruff, mypy, pytest, frontend syntax checks, frontend unit tests, the frontend build, dependency audit, and the installed-wheel smoke test. Use `docs/release.md` before publishing a release.

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
- `config.example.yaml` and `config.docker.yaml` must carry the same user-facing quality settings. Docker may change service hosts and model backend, but it must not silently use different answer thresholds.
- Query quality settings mean:
  - `min_rerank_score`: how strong the best reranked match must be before DocFlow writes a full answer.
  - `min_vector_score`: the equivalent floor when vector score is the available signal.
  - `default_answer_chunks` / `min_answer_chunks`: how many source chunks are allowed to support an answer before the rest becomes related material.
  - `related_notes_limit`: how many extra related sources may be shown without being treated as direct evidence.
  - `fallback_mode`: must stay visibly degraded; model failures may show source snippets, but must not look like a normal answer.
- SQLite schema changes belong in store migration code and need tests that prove an existing database can open after the change.
- Config changes should keep old defaults working. If a rebuild or reindex is unavoidable, document that clearly before release.
- Vector-store changes must say whether existing Qdrant collections are compatible, need a `rebuild --qdrant-only`, or require a full reindex.
- Model changes must say whether old embeddings, reranker scores, or answer behavior remain comparable.
