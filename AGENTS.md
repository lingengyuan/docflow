# Repository Guidelines

## Project Structure & Module Organization

DocFlow is a local-first Python document Q&A app. `main.py` is the entry point. Core code lives in `src/`: `src/api/` serves FastAPI, `src/ingest/` handles parsing and storage, and `src/query/` handles retrieval and answers. Tests are in `tests/`. The browser UI is `frontend/index.html`. Notes and plans live under `docs/` and `plans/`. Runtime data such as `docflow.db`, Qdrant storage, and generated indexes are not source files.

## Build, Test, and Development Commands

- `python -m venv .venv && source .venv/bin/activate`: create a virtual environment.
- `pip install -r requirements.txt`: install pinned dependencies.
- `docker run -d --name qdrant -p 6333:6333 qdrant/qdrant`: start Qdrant.
- `python main.py serve`: run the local web app on port 8000.
- `python main.py scan`: scan configured watched folders from `config.yaml`.
- `python main.py ingest /path/to/file.pdf`: ingest one file manually.
- `python main.py benchmark README.md docs/HANDOFF-v3.md`: run a dry-run benchmark.
- `.venv/bin/python -m pytest`: run the full test suite.

## Project Principles

- First principles: solve the real problem, not copied patterns.
- DRY: remove meaningful duplication without hiding intent.
- Easy Change: keep modules small, clear, and localized.
- Orthogonality: keep ingestion, storage, retrieval, API, and UI separate.
- No masking fallbacks: fallback behavior must not hide failures, data loss, stale data, or reduced answer quality. In project-rule terms, the fallback principle is to expose failures clearly instead of masking reduced answer quality.
- User-facing UI must feel like a finished personal knowledge product, not a developer console or prototype. Follow the saved UI references exactly for information architecture, visual hierarchy, spacing, and wording. Do not expose code, shell commands, script names, dry-run/repair/restore/install/doctor/browser-acceptance wording, maintenance commands, recovery advice, or copyable terminal commands in the normal browser UI. Keep those capabilities in docs, CLI, tests, or internal implementation only.
- For UI reference phases, visual parity is a blocking requirement. After changes, generate a fresh reference-vs-current screenshot comparison for every saved UI reference. If any page still differs in layout, state, density, wording, or user-facing controls, the phase is not complete and the next phase must not start.
- After every planned phase, request an independent subagent review before reporting completion or starting the next phase. The subagent must score the project with the strictest standard across code quality, architecture, UI/UX, product fit, aesthetics, usability, reliability, maintainability, open-source readiness, and documentation consistency. Treat that score as a guardrail against optimistic self-assessment or wasted work; if the review shows the phase did not materially improve the project, adjust the next plan before continuing.
- Unit tests are required for behavior changes.
- End-to-end testing is required after feature work; exercise the user flow before handoff.

## Coding Style & Naming Conventions

Use Python 3.11+ and the existing style: 4-space indentation, snake_case for functions and variables, PascalCase for classes, and focused modules. Prefer `pathlib.Path` for filesystem work and keep configuration in `config.yaml`. Add type hints for new public helpers.

## Testing Guidelines

Tests use `pytest`. Name files `test_*.py`, name test functions `test_*`, and keep fakes close to the behavior they support. Prefer deterministic fakes over live model calls. Run `.venv/bin/python -m pytest` before handoff.

## Phase Handoff Rule

When working from a phased plan, finish phases in order. After completing each phase, validate the work, request the independent subagent review required above, then write or update a local phase handoff document under `docs/history/phase-handoffs/`, commit, and push the completed phase to GitHub before reporting completion or starting the next phase. Phase handoff documents are local session records only: do not stage, commit, or push new or updated phase handoff documents unless the user explicitly asks for that. The handoff must state the completed scope, changed files, validation commands and results, independent review score and findings, known limitations, and the exact next phase tasks so a later session can resume without rediscovering context.

## Commit & Pull Request Guidelines

Recent commits use short imperative messages, often with scoped prefixes such as `docs:` or `feat(embedding):`. Keep commits focused. Pull requests should include a summary, reason, test results, and screenshots for visible UI changes. Keep README commands aligned with real `main.py` entry points.

## Security & Configuration Tips

DocFlow is designed for local private data. Do not commit local databases, Qdrant storage, caches, watched-folder contents, or machine-specific secrets. Review `config.yaml` before changing paths, model backends, or watched extensions.
