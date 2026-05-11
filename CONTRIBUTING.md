# Contributing to DocFlow

DocFlow is a local-first personal knowledge assistant. Contributions should keep the app private by default, easy to run, and understandable for users who are not developers.

## Before You Start

- Read `README.md`, `docs/architecture.md`, `docs/privacy.md`, and `docs/development.md`.
- Use Python 3.11 or newer.
- Create a local virtual environment and install the project with `pip install -e .`.
- Start Qdrant with `docker compose up -d qdrant` before running flows that need vectors.

## Development Rules

- Keep changes focused and small enough to review.
- Prefer existing modules and patterns over new framework choices.
- Keep normal browser UI copy user-facing. Do not expose shell commands, repair commands, internal script names, or developer-only wording in the product UI.
- Add or update tests for behavior changes.
- Update README or docs when user-facing behavior, commands, configuration, or validation results change.
- Do not commit local databases, Qdrant storage, generated indexes, model caches, local notes, or secrets.

## Checks Before Pull Request

Run the checks that match your change:

```bash
.venv/bin/python -m pytest -q
.venv/bin/python -m ruff check .
.venv/bin/python -m mypy
docflow browser-acceptance
docflow doctor --offline
```

All listed checks are expected to pass before a pull request is ready for review.

## Pull Request Expectations

Include:

- What changed.
- Why it changed.
- Tests or checks run.
- Screenshots for visible UI changes.
- Privacy impact when network access, model backends, watched folders, or external imports are touched.

## Reporting Issues

Use the GitHub issue template when possible. Include your operating system, Python version, DocFlow commit, steps to reproduce, and logs or screenshots if available.
