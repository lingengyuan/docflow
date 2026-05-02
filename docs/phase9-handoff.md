# Phase 9 Handoff: One-Command Startup

Date: 2026-05-03

Update: Phase 9 follow-up work is now documented in `docs/phase9-followup-handoff.md`. It covers the runtime health check fix, launchd service commands, favicon, and final browser validation.

## Status

Phase 9 is implemented for optimization plan section 2.13: one-command startup and background service foundation.

The stable path is now:

```bash
.venv/bin/python main.py doctor
.venv/bin/python main.py start
scripts/start.sh
```

`doctor` checks the local machine without starting the service. `start` runs the same startup checks, tries to start an existing `qdrant` Docker container when Qdrant is down, prints the local URL, and then starts the FastAPI service.

Launchd/background auto-start is intentionally not implemented yet. The one-command path is now in place and should be used for a few normal sessions before adding OS-level background service behavior.

## Completed Scope

- Added a startup preflight module for Python dependencies, config, SQLite, Qdrant, Ollama, and app port checks.
- Added `python main.py doctor` for local dependency diagnosis.
- Added `python main.py start` for checked startup.
- Added `scripts/start.sh` as a shell-friendly one-command wrapper that prefers `.venv/bin/python`.
- Added `--check-only` for startup verification without loading models or starting the long-running service.
- Added structured JSON output for startup checks with `--json`.
- Kept SQLite health failures visible but not startup-blocking, because the app can still start and show the health state in the UI.
- Updated README English and Chinese sections to use the new startup path.
- Added deterministic startup tests.

## Changed Files

- `src/maintenance/startup.py`
  - New startup checks and launcher helpers.
- `main.py`
  - New `doctor` and `start` commands.
- `scripts/start.sh`
  - New executable startup wrapper.
- `tests/test_startup.py`
  - Tests for aggregate status, SQLite missing DB behavior, port conflicts, Qdrant blocker reporting, Qdrant Docker guidance, and formatted output.
- `README.md`
  - English and Chinese quick-start and development commands updated.
- `docs/phase8-handoff.md`
  - Points future readers to this Phase 9 handoff.
- `docs/phase9-handoff.md`
  - This handoff.

## Validation

Commands run:

```bash
.venv/bin/python -m pytest tests/test_startup.py
.venv/bin/python -m pytest
.venv/bin/python main.py doctor --json
.venv/bin/python main.py start --check-only --json
scripts/start.sh --check-only --port 8011
node - <<'NODE'
const fs = require('fs');
const vm = require('vm');
const html = fs.readFileSync('frontend/index.html', 'utf8');
const scripts = [...html.matchAll(/<script(?![^>]*src=)[^>]*>([\s\S]*?)<\/script>/gi)].map(m => m[1]);
scripts.forEach((script, index) => new vm.Script(script, { filename: `frontend/index.html#script${index + 1}` }));
console.log(`checked ${scripts.length} inline scripts`);
NODE
.venv/bin/python main.py start --port 8011
curl -s http://127.0.0.1:8011/api/health
/Users/hughlin/.codex/skills/readme-maintainer/scripts/check_bilingual_readme.sh README.md
```

Results:

- `tests/test_startup.py`: 6 passed.
- Full test suite: 111 passed, 5 warnings.
- README bilingual parity check: passed.
- Frontend inline script syntax check: checked 2 inline scripts.
- `main.py doctor --json`: returned `status: ok`, `can_start: true`.
- `main.py start --check-only --json`: returned `status: ok`, `can_start: true`.
- `scripts/start.sh --check-only --port 8011`: returned all startup checks as ok.
- Real startup on port 8011:
  - Startup check printed ok.
  - Service reached `Application startup complete`.
  - Uvicorn served `http://0.0.0.0:8011`.
  - Browser opened `http://127.0.0.1:8011/` with page title `DocFlow`.
  - Browser snapshot showed the main chat UI and dependency status panel.

## Known Limitations

- During the real startup smoke test, background scanning began indexing changed files. While that was running, `/api/health` and the UI dependency panel reported SQLite as `unavailable` with `malformed inverted index for FTS5 table main.chunks_fts`.
- After the service was stopped, `.venv/bin/python main.py doctor --json --port 8011` returned SQLite `quick_check: ok` again.
- This means the one-command startup path is working, but the runtime health check can still show SQLite unavailable while indexing is active. Treat this as a runtime index/health issue for a later phase, not a startup-command blocker.
- Browser verification produced only the existing Tailwind CDN warning and a missing `favicon.ico` 404. Neither blocked the tested page flow.
- `start` can only auto-start Qdrant when a Docker container named `qdrant` already exists. If it does not exist, the command prints the exact `docker run` command instead of creating the container automatically.
- `launchd` or menu-bar style background startup is still deferred.

## Next Phase

The original optimization plan's Phase 9 scope is complete. There is no numbered Phase 10 in the current plan.

Recommended next tasks:

1. Decide whether to add `launchd` auto-start now that `main.py start` is stable.
2. Investigate the runtime SQLite FTS5 health behavior during background indexing, especially why API health can report malformed FTS while standalone `doctor` reports ok after shutdown.
3. Consider adding a tiny favicon or suppressing the harmless browser 404.
4. If background service behavior is added, keep `main.py start` as the underlying command and add tests around the generated plist or install instructions.
5. Run `.venv/bin/python -m pytest`, `main.py start --check-only`, a real browser startup check, README parity check, `git diff --check`, and a new handoff before reporting completion.
