# Phase 9 Handoff: One-Command Startup

Date: 2026-05-03

Update: Phase 9 follow-up work is now documented in `docs/phase9-followup-handoff.md`. It covers the runtime health check fix, launchd service commands, the installed background service validation, favicon, and final browser validation.

## Status

Phase 9 is implemented for optimization plan section 2.13: one-command startup and background service foundation.

The stable path is now:

```bash
.venv/bin/python main.py doctor
.venv/bin/python main.py start
scripts/start.sh
```

`doctor` checks the local machine without starting the service. `start` runs the same startup checks, tries to start an existing `qdrant` Docker container when Qdrant is down, prints the local URL, and then starts the FastAPI service.

Launchd/background auto-start was completed in the Phase 9 follow-up. The one-command path still remains the underlying service command, and `docs/phase9-followup-handoff.md` is now the source of truth for daily background-service usage.

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

- The runtime SQLite health false positive found during Phase 9 was fixed in the follow-up handoff.
- The missing favicon found during Phase 9 was fixed in the follow-up handoff.
- Browser verification still produced the existing Tailwind CDN warning. It does not block the tested page flow.
- `start` can only auto-start Qdrant when a Docker container named `qdrant` already exists. If it does not exist, the command prints the exact `docker run` command instead of creating the container automatically.
- Launchd background startup was added and verified in the follow-up handoff.

## Next Phase

The original optimization plan's Phase 9 scope is complete. There is no numbered Phase 10 in the current plan.

Recommended next tasks:

1. Use `docs/phase9-followup-handoff.md` as the current source of truth.
2. After the next login, verify the installed background service again with `python main.py service status` and the service logs.
3. If background scans feel too heavy after login, improve queue pacing and startup scan behavior next.
