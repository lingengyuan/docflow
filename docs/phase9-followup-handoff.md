# Phase 9 Follow-Up Handoff: Startup Reliability and Background Service

Date: 2026-05-03

## Status

The follow-up plan after Phase 9 is implemented in the requested order:

1. Runtime SQLite health false positive fixed.
2. macOS background service commands added.
3. Favicon added.
4. Real startup and browser validation completed.
5. Background service installed and verified on this machine.

## Completed Scope

- Changed live `/api/health` SQLite checking to use a lightweight runtime read/write check when the app is running.
- Kept full SQLite integrity checking in `python main.py doctor --strict`.
- Added launchd helper support through `python main.py service install|status|uninstall`.
- Added dry-run support for service installation and removal.
- Added `scripts/service.sh` as a shell wrapper for service commands.
- Added a local SVG favicon and linked it from the frontend.
- Updated README English and Chinese sections for runtime health behavior, service commands, and project structure.
- Added tests for runtime SQLite health, launchd plist/service dry-runs, and favicon serving.
- Installed and verified the launchd background service on this machine.

## Changed Files

- `src/api/app.py`
  - Runtime SQLite health no longer runs `PRAGMA quick_check` while the app is active.
- `src/maintenance/launchd.py`
  - New macOS launchd plist generation, install, status, and uninstall helpers.
- `main.py`
  - New `service` command group.
- `scripts/service.sh`
  - New service command wrapper.
- `frontend/index.html`
  - Links the local favicon.
- `frontend/favicon.svg`
  - New local icon asset.
- `tests/test_api_health.py`
  - Added coverage that runtime health skips deep quick_check.
- `tests/test_launchd.py`
  - Added plist and dry-run coverage.
- `tests/test_static_assets.py`
  - Added favicon serving coverage.
- `README.md`
  - Updated English and Chinese usage/docs.
- `docs/phase9-handoff.md`
  - Points readers to this follow-up handoff.
- `docs/phase9-followup-handoff.md`
  - This handoff.

## Validation

Commands run:

```bash
.venv/bin/python -m pytest tests/test_api_health.py
.venv/bin/python -m pytest tests/test_launchd.py tests/test_startup.py
.venv/bin/python -m pytest tests/test_static_assets.py
.venv/bin/python main.py service install --dry-run --port 8011
scripts/service.sh install --dry-run --port 8011
.venv/bin/python main.py service install
.venv/bin/python main.py service status
lsof -nP -iTCP:8000 -sTCP:LISTEN
tail -80 ~/Library/Logs/docflow/docflow.out.log
tail -80 ~/Library/Logs/docflow/docflow.err.log
/Users/hughlin/.codex/skills/readme-maintainer/scripts/check_bilingual_readme.sh README.md
.venv/bin/python -m pytest
.venv/bin/python main.py start --check-only --port 8011
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
curl -I -s http://127.0.0.1:8011/favicon.svg
curl -s -X POST http://127.0.0.1:8011/api/query -H 'Content-Type: application/json' -d '{"question":"DocFlow 是什么？"}'
curl -s http://127.0.0.1:8000/api/health
curl -I -s http://127.0.0.1:8000/favicon.svg
curl -s -X POST http://127.0.0.1:8000/api/query -H 'Content-Type: application/json' -d '{"question":"DocFlow 当前后台服务是否可用？"}'
browser open http://127.0.0.1:8011
browser open http://localhost:8000
browser click health panel, Files, and History
```

Results:

- Runtime SQLite health test: passed.
- Launchd tests: passed.
- Favicon static asset test: passed.
- Full test suite: 116 passed, 5 warnings.
- Startup check: returned all startup checks ok on port 8011.
- Service dry-run: printed the expected launchd plist and launchctl commands without writing the plist.
- Service install: wrote `~/Library/LaunchAgents/com.docflow.local.plist`; `launchctl bootstrap` and `launchctl kickstart` returned success. The initial `bootout` returned code 5 only because there was no old service to unload.
- Service status: loaded and running under launchd with PID 9109.
- Port 8000: listening on `127.0.0.1` from the launchd-managed Python process.
- Service logs: startup check completed, Uvicorn started, and no fatal startup error was present in `~/Library/Logs/docflow/docflow.out.log` or `docflow.err.log`.
- README bilingual parity check: passed.
- Real app startup: reached `Application startup complete` and served `http://0.0.0.0:8011`.
- Runtime `/api/health` during background indexing:
  - Overall `status: degraded` because optional enhanced/VLM model caches are missing.
  - SQLite `status: ok`.
  - Query and ingest capabilities were both `true`.
- `favicon.svg`: returned HTTP 200 with `content-type: image/svg+xml`.
- Background service `/api/health` on port 8000:
  - Overall `status: degraded` because optional enhanced/VLM model caches are missing.
  - SQLite, Qdrant, and Ollama were all `ok`.
  - Query and ingest capabilities were both `true`.
- Browser validation:
  - Page opened with title `DocFlow`.
  - Health panel showed SQLite ok, Qdrant ok, Ollama ok, and models degraded.
  - Files view rendered the indexed file table and live queue progress.
  - History view rendered the new `DocFlow 是什么？` query.
  - Console had 0 errors and 1 existing Tailwind CDN warning.
- Real query:
  - `POST /api/query` returned an answer with citations.
- Background service real query:
  - `POST /api/query` on port 8000 returned an answer with citations.
- Post-shutdown `main.py doctor --json --port 8011`: returned `status: ok` and SQLite `quick_check: ok`.

## Known Limitations

- The launchd service is currently installed and running on this machine. Use `python main.py service uninstall` if it should be removed.
- The service expects Docker/Qdrant and optional Ollama to be available in the user session. The underlying command is still `python main.py start`, so startup checks remain visible in logs.
- Runtime health is still `degraded` on this machine because optional enhanced/VLM model caches are missing. Core query and ingest capabilities are available.

## Next Tasks

1. Use `docs/phase10-validation-issues.md` for the latest full-product validation results.
2. Use `docs/phase10-optimization-plan.md` for the next stability optimization plan.
3. After the next login, run `python main.py service status` and inspect `~/Library/Logs/docflow/docflow.out.log` plus `docflow.err.log`.
4. If background startup is no longer wanted, run `python main.py service uninstall`.
