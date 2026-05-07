# Phase 24 Handoff: Local Install Plan

Date: 2026-05-07

## Status

Complete.

## Prior Commit

Phase 23 was committed before Phase 24 work started:

```bash
git log -1 --oneline
# 976da23 fix(ingest): complete phase23 id hardening
```

## Scope

Phase 24 lowered the first-run and maintenance cost for a local DocFlow setup.

Implemented:

- Added `python main.py install-local`.
- Added `scripts/install_local.sh`.
- The install command is safe by default:
  - Default mode is a dry-run plan.
  - `--apply` is required before commands are executed.
  - `--with-service` opts into installing the launchd service during apply.
  - `--skip-deps` skips dependency installation for repeat checks.
- The local install plan covers:
  - Virtual environment creation when missing.
  - Python dependency installation unless skipped.
  - Startup requirement check.
  - Disposable restore drill.
  - Vector ID repair preview.
  - Launchd service install preview by default.
- Settings now shows the current maintenance commands:
  - `python main.py install-local`
  - `python main.py restore-drill`
  - `python main.py repair-ids --dry-run`
  - `python main.py backup --dry-run`
- Updated README, local deployment guide, changelog, and maturity scorecard.

## Changed Files

- `CHANGELOG.md`
- `README.md`
- `docs/LOCAL_DEPLOYMENT.md`
- `docs/phase24-handoff.md`
- `eval/phase11_maturity_dimensions.json`
- `frontend/index.html`
- `main.py`
- `scripts/install_local.sh`
- `src/maintenance/local_install.py`
- `tests/test_local_install.py`
- `tests/test_static_assets.py`

## Validation Results

Targeted tests:

```bash
.venv/bin/python -m pytest tests/test_local_install.py tests/test_static_assets.py
# 19 passed, 5 warnings
```

Full tests:

```bash
.venv/bin/python -m pytest
# 199 passed, 5 warnings
```

Local install dry-run:

```bash
.venv/bin/python main.py install-local
# status=dry_run
# steps=install_python_deps,startup_check,restore_drill,repair_ids_preview,service_dry_run
```

Wrapper script dry-run:

```bash
scripts/install_local.sh --skip-deps
# status=dry_run
# steps=startup_check,restore_drill,repair_ids_preview,service_dry_run
```

CSS build:

```bash
npm run build:css
# passed
# warning: Browserslist caniuse-lite data is outdated
```

Sample suite:

```bash
.venv/bin/python main.py sample-suite --json
# passed=5
# failed=0
```

Restore drill:

```bash
.venv/bin/python main.py restore-drill --json
# status=ok
# passed=11
# failed=0
# files=265
# chunks=10099
# chunks_jsonl=10099
# quick_check=ok
# duplicate_qdrant_ids=[]
# id_counter.value=29345
# id_counter.expected_min=29345
```

Live consistency:

```bash
.venv/bin/python main.py check --json
# status=ok
# sqlite_chunks=10099
# qdrant_points=10099
# id_counter.status=ok
# duplicate_qdrant_ids=[]
```

Repair dry-run:

```bash
.venv/bin/python main.py repair-ids --dry-run
# status=dry_run
# id_counter.status=ok
# duplicate_qdrant_ids=[]
# affected_files=[]
# actions=[]
```

Maturity score:

```bash
.venv/bin/python main.py maturity-eval --skip-retrieval --json
# overall_score=8.68/9.0
# setup_ops=9.0
# testing_discipline=9.0
# product_maturity=8.9
```

README bilingual parity:

```bash
/Users/hughlin/.agents/skills/readme-maintainer/scripts/check_bilingual_readme.sh README.md
# Bilingual README parity check passed.
```

Browser smoke:

```text
Opened http://127.0.0.1:8000/ and checked Settings.
The maintenance command panel shows install-local, restore-drill,
repair-ids --dry-run, and backup --dry-run.
```

Diff hygiene:

```bash
git diff --check
# passed
```

## Known Limitations

- `install-local --apply` was not executed during validation because it can
  install dependencies and change the user's launchd service. The dry-run plan
  and unit tests verify the command sequence.
- This is not yet a packaged macOS app or signed installer.
- The install plan assumes Qdrant and local models are managed by the existing
  project commands and deployment guide; it does not create or download every
  external dependency by itself.
- Settings shows the maintenance commands, but does not execute them from the
  browser.

## Next Phase Tasks

Phase 25 should continue product maturity from this safer local setup baseline:

1. Decide whether to build a packaged desktop launcher, a signed installer, or
   a stronger one-command bootstrap script.
2. Add copy buttons or a guided maintenance panel if Settings should make these
   commands easier to run.
3. Expand browser regression into a repeatable multi-page acceptance command.
4. Keep running the release checklist: full tests, restore drill, consistency
   check, sample suite, maturity evaluation, README parity, browser smoke, and
   `git diff --check`.
5. Write `docs/phase25-handoff.md` before reporting Phase 25 completion.
