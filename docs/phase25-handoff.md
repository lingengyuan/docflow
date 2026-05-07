# Phase 25 Handoff: Repeatable Browser Acceptance

Date: 2026-05-07

## Status

Complete.

## Prior Commit

Phase 24 was committed before Phase 25 work started:

```bash
git log -1 --oneline
# 7f08d5f feat(setup): complete phase24 local install plan
```

## Scope

Phase 25 turned the manual browser smoke pass into a repeatable command.

Implemented:

- Added `python main.py browser-acceptance`.
- Added `scripts/run_browser_acceptance.py`.
- Added `src/quality/browser_acceptance.py`.
- The command opens the running DocFlow app in Chromium and checks:
  - Chat page shell, question input, send button, and scope selector.
  - Library page shell, refresh button, scan button, upload zone, filters, and table.
  - Notes page shell, note form, URL import form, knowledge output panel, and recent list area.
  - Settings page shell, health button, model list, watched-folder list, recovery list, and current maintenance commands.
- The command saves screenshots to `output/playwright/phase25-browser-acceptance`.
- Added tests for the browser acceptance plan, failure reporting, and unreachable-server handling.
- Added Playwright, pyee, and greenlet to `requirements.txt` so the command can import the browser runner after dependency install.
- Updated README, local deployment guide, changelog, and maturity scorecard.

## Changed Files

- `CHANGELOG.md`
- `README.md`
- `docs/LOCAL_DEPLOYMENT.md`
- `docs/phase25-handoff.md`
- `eval/phase11_maturity_dimensions.json`
- `main.py`
- `requirements.txt`
- `scripts/run_browser_acceptance.py`
- `src/quality/browser_acceptance.py`
- `tests/test_browser_acceptance.py`
- `tests/test_static_assets.py`

## Validation Results

Targeted tests:

```bash
.venv/bin/python -m pytest tests/test_browser_acceptance.py tests/test_static_assets.py
# 20 passed, 5 warnings
```

Full tests:

```bash
.venv/bin/python -m pytest
# 203 passed, 5 warnings
```

Browser acceptance:

```bash
.venv/bin/python main.py browser-acceptance --json
# status=ok
# passed=45
# failed=0
# screenshots_dir=/Users/hughlin/Projects/docflow/output/playwright/phase25-browser-acceptance
```

Screenshot artifacts:

```text
output/playwright/phase25-browser-acceptance/01-chat.png
output/playwright/phase25-browser-acceptance/02-library.png
output/playwright/phase25-browser-acceptance/03-notes.png
output/playwright/phase25-browser-acceptance/04-settings.png
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
# overall_score=8.69/9.0
# ui_usability=8.6
# testing_discipline=9.0
# product_maturity=8.9
```

CSS build:

```bash
npm run build:css
# passed
# warning: Browserslist caniuse-lite data is outdated
```

README bilingual parity and diff hygiene:

```bash
/Users/hughlin/.agents/skills/readme-maintainer/scripts/check_bilingual_readme.sh README.md
# Bilingual README parity check passed.

git diff --check
# passed
```

## Known Limitations

- The command requires the DocFlow web service to already be running.
- The command requires Playwright's Chromium browser runtime. If it is missing,
  run `.venv/bin/python -m playwright install chromium`.
- Phase 25 checks that the four main pages render and expose expected controls;
  it does not yet exercise full data-changing flows such as upload, scan,
  webpage import, note save, or knowledge output generation in the browser.
- Screenshots are runtime artifacts under `output/playwright/` and are not
  source files.

## Next Phase Tasks

Phase 26 should build on the new browser acceptance command:

1. Add data-changing browser flows with temporary files: upload, scan, URL import,
   note save, and knowledge output generation.
2. Add failure-state checks for missing Qdrant, missing Ollama, missing optional
   VLM/OCR, and weak retrieval evidence.
3. Decide whether the next packaging step should be a stronger bootstrap script,
   a desktop launcher, or a signed macOS app.
4. Keep the release checklist: full tests, browser acceptance, sample suite,
   restore drill, consistency check, maturity evaluation, README parity, and
   `git diff --check`.
5. Write `docs/phase26-handoff.md` before reporting Phase 26 completion.
