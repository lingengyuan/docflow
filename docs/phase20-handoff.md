# Phase 20 Handoff: Fully Local UI Assets

Date: 2026-05-07

## Status

Complete.

## Scope

Phase 20 removed the last known runtime remote UI asset from the main browser
app. The page no longer requests Google Fonts for Material Symbols; icons are
rendered locally as inline SVG while preserving the existing button layout and
dynamic icon state changes.

Implemented:

- Removed the Google Fonts Material Symbols stylesheet from `frontend/index.html`.
- Added a local SVG icon renderer for the existing `.material-symbols-outlined`
  icon spans.
- Updated icon state changes for copy, save, refresh, and scan actions to use
  the local renderer.
- Kept dynamically inserted icons working through a mutation observer.
- Fixed the mobile Chat status pill so the status text does not wrap.
- Added static regression coverage for remote icon-font requests.
- Updated README, local deployment docs, changelog, and maturity scoring to
  reflect fully local browser UI assets.

## Changed Files

- `CHANGELOG.md`
- `README.md`
- `docs/LOCAL_DEPLOYMENT.md`
- `docs/phase20-handoff.md`
- `eval/phase11_maturity_dimensions.json`
- `frontend/index.html`
- `frontend/styles.css`
- `tests/test_static_assets.py`

## Validation Results

CSS build:

```bash
npm run build:css
# passed
# Browserslist printed an outdated caniuse-lite warning.
```

Frontend script syntax:

```bash
perl -0ne 'print $1 if /<script>(.*)<\/script>/s' frontend/index.html | node --check -
# passed
```

Targeted static asset tests:

```bash
.venv/bin/python -m pytest tests/test_static_assets.py
# 15 passed, 5 warnings
```

Full tests:

```bash
.venv/bin/python -m pytest
# 181 passed, 5 warnings
```

Live project consistency:

```bash
.venv/bin/python main.py check --json
# status=ok
# sqlite_chunks=10035
# qdrant_points=10035
# missing_points=0
# orphan_points=0
# file_chunk_mismatches=0
# missing_source_files=0
```

Maturity score:

```bash
.venv/bin/python main.py maturity-eval --skip-retrieval --json
# overall_score=8.57/9.0
```

Browser validation:

```bash
.venv/bin/python <one-off Playwright Phase 20 browser check>
# desktop icons: 90 / 90 rendered as SVG
# dynamic icons after navigation: 1146 / 1146 rendered as SVG
# mobile icons: 90 / 90 rendered as SVG
# no Google Fonts, Material Symbols, or Tailwind CDN requests
# no console errors
# mobile horizontal overflow=false
# mobile status label did not wrap
```

Screenshots from the browser check:

- `/tmp/docflow-phase20-chat.png`
- `/tmp/docflow-phase20-mobile.png`

Diff hygiene:

```bash
git diff --check
# passed
```

## Known Limitations

- The Codex in-app browser backend was unavailable in this session, so browser
  validation used local Playwright against the same running app.
- `docs/code.html` remains a historical prototype containing old remote asset
  references; it is not served as the runtime browser UI.
- This phase did not rerun the fixed retrieval evidence set because retrieval
  behavior did not change.
- This phase does not package DocFlow as a desktop app or one-click installer.

## Next Phase Tasks

Phase 21 should be the real sample suite phase:

1. Add repeatable local sample fixtures or scripts for images, screenshots,
   scanned PDFs, and table-heavy documents.
2. Validate that OCR, VLM-gated image handling, table chunking, source previews,
   and generated knowledge outputs work from those samples.
3. Keep samples small and safe to commit, or generate them during tests if real
   files would be too large.
4. Run `.venv/bin/python -m pytest`, `.venv/bin/python main.py check --json`,
   and a browser workflow that uses the sample files.
5. Write `docs/phase21-handoff.md` with scope, changed files, validation
   results, limitations, and exact next tasks before reporting completion.
