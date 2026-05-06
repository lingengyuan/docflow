# Phase 18 Handoff: Final Hardening and Release Polish

Date: 2026-05-06

## Status

Complete.

## Scope

Phase 18 completed the final acceptance, local deployment, release documentation, and product-polish pass for the current optimization plan.

Implemented:

- Replaced runtime Tailwind CDN usage with committed local CSS.
- Added Tailwind build files and an `npm run build:css` command.
- Added `LICENSE`, `CHANGELOG.md`, and `docs/LOCAL_DEPLOYMENT.md`.
- Added README screenshots, local deployment references, release target, and CSS rebuild guidance.
- Fixed configured code-text file support in the parser registry.
- Updated `config.yaml` and upload accept lists so Markdown, code text, and image extensions are aligned.
- Added parser-registry coverage for configured code extensions and VLM-gated image extensions.
- Updated the rolling maturity scorecard to the Phase 18 score.
- Added `docs/phase18-final-acceptance.md`.

## Changed Files

- `.gitignore`
- `README.md`
- `CHANGELOG.md`
- `LICENSE`
- `config.yaml`
- `package.json`
- `package-lock.json`
- `tailwind.config.js`
- `frontend/index.html`
- `frontend/tailwind.css`
- `frontend/styles.css`
- `src/ingest/parsers/__init__.py`
- `src/ingest/pipeline.py`
- `tests/test_parser_registry.py`
- `tests/test_static_assets.py`
- `eval/phase11_maturity_dimensions.json`
- `docs/LOCAL_DEPLOYMENT.md`
- `docs/phase18-final-acceptance.md`
- `docs/phase18-handoff.md`
- `docs/phase18-chat-desktop.png`
- `docs/phase18-library-desktop.png`
- `docs/phase18-notes-desktop.png`
- `docs/phase18-settings-desktop.png`
- `docs/phase18-settings-mobile.png`

## Validation Results

CSS build:

```bash
npm run build:css
# passed
# Browserslist printed an outdated caniuse-lite warning.
```

Targeted tests:

```bash
.venv/bin/python -m pytest tests/test_parser_registry.py tests/test_static_assets.py tests/test_maturity_eval.py
# 20 passed, 5 warnings
```

Full tests:

```bash
.venv/bin/python -m pytest
# 174 passed, 5 warnings
```

Live service health:

```bash
launchctl kickstart -k gui/$(id -u)/com.docflow.local
curl -fsS http://127.0.0.1:8000/api/health
# status=ok
# SQLite ok, Qdrant ok, Ollama ok, local model cache ok
```

Browser validation:

```bash
.venv/bin/python <one-off Playwright Phase 18 acceptance check>
# passed
# screenshots written under docs/phase18-*.png
```

Sample validation:

```bash
.venv/bin/python <one-off Phase 18 table/OCR sample check>
# table_chunk_types=["table", "table_summary"]
# scanned PDF OCR returned text from a temporary image-only PDF
```

Restore rehearsal:

```bash
.venv/bin/python main.py backup --output /tmp/docflow-phase18-backups --keep 2
.venv/bin/python main.py restore-plan /tmp/docflow-phase18-backups/docflow-backup-20260506-125016-284259.tar.gz
# archive contents and extracted SQLite/chunks counts matched manifest
```

Maturity and fixed retrieval evaluation:

```bash
.venv/bin/python main.py maturity-eval --no-rerank --json
# overall_score=8.51/9.0
# retrieval_eval=11/27 passed
```

Final consistency check:

```bash
.venv/bin/python main.py check --json
# status=ok
# sqlite_chunks=10026
# qdrant_points=10026
# no missing points, orphan points, file chunk mismatches, or missing source files
```

Note: the restore rehearsal archive was created before the final background code-file scan completed, so the backup evidence has `9850` chunks while the final live index has `10026` chunks.

Browser Use note:

- The Browser Use Node REPL tool was discovered, but the in-app-browser backend reported that no Codex IAB browser backend was available.
- Playwright was used as the browser validation fallback.

## Known Limitations

- The browser still loads Material Symbols from Google Fonts.
- Fixed retrieval evaluation is still only `11 / 27` without reranking.
- Live VLM image parsing was not completed because the temporary run began model-file fetching; health reports the configured VLM as available, but this handoff does not count live image parsing as fully verified.
- Restore testing was non-destructive; no full overwrite restore was performed against a disposable copied project.
- Knowledge output templates are fixed and do not yet support custom templates, rich previews, or queued multi-output batch generation.
- The project is not packaged as a desktop app or one-click installer.

## Next Work

The current optimization plan can be considered complete through Phase 18, with the honest final score at `8.51 / 9.0`.

Recommended next phases if continuing toward true 9/9:

1. Retrieval quality phase: improve and rerun the fixed evidence set until core product questions pass reliably.
2. Fully local UI assets phase: replace the Google Fonts Material Symbols runtime request.
3. Real sample suite phase: add repeatable image, screenshot, scanned PDF, and table-document samples.
4. Restore drill phase: perform a full overwrite restore in a disposable project copy.
5. Packaging phase: create a one-click local installer or packaged desktop launch path.
