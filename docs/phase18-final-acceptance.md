# Phase 18 Final Acceptance Report

Date: 2026-05-06

## Summary

Phase 18 completed the final hardening and release-polish pass for the current DocFlow optimization plan.

Current maturity score:

- Overall: `8.51 / 9.0`
- At target: `0 / 12`
- Near target: `12 / 12`
- Largest remaining gaps: Library management, format support, answer reliability, model runtime, and local workflow.

This is not a full 9/9 release yet. The project is now much closer to a mature local personal knowledge assistant, but several items still need more work before every dimension can honestly be scored at 9.

## Completed Acceptance Scope

- Removed the runtime Tailwind CDN dependency and switched the browser UI to committed local CSS at `frontend/styles.css`.
- Added a repeatable CSS build command through `package.json` and `tailwind.config.js`.
- Added release-facing project files: `LICENSE`, `CHANGELOG.md`, and `docs/LOCAL_DEPLOYMENT.md`.
- Added README release notes, setup guidance, screenshots, and local deployment references.
- Fixed configured code-file support so `.py`, `.rs`, `.ts`, `.css`, and `.sh` files are parsed as text instead of being listed but unsupported.
- Synchronized upload accept lists with Markdown, code text, and image extensions.
- Ran a non-destructive backup and restore rehearsal against a temporary directory.
- Ran browser validation across Chat, Library, Notes, Settings, and mobile Settings.
- Ran a light real-sample validation for Markdown tables and scanned PDF OCR.

## Screenshots

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
# note: Browserslist printed an outdated caniuse-lite warning, but CSS generation completed.
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
# OCR available, VLM health check reported available
```

Browser validation:

```bash
.venv/bin/python <one-off Playwright Phase 18 acceptance check>
# passed
# no console warnings or errors
# confirmed /styles.css is loaded and Tailwind CDN is absent
# confirmed Chat, Library, Notes, and Settings switch correctly
# confirmed Library refresh button works without page errors
# confirmed upload accept lists include code and image extensions
# confirmed 390px mobile Settings has no horizontal overflow
```

Sample validation:

```bash
.venv/bin/python <one-off Phase 18 table/OCR sample check>
# table_chunk_types=["table", "table_summary"]
# scanned PDF OCR: is_scanned=true, page_is_ocr=true
# OCR preview: PHASE18 OCR SAMPLE / Invoice total: 1280 USD
```

Backup and restore rehearsal:

```bash
.venv/bin/python main.py backup --output /tmp/docflow-phase18-backups --keep 2
# status=done
# archive=/tmp/docflow-phase18-backups/docflow-backup-20260506-125016-284259.tar.gz
# files=250
# chunks=9850
```

```bash
.venv/bin/python main.py restore-plan /tmp/docflow-phase18-backups/docflow-backup-20260506-125016-284259.tar.gz
# status=ok
# members: chunks.jsonl, config.yaml, docflow.db, manifest.json
# missing=[]
```

Temporary extraction check:

```json
{
  "quick_check": "ok",
  "manifest_files": 250,
  "manifest_chunks": 9850,
  "sqlite_files": 250,
  "sqlite_chunks": 9850,
  "jsonl_chunks": 9850
}
```

Maturity and retrieval evaluation:

```bash
.venv/bin/python main.py maturity-eval --no-rerank --json
# overall_score=8.51/9.0
# retrieval_eval=11/27 passed
```

Final consistency check after the code-file parser fix allowed background scan to ingest additional code snippets:

```bash
.venv/bin/python main.py check --json
# status=ok
# sqlite_chunks=10026
# qdrant_points=10026
# no missing points, orphan points, file chunk mismatches, or missing source files
```

The backup rehearsal archive was created before the final background code-file scan completed, so its `9850` chunk count is expected to differ from the final live consistency count of `10026`.

## Not Yet 9/9

- The browser still requests Material Symbols from Google Fonts. Tailwind CDN is gone, but one runtime external font request remains.
- Fixed retrieval evaluation is still weak: `11 / 27` passed without reranking. This should become a dedicated retrieval-quality phase before scoring answer reliability at 9.
- Live VLM image parsing was attempted with a temporary image, but it began model-file fetching and was stopped. Health reports the VLM cache as available, but live image understanding is not counted as fully verified in this report.
- The restore rehearsal was non-destructive. A full overwrite restore should still be rehearsed in a disposable copy.
- Knowledge output templates are fixed. There is no custom template editor, rich preview-before-save, or multi-output batch queue yet.
- There is no packaged desktop app or one-click installer.

## Acceptance Decision

Phase 18 is accepted as the final hardening and release-polish phase for the current optimization plan.

The current project state is usable and substantially more mature than the Phase 11 baseline, but it should be described as an `8.51 / 9.0` local personal knowledge assistant, not as a fully completed 9/9 mature product.
