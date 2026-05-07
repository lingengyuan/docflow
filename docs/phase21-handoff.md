# Phase 21 Handoff: Generated Real Sample Suite

Date: 2026-05-07

## Status

Complete.

## Prior Commit

Phase 20 was committed before Phase 21 work started:

```bash
git log -1 --oneline
# 2f8194e feat(ui): complete phase20 local assets
```

## Scope

Phase 21 added a repeatable generated sample suite so multi-format behavior can
be checked before handoff or release without relying on private user files.

Implemented:

- Added `python main.py sample-suite`.
- Added `scripts/run_sample_suite.py`.
- Added `src/quality/sample_suite.py` to generate temporary local samples and
  validate them.
- Generated sample files include:
  - Markdown with a table.
  - Screenshot-like PNG.
  - Image-only scanned PDF.
- The suite validates:
  - Scanned PDF detection and OCR path with deterministic local OCR output.
  - VLM image parsing path with deterministic local VLM output.
  - Markdown table and table summary chunking.
  - Source preview `HEAD` and `GET` behavior.
  - Knowledge output generation from a selected file and queue submission.
- Added automated tests for sample generation and the full sample suite.
- Added browser validation using the generated sample file in Library and Notes.
- Updated README, local deployment release checklist, changelog, and maturity
  scoring.

## Changed Files

- `CHANGELOG.md`
- `README.md`
- `docs/LOCAL_DEPLOYMENT.md`
- `docs/phase21-handoff.md`
- `eval/phase11_maturity_dimensions.json`
- `main.py`
- `scripts/run_sample_suite.py`
- `src/quality/sample_suite.py`
- `tests/test_sample_suite.py`

## Validation Results

Phase 20 commit:

```bash
git log -1 --oneline
# 2f8194e feat(ui): complete phase20 local assets
```

Sample suite:

```bash
.venv/bin/python main.py sample-suite --json
# passed=5
# failed=0
# checks=scanned_pdf_ocr, vlm_image_parse, table_chunking, source_preview_api, knowledge_output_api
```

Generated sample output:

```bash
/tmp/docflow-phase21-samples/
# phase21-table.md
# phase21-screenshot.png
# phase21-scanned.pdf
# knowledge-*-phase21-sample-summary.md
```

Targeted sample tests:

```bash
.venv/bin/python -m pytest tests/test_sample_suite.py
# 2 passed, 5 warnings
```

Full tests:

```bash
.venv/bin/python -m pytest
# 183 passed, 5 warnings
```

Browser validation:

```bash
.venv/bin/python <one-off Playwright Phase 21 browser check>
# Library displayed phase21-table.md
# preview action was visible
# selecting the sample showed "已选 1 个文件"
# Notes knowledge output panel received "Library 中选中的 1 个文件"
# icons rendered as local SVG: 94 / 94
# console_errors=[]
```

Browser screenshot:

- `/tmp/docflow-phase21-browser.png`

Maturity score:

```bash
.venv/bin/python main.py maturity-eval --skip-retrieval --json
# overall_score=8.62/9.0
```

Live project consistency:

```bash
.venv/bin/python main.py check --json
# status=ok
# sqlite_chunks=10091
# qdrant_points=10091
# missing_points=0
# orphan_points=0
```

Diff hygiene:

```bash
git diff --check
# passed
```

## Known Limitations

- OCR and VLM sample checks use deterministic local fakes to verify the DocFlow
  code path without pulling or running large models. They do not measure live
  model quality.
- The generated sample suite does not yet cover DOCX, webpage import, or a real
  photo/screenshot corpus.
- Browser validation used a routed sample file in the running page instead of
  uploading a temporary file into the live personal library, so it did not leave
  user data behind.
- The suite writes generated files under `/tmp/docflow-phase21-samples` by
  default and refreshes that directory on each run.

## Next Phase Tasks

Phase 22 should be the restore drill phase:

1. Create a disposable copy of the project data paths or a temporary config.
2. Run a real backup from the current live project.
3. Restore into the disposable target without touching the live database or
   Qdrant collection.
4. Verify restored SQLite metadata, chunk counts, exported chunks, and source
   paths.
5. Run `.venv/bin/python -m pytest`, `.venv/bin/python main.py check --json`,
   and a focused restore validation command.
6. Write `docs/phase22-handoff.md` with scope, changed files, validation
   results, limitations, and exact next tasks before reporting completion.
