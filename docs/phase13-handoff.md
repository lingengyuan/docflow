# Phase 13 Handoff: Local Capture Workflows

Date: 2026-05-04

## Status

Complete.

## Scope

Phase 13 expanded how new knowledge enters DocFlow.

Implemented:

- Added webpage import from a URL into a local Markdown file.
- Added quick Markdown note creation from the Library view.
- Added "save answer as note" action beside generated answers.
- Added local Markdown import helpers for webpage parsing, quick notes, and answer notes.
- Added API endpoints for URL import, note creation, and answer-note creation.
- Improved the Library header on narrow screens so action buttons do not crush the title.
- Updated README and maturity scores after verification.

## Changed Files

- `src/ingest/imports.py`
- `src/api/app.py`
- `frontend/index.html`
- `tests/test_import_workflows.py`
- `tests/test_static_assets.py`
- `README.md`
- `eval/phase11_maturity_dimensions.json`
- `docs/phase13-handoff.md`

## Validation Results

Targeted tests:

```bash
.venv/bin/python -m pytest tests/test_import_workflows.py tests/test_static_assets.py tests/test_maturity_eval.py
# 17 passed, 5 warnings
```

Maturity score:

```bash
.venv/bin/python main.py maturity-eval --skip-retrieval
# Overall score: 7.33/9.0
# 文档问答主流程: 8.0/9.0
# 文档格式支持: 8.0/9.0
# 扩展生态: 7.0/9.0
```

Browser validation:

```bash
.venv/bin/python <one-off Playwright Library workflow check>
# ok
# screenshots:
# /tmp/docflow-phase13-note-panel.png
# /tmp/docflow-phase13-url-panel.png
# /tmp/docflow-phase13-mobile-library-fixed.png
```

Covered in the browser flow:

- Opened the Library view on `http://127.0.0.1:8000`.
- Opened the quick-note panel and filled title, note body, collection, and tags.
- Opened the webpage-import panel and verified URL mode hides the note-body field.
- Checked a narrow mobile-size viewport after the responsive header fix.
- Did not submit temporary browser notes during this visual flow; API submit behavior is covered by tests with temporary directories.

Full validation:

```bash
.venv/bin/python -m pytest
# 151 passed, 5 warnings

.venv/bin/python main.py check --json
# status=ok, sqlite_chunks=9831, qdrant_points=9831

curl -s -X POST http://127.0.0.1:8000/api/import/url ...
# rejected non-HTTP URL with: Only http and https URLs can be imported
```

## Known Limitations

- Webpage import uses a simple built-in HTML reader. It is enough for readable articles and docs pages, but it is not a full browser-rendered clipping engine.
- URL import fetches public HTTP/HTTPS pages from the backend. It does not yet support authenticated pages or JavaScript-only content.
- Saved answer notes currently save the answer text and optional question/source rows; they do not preserve full chat thread context.
- VLM/OCR sample checks for images, screenshots, and table-like images are still listed for the final maturity pass.

## Next Phase

Continue with Phase 14: improve scoped questioning and answer reliability.

Recommended Phase 14 tasks:

1. Add query scope modes for all library, selected collection, selected file, and full-text mode.
2. Make insufficient-evidence answers refuse clearly instead of sounding confident.
3. Expand the fixed retrieval evaluation set with real questions from the new Library and capture workflows.
4. Add UI controls for scope selection in Chat.
5. Run full tests, browser validation, maturity score update, and write `docs/phase14-handoff.md`.
