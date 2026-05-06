# Phase 17 Handoff: Knowledge Output Workflows

Date: 2026-05-06

## Status

Complete.

## Scope

Phase 17 improved local knowledge workflow maturity by adding reusable knowledge output generation.

Implemented:

- Added four knowledge output types: structured summaries, learning cards, action items, and project briefs.
- Added `/api/knowledge-output` to generate a Markdown output from pasted source text, selected Library files, or both.
- Saved generated outputs into the first watched folder, queued them for ingest, and tagged them with `knowledge-output` plus the output type.
- Added the `Knowledge Outputs` collection as the default destination for generated outputs.
- Added a Notes workspace panel for generating knowledge outputs.
- Added a Library batch action that carries selected files into the Notes knowledge output panel.
- Updated README, maturity scoring, fixed evidence questions, and tests.

## Changed Files

- `src/knowledge_outputs.py`
- `src/ingest/imports.py`
- `src/query/generator.py`
- `src/query/engine.py`
- `src/api/app.py`
- `frontend/index.html`
- `tests/test_import_workflows.py`
- `tests/test_generator.py`
- `tests/test_static_assets.py`
- `README.md`
- `eval/phase11_maturity_dimensions.json`
- `eval/phase11_questions.jsonl`
- `docs/phase17-handoff.md`

## Validation Results

Targeted tests:

```bash
.venv/bin/python -m pytest tests/test_import_workflows.py tests/test_generator.py tests/test_static_assets.py tests/test_maturity_eval.py
# 36 passed, 5 warnings
```

Full tests:

```bash
.venv/bin/python -m pytest
# 170 passed, 5 warnings
```

Live service health:

```bash
launchctl kickstart -k gui/$(id -u)/com.docflow.local
curl -fsS http://127.0.0.1:8000/api/health
# status=ok
# SQLite ok, Qdrant ok, Ollama ok, local model cache ok
```

Consistency check:

```bash
.venv/bin/python main.py check --json
# status=ok, sqlite_chunks=9850, qdrant_points=9850
# no missing points, orphan points, chunk mismatches, or missing source files
```

Maturity score:

```bash
.venv/bin/python main.py maturity-eval --skip-retrieval
# Overall score: 8.12/9.0
# local_workflow: 8.3/9.0
# testing_discipline: 8.4/9.0
# product_maturity: 7.6/9.0
```

Diff hygiene:

```bash
git diff --check
# clean
```

Browser validation:

```bash
.venv/bin/python <one-off Playwright Phase 17 Notes/Library check>
# phase17 browser validation passed
# desktop=/tmp/docflow-phase17-notes-desktop.png
# mobile=/tmp/docflow-phase17-knowledge-mobile.png
```

Covered in the browser flow:

- Restarted and opened the local app at `http://127.0.0.1:8000`.
- Opened Notes and verified the knowledge output panel, four output types, default collection copy, and submit action.
- Opened Library, selected one done file, clicked `知识产物`, and verified Notes received the selected file context.
- Submitted the knowledge output form with a mocked API response to verify browser payload and success state without writing a temporary live knowledge file.
- Verified desktop and 390px mobile layouts, including the fixed short mobile collection placeholder and no horizontal overflow.
- Checked browser console/page errors; none were reported.

Browser Use note:

- The Browser Use Node REPL execution tool was not exposed in this session after discovery attempts.
- Computer Use could not inspect the Codex app window because `com.openai.codex` is blocked for safety.
- Playwright was used as the fallback browser validation path.

## Known Limitations

- Template choices are fixed. There is no user-editable template editor yet.
- The browser validation intercepted `/api/knowledge-output` to avoid creating and then cleaning a live test Markdown file; backend saving and selected-file source assembly are covered by automated API tests.
- Generated output preview is returned by the API but not yet displayed as a rich preview before saving.
- Batch generation over many files still runs as one foreground request; there is no multi-output queue or progress view yet.

## Next Phase

Continue with Phase 18: final 9-point maturity hardening and release polish.

Recommended Phase 18 tasks:

1. Produce the final end-to-end acceptance report with screenshots, fixed retrieval checks, and current maturity score.
2. Improve final product packaging: remove Tailwind CDN, add production static build guidance, and include screenshots in README.
3. Add ordinary-user setup and troubleshooting docs for first run, model cache preparation, backup, and restore.
4. Add real restore rehearsal evidence and document the result.
5. Expand real sample validation for images, OCR, tables, and screenshot-like documents.
6. Decide whether knowledge output templates need preview, custom templates, or multi-file batch output before scoring local workflow at 9.
