# Phase 14 Handoff: Scoped Questions and Answer Reliability

Date: 2026-05-05

## Status

Complete.

## Scope

Phase 14 improved how users control the answer scope and how DocFlow behaves when evidence is weak.

Implemented:

- Added Chat scope modes for all library, selected collection, selected file, and full-text mode.
- Added backend scope resolution for collections and file IDs before retrieval.
- Added full-text retrieval mode that skips vector search and uses SQLite FTS results.
- Added insufficient-evidence handling before answer generation so weak matches return a clear refusal.
- Added scope support to `/api/query`, `/api/query/stream`, and `/api/debug/retrieve`.
- Added Chat UI controls for scope mode, collection selection, file selection, and scope option refresh.
- Expanded fixed retrieval evidence cases for current scope controls and insufficient-evidence behavior.
- Updated README and maturity scores.

## Changed Files

- `src/api/app.py`
- `src/query/engine.py`
- `src/query/retriever.py`
- `frontend/index.html`
- `tests/test_query_scope.py`
- `tests/test_engine.py`
- `tests/test_retriever.py`
- `tests/test_conversations.py`
- `tests/test_api_debug.py`
- `tests/test_static_assets.py`
- `README.md`
- `eval/phase11_maturity_dimensions.json`
- `eval/phase11_questions.jsonl`
- `docs/phase14-handoff.md`

## Validation Results

Targeted tests:

```bash
.venv/bin/python -m pytest tests/test_query_scope.py tests/test_engine.py tests/test_retriever.py tests/test_conversations.py tests/test_api_debug.py tests/test_static_assets.py tests/test_maturity_eval.py
# 52 passed, 5 warnings
```

Maturity score:

```bash
.venv/bin/python main.py maturity-eval --skip-retrieval
# Overall score: 7.52/9.0
# 文档问答主流程: 8.5/9.0
# 搜索和答案可靠性: 8.0/9.0
# 界面可用性: 7.0/9.0
```

Browser validation:

```bash
.venv/bin/python <one-off Playwright Chat scope check>
# phase14 browser scope check passed
# screenshots:
# /tmp/docflow-phase14-collection-scope.png
# /tmp/docflow-phase14-file-scope.png
# /tmp/docflow-phase14-mobile-scope.png
```

Covered in the browser flow:

- Opened `http://127.0.0.1:8000`.
- Verified the Chat scope controls are visible.
- Switched to selected-collection mode and verified the collection selector appears.
- Switched to selected-file mode and verified the file selector appears.
- Switched to full-text mode and verified collection/file selectors hide.
- Checked the scope control at a 390px-wide viewport.

Live API checks:

```bash
curl -sS -X POST http://127.0.0.1:8000/api/query ...
# collection scope without a collection was rejected with:
# Collection is required for collection scope

curl -sS -X POST http://127.0.0.1:8000/api/debug/retrieve ...
# {'retrieval_mode': 'full_text', 'vector': 0, 'fts': 20, 'status': 'ok'}
```

Full validation:

```bash
.venv/bin/python -m pytest
# 160 passed, 5 warnings

.venv/bin/python main.py check --json
# status=ok, sqlite_chunks=9831, qdrant_points=9831

git diff --check
# clean
```

## Known Limitations

- Collection and file scopes resolve to file names because the existing retriever filter is file-name based. Duplicate file names in different folders can still collide.
- Full-text mode currently uses SQLite FTS plus existing rerank behavior; it is not a separate Boolean query builder.
- Insufficient-evidence detection is conservative: it blocks clearly weak reranker scores but does not yet produce a detailed confidence explanation.
- Browser validation used Playwright because the Browser Use Node REPL tool was not available in this session.

## Next Phase

Continue with Phase 15: improve product-level UI maturity.

Recommended Phase 15 tasks:

1. Reorganize the app shell around Chat, Library, Notes, and Settings.
2. Add clearer empty states, error states, and first-run guidance.
3. Move model, health, paths, and maintenance actions toward a Settings view.
4. Reduce UI clutter in the Chat header and make scope/status controls feel coherent.
5. Run full tests, browser validation, maturity score update, and write `docs/phase15-handoff.md`.
