# Phase 34 Handoff - API Layering and AppState

Date: 2026-05-10

## Completed Scope

Phase 34 completed the API layering work from the post-review roadmap while preserving existing endpoint behavior.

Completed work:

- Added `AppState` as the shared runtime container for the store, ingest pipeline, query engine, ingest queue, watcher, watched folders, model options, and model task controller.
- Replaced the bare LLM switch status dictionary with a thread-safe mapping object while keeping the same JSON shape returned by `/api/llm`.
- Moved API request and response schemas into `src/api/schemas.py`.
- Split route registration into domain files:
  - `src/api/routes/query.py`
  - `src/api/routes/library.py`
  - `src/api/routes/imports.py`
  - `src/api/routes/settings.py`
  - `src/api/routes/maintenance.py`
- Added service helpers for query formatting and citation shaping, imports and knowledge-output source building, and health/storage assembly.
- Removed direct `/api` route decorators from `src/api/app.py`; it now registers API domains through route modules before mounting the frontend.
- Kept compatibility for existing tests and callers that patch or inspect `src.api.app` runtime attributes.
- Added Phase34 structure tests to lock in the AppState, route split, service files, and thread-safe LLM switch state.

## Changed Files

- `src/api/app.py`
- `src/api/state.py`
- `src/api/schemas.py`
- `src/api/routes/__init__.py`
- `src/api/routes/query.py`
- `src/api/routes/library.py`
- `src/api/routes/imports.py`
- `src/api/routes/settings.py`
- `src/api/routes/maintenance.py`
- `src/api/services/__init__.py`
- `src/api/services/query_service.py`
- `src/api/services/import_service.py`
- `src/api/services/health_service.py`
- `tests/test_api_structure.py`
- `docs/phase34-handoff.md`

## Validation

Commands run:

```bash
python -m py_compile src/api/app.py src/api/state.py src/api/schemas.py src/api/services/*.py src/api/routes/*.py
.venv/bin/python -m pytest tests/test_api_health.py tests/test_api_llm.py tests/test_api_storage.py tests/test_api_files.py tests/test_import_workflows.py tests/test_api_debug.py tests/test_conversations.py tests/test_query_scope.py tests/test_static_assets.py
.venv/bin/python -m pytest tests/test_api_structure.py
.venv/bin/python -m pytest
.venv/bin/python main.py browser-acceptance --base-url http://127.0.0.1:8010 --screenshots-dir output/playwright/phase34-browser-acceptance --with-mutation-flow --json
find /Users/hughlin/Documents/DocFlow -maxdepth 1 -iname '*phase32-acceptance*' -o -iname '*phase34-acceptance*'
sqlite3 docflow.db "select count(*) from files where file_name like '%phase32-acceptance%' or file_name like '%phase34-acceptance%'; select count(*) from history where question like '%验收标记%';"
git diff --check
```

Results:

- Python compile check passed.
- Targeted API and frontend regression tests passed: 66 passed, 5 warnings.
- Phase34 structure tests passed: 3 passed, 5 warnings.
- Full test suite passed: 221 passed, 5 warnings.
- Browser acceptance passed with mutation flow: 74 passed, 0 failed.
- Browser screenshots were written to `output/playwright/phase34-browser-acceptance/`.
- Screenshot review checked Chat, Library, and Settings. No visible layout regression was found.
- The browser mutation flow created a temporary Markdown note, waited for ingestion, queried it, then cleaned up the file, database record, vector, history row, and conversation.
- Follow-up cleanup checks found no remaining Phase32 or Phase34 acceptance files or records.
- `git diff --check` passed.

## Known Limitations

- `src/api/app.py` is smaller and no longer owns route registration directly, but it still contains several compatibility wrappers and health helper functions. This keeps Phase34 behavior stable and avoids forcing a large endpoint migration in one step.
- The AppState compatibility layer is intentionally preserved because existing tests and local scripts still patch `src.api.app` runtime attributes directly.
- Phase34 does not change endpoint return formats, UI text, or user-facing behavior.
- The browser acceptance mutation note name still uses the existing `phase32-acceptance` prefix from the shared acceptance runner. Cleanup confirms it does not leave files or records behind.

## Next Phase

Proceed to Phase35 after this Phase34 checkpoint is committed and pushed.

Recommended Phase35 focus:

1. Add explicit status types for files instead of passing raw status strings through the codebase.
2. Type core records that cross storage, retrieval, and API boundaries.
3. Fix the FTS5 row-id mapping risk with a targeted regression test.
4. Review HTTP client usage and timeout handling.
5. Review SQLite, Qdrant, and runtime lifecycle closing behavior.
