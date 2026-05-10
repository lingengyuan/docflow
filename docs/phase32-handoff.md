# Phase 32 Handoff - Low-Cost High-Impact Fixes

Date: 2026-05-10

## Completed Scope

Phase 32 completed the low-cost user-experience and answer-quality fixes from the post-review roadmap.

Completed work:

- Fixed English follow-up detection so independent questions such as `what is bitcoin` are no longer merged with the previous question.
- Kept whole-word English follow-up support for natural follow-ups such as `what about it`.
- Improved answer rendering for common Markdown: headings, unordered lists, ordered lists, links, fenced code blocks, blockquotes, inline code, bold text, and existing tables.
- Replaced raw chat failures with short user-facing messages, including timeout, service-not-ready, disconnected-service, stale-conversation, and unavailable-model cases.
- Hardened streamed error handling so malformed error payloads still render a normal user-facing message.
- Reduced noisy Library refreshes during queue polling by refreshing the file table only when queue state actually changes.
- Added an optional browser acceptance mutation flow that creates a temporary Markdown note, waits for it to be indexed, asks a file-scoped question, and cleans up the temporary file, database record, vector, history row, and conversation.
- Fixed the root cause found by the mutation flow: files already marked `pending` were incorrectly treated as unchanged and skipped by ingestion.

## Changed Files

- `frontend/index.html`
- `frontend/styles.css`
- `scripts/run_browser_acceptance.py`
- `src/api/app.py`
- `src/ingest/store.py`
- `src/quality/browser_acceptance.py`
- `tests/test_browser_acceptance.py`
- `tests/test_conversations.py`
- `tests/test_frontend_markdown.py`
- `tests/test_static_assets.py`
- `tests/test_store.py`
- `docs/phase32-handoff.md`

## Validation

Commands run:

```bash
npm run build:css
.venv/bin/python -m pytest tests/test_conversations.py tests/test_frontend_markdown.py tests/test_static_assets.py tests/test_browser_acceptance.py
.venv/bin/python -m pytest tests/test_store.py tests/test_import_workflows.py tests/test_browser_acceptance.py tests/test_conversations.py tests/test_frontend_markdown.py tests/test_static_assets.py
.venv/bin/python -m pytest tests/test_browser_acceptance.py tests/test_store.py tests/test_static_assets.py tests/test_frontend_markdown.py tests/test_conversations.py
.venv/bin/python -m pytest
.venv/bin/python main.py browser-acceptance --base-url http://127.0.0.1:8010 --screenshots-dir output/playwright/phase32-browser-acceptance --with-mutation-flow --timeout-ms 90000 --json
find ~/Documents/DocFlow -maxdepth 1 -iname '*phase32-acceptance*' -print
sqlite3 docflow.db "select count(*) from files where file_name like '%phase32-acceptance%'; select count(*) from history where question='这条临时笔记里的验收标记是什么？请只回答标记。'; select count(*) from messages where content='这条临时笔记里的验收标记是什么？请只回答标记。'; select count(*) from conversations where title='这条临时笔记里的验收标记是什么？请只回答标记。';"
```

Results:

- CSS build passed.
- Targeted tests passed:
  - 36 passed, 5 warnings before the pending-ingest fix.
  - 66 passed, 5 warnings after the pending-ingest fix.
  - 57 passed, 5 warnings after mutation-flow cleanup was extended.
- Full test suite passed after all changes: 217 passed, 5 warnings.
- Final targeted regression pass after the streamed-error hardening also passed: 57 passed, 5 warnings.
- First browser mutation run found a real ingestion bug: the temporary note remained `pending` because pending records were skipped as unchanged.
- After the fix, browser acceptance passed: 74 passed, 0 failed.
- Mutation-flow cleanup verified no temporary Phase32 files, file records, history rows, messages, or conversations remained.
- Browser screenshots were written to `output/playwright/phase32-browser-acceptance/`.
- Manual screenshot review checked Chat and Notes layouts; no visible layout break was found.

## Known Limitations

- The mutation-flow query proves the temporary note can be created, indexed, selected, queried, and cleaned. The local model response is intentionally not asserted against exact wording because answer text can vary by model.
- Browser acceptance uses the user's current local data, so screenshot content and file counts will change as the knowledge base changes.
- Queue refresh optimization is intentionally conservative. It still refreshes on meaningful progress changes so users see state transitions during active ingestion.

## Next Phase

Proceed to Phase 33 after this Phase32 checkpoint is committed and pushed.

Recommended Phase33 focus:

1. Continue the current roadmap before starting deeper review-driven architecture changes.
2. Keep the mutation-flow acceptance enabled for phases that touch notes, ingestion, query scope, or cleanup.
3. If answer-quality work resumes, add a separate quality check for whether retrieved temporary-note content is reflected in the final model answer.
