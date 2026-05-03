# Phase 10 Validation Issues: Daily-Use Stability

Date: 2026-05-03

## Status

This document records the full-product validation run after Phase 9 follow-up. The product is usable for local daily document Q&A, but the validation exposed stability and UX issues that should be handled before treating the project as a polished daily driver.

## Validation Scope

- Background launchd service on `http://localhost:8000`.
- Full automated test suite.
- Health, queue, files, history, favorites, sources, preview, chunks, LLM, query, and summarize APIs.
- Browser flows:
  - Chat view.
  - Health panel.
  - LLM selector.
  - Conversation menu.
  - Files view.
  - Upload.
  - Favorites.
  - File selection and summarize action.
  - History view.
  - History replay.
  - Citation/source preview.
  - Copy and Markdown export.

## Current Project State

- Service is installed and running through launchd.
- Full test suite passed: 116 tests passed, 5 warnings.
- Health endpoint returns `degraded`, not `unavailable`.
- Core checks are ok:
  - API ok.
  - SQLite ok.
  - Qdrant ok.
  - Ollama ok.
- Query and ingest capabilities are true.
- Queue is currently empty.
- File library has 247 records, all `done`.
- `mlx-community/Qwen3-4B-4bit` and `mlx-community/Qwen3-8B-4bit` are cached.
- Optional VLM cache is missing, so health remains `degraded`.

## What Worked

- App shell opened in the browser with title `DocFlow`.
- Health panel rendered dependency rows correctly.
- Files page rendered file table, statuses, queue banner, and actions.
- Upload flow worked with a temporary text file; it was indexed as `done` with 1 chunk, then the test file was removed.
- Favorite toggle worked and was restored to the original state.
- File selection showed the summarize action bar.
- History page rendered past queries.
- Clicking a history item replayed the answer and citations into Chat.
- Clicking a citation opened the source preview in a new tab.
- Copy answer worked.
- Markdown export worked and produced a downloaded Markdown file.
- After service restart, browser Chat produced a cited answer in about 3.2 seconds.
- After service restart, synchronous `/api/query` produced a cited answer in about 4.2 seconds.

## Issues

### P0: A stuck ML task can block all later model-backed work

Observed behavior:

- During browser Chat validation, a streaming query submitted successfully but did not produce answer text.
- The send button stayed disabled while waiting.
- After that, `/api/query`, `/api/summarize`, and `/api/debug/retrieve` timed out at 20-45 seconds.
- A launchd service restart restored normal query behavior.

Evidence:

- `/api/query/stream` returned HTTP 200, but the browser did not receive answer content.
- After the stall, synchronous model-backed endpoints timed out.
- After `launchctl kickstart -k gui/$(id -u)/com.docflow.local`, `/api/query` returned in about 4.2 seconds.

Likely cause:

- The app uses one shared `ml_executor` worker for retrieval reranking, LLM generation, summarize, debug retrieval, and model switching.
- If one streaming task stalls inside the worker, later model-backed work queues behind it.

Impact:

- A single failed or long-running user request can make the app feel broken until service restart.

Priority:

- Highest. This is the main blocker for stable daily use.

### P1: Background ingest competes with foreground question answering

Observed behavior:

- During validation, startup/background scanning processed many files.
- Some embedding batches took 60-132 seconds.
- While this was happening, foreground query, summarize, and debug retrieval became slow or timed out.

Evidence:

- Queue showed more than 100 pending files early in validation.
- Logs showed examples such as:
  - `README.md` ingest total about 60 seconds.
  - `pixel-game-plan.md` ingest total about 132 seconds.
  - Other large batches occupied embedding for multiple seconds.

Impact:

- Login-time or scan-time ingest can make Chat feel unreliable even though the service is technically running.

Priority:

- High. This affects the first minutes after startup and any large watched-folder change.

### P1: Model switching lacks guardrails and feedback

Observed behavior:

- The LLM selector listed both 4B and 8B models.
- Clicking the 8B option did not clearly show progress or failure.
- The app appeared unchanged from the user's perspective.

Evidence:

- `/api/llm` still reported the 4B model after the browser click.
- Health later showed the 8B cache was present, suggesting the switch path can trigger model work without clear UI feedback.

Impact:

- Users can accidentally start expensive model work and receive no clear explanation.

Priority:

- High. This overlaps with the shared-worker blocking issue.

### P1: Destructive history action has no confirmation

Observed behavior:

- The History page exposes `清空历史`.
- The validation did not click it because it would delete real user history.

Impact:

- A mistaken click can remove real history with no second chance.

Priority:

- High. It is a direct data-loss risk.

### P2: Health status is technically correct but still confusing

Observed behavior:

- Health is `degraded` because the optional VLM cache is missing.
- Core daily-use features still work.

Impact:

- Users may read `degraded` as "the app is broken" even when query, ingest, OCR, SQLite, Qdrant, and Ollama are available.

Priority:

- Medium. It is not a functional blocker, but it affects trust.

### P2: HEAD requests for file previews return 404

Observed behavior:

- `GET /api/file/{id}/preview` worked.
- `HEAD /api/file/{id}/preview` returned 404 for tested files.

Impact:

- Browser preview still works, but link checkers or clients that probe with `HEAD` get misleading failures.

Priority:

- Medium-low.

### P3: Browser still requests `/favicon.ico`

Observed behavior:

- The frontend links `/favicon.svg`.
- Logs still showed a browser `GET /favicon.ico` 404 during source preview.

Impact:

- Harmless, but it adds noise to logs.

Priority:

- Low.

## Validation Commands and Results

```bash
.venv/bin/python -m pytest
# 116 passed, 5 warnings

.venv/bin/python main.py service status
# loaded, running=true

curl -s http://127.0.0.1:8000/api/health
# status=degraded, api/sqlite/qdrant/ollama ok, query=true, ingest=true

curl -s http://127.0.0.1:8000/api/files
# 247 files, all done

curl -s http://127.0.0.1:8000/api/queue
# queue_size=0

curl -s -X POST http://127.0.0.1:8000/api/query \
  -H 'Content-Type: application/json' \
  -d '{"question":"DocFlow 当前状态如何？请简短回答。"}'
# after restart: HTTP 200, about 4.2s, cited answer returned
```

## Recommended Direction

Fix the stability and foreground-priority path before adding new user-visible features. The next plan should focus on:

1. Making model-backed tasks cancellable and bounded.
2. Separating foreground query work from background ingest work.
3. Adding user-visible state for long model operations.
4. Adding confirmation for destructive actions.
5. Making health labels easier to understand.

