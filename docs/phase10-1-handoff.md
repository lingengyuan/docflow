# Phase 10.1 Handoff: Stop Permanent Hangs

Date: 2026-05-03

## Status

Complete.

## Completed Scope

- Added a foreground model-task controller for API-facing model work.
- Added bounded timeout handling for:
  - `/api/query`
  - `/api/query/stream`
  - `/api/summarize`
  - `/api/debug/retrieve`
  - `/api/llm` model switching
  - startup warmup
- Streaming Chat now emits a visible `error` event when model work does not produce citations/tokens in time.
- Browser disconnects and stream timeouts now mark the active stream cancelled and retire the blocked worker so later requests can run.
- Timed-out non-stream endpoints now return HTTP 504 with a clear message.
- Added logging for task start, finish, timeout, cancellation, and worker retirement.

## Changed Files

- `src/api/model_tasks.py`
  - New model-task controller with timeout and worker-retirement behavior.
- `src/api/app.py`
  - Replaced direct shared executor use with the controller.
  - Added stream timeout/cancellation handling.
  - Added 504 timeout responses for foreground endpoints.
- `tests/test_model_tasks.py`
  - Verifies timeout retirement and cancellation recovery.
- `tests/test_conversations.py`
  - Verifies `/api/query` timeout recovery without restart.
  - Verifies stream timeout emits error and does not write an assistant message.
- `tests/test_api_debug.py`
  - Verifies `/api/debug/retrieve` timeout behavior.
- `docs/phase10-optimization-plan.md`
  - Updated status after Phase 10.1 completion.

## Validation

Targeted tests:

```bash
.venv/bin/python -m pytest tests/test_model_tasks.py tests/test_conversations.py tests/test_api_debug.py
```

Result:

- `11 passed`

Full test suite:

```bash
.venv/bin/python -m pytest
```

Result:

- `121 passed`
- `5 warnings` from third-party SWIG bindings, unchanged from prior runs.

Live service validation after `launchctl kickstart -k gui/$(id -u)/com.docflow.local`:

```bash
.venv/bin/python main.py service status
curl -s http://127.0.0.1:8000/api/health
curl -s http://127.0.0.1:8000/api/queue
curl -s -X POST http://127.0.0.1:8000/api/debug/retrieve ...
curl -s -X POST http://127.0.0.1:8000/api/query ...
curl -sN -X POST http://127.0.0.1:8000/api/query/stream ...
```

Result:

- Service is loaded and running.
- Health is `degraded` only because optional VLM cache is missing; query and ingest capabilities are true.
- Queue is empty.
- `/api/debug/retrieve` returned normally in about 4.7s.
- `/api/query` returned normally in about 4.0s with 6 citations.
- `/api/query/stream` returned `conversation`, `citations`, `token`, and `done` in about 3.2s with no error event.

Browser validation:

- Opened `http://localhost:8000`.
- Sent `请用一句话说明 DocFlow 当前状态。`
- Answer rendered with citations.
- Send button re-enabled after completion.
- Browser console had no errors; only the existing Tailwind CDN warning.
- Screenshot artifact: `output/playwright/phase10-1-chat.png`.

## Known Limitations

- Python threads cannot be forcibly killed safely. When an underlying MLX call is truly stuck, the controller retires the blocked worker and creates a fresh worker for later requests; the old call may still finish in the background.
- Background indexing competition is not solved in Phase 10.1. That is Phase 10.2.
- UI-level timeout wording is handled through the existing Chat error display; there is not yet a dedicated retry button or busy-state panel.

## Next Tasks

Continue with Phase 10.2: Foreground Priority Over Background Ingest.

Exact next steps:

1. Add a foreground-active state to the app.
2. Make background ingest check foreground-active state before starting new heavy batches.
3. Add queue paused/backoff state to `/api/queue`.
4. Show queue pause/resume truthfully in the Files page.
5. Add unit and browser tests for background pause/resume while Chat is active.
