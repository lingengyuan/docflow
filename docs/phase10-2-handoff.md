# Phase 10.2 Handoff: Foreground Priority Over Background Ingest

Date: 2026-05-03

## Status

Complete.

## Completed Scope

- Added foreground-active tracking to the model-task controller.
- Background ingest now checks foreground activity before starting a new heavy ingest batch.
- Prepared files can still be parsed while Chat is active, but embedding/index-writing batches wait.
- `/api/queue` now reports pause state and foreground task state.
- Files page queue banner now shows when background indexing is paused for foreground Chat.

## Changed Files

- `src/api/model_tasks.py`
  - Tracks active foreground task count and last start/finish timestamps.
  - Exposes foreground status for `/api/queue`.
- `src/ingest/queue.py`
  - Accepts a foreground pause callback.
  - Pauses before legacy ingest work and before prepared batch processing.
  - Reports `paused`, `pause_reason`, and `paused_since`.
- `src/api/app.py`
  - Wires foreground activity into `IngestQueue`.
  - Adds foreground state to `/api/queue`.
- `frontend/index.html`
  - Shows queue pause copy in the Files page banner.
- `tests/test_queue.py`
  - Adds coverage that prepared batches pause while foreground Chat is active and resume afterward.
- `docs/phase10-optimization-plan.md`
  - Updated status after Phase 10.2 completion.

## Validation

Targeted tests:

```bash
.venv/bin/python -m pytest tests/test_model_tasks.py tests/test_queue.py tests/test_api_health.py
```

Result:

- `10 passed`

Full test suite:

```bash
.venv/bin/python -m pytest
```

Result:

- `122 passed`
- `5 warnings` from third-party SWIG bindings, unchanged from prior runs.

Live service validation after `launchctl kickstart -k gui/$(id -u)/com.docflow.local`:

```bash
.venv/bin/python main.py service status
curl -s http://127.0.0.1:8000/api/health
curl -s http://127.0.0.1:8000/api/queue
curl -s -X POST http://127.0.0.1:8000/api/query ...
```

Result:

- Service is loaded and running.
- Health is `degraded` only because optional VLM cache is missing; query and ingest capabilities are true.
- `/api/queue` returns `paused`, `pause_reason`, `paused_since`, and `foreground`.
- During an active stream request, `/api/queue` reported `foreground_active: true`.
- `/api/query` returned normally in about 5.1s with 6 citations.

Browser validation:

- Opened `http://localhost:8000`.
- Sent `请用一句话说明 DocFlow 当前状态。`
- Answer rendered with citations and send button re-enabled.
- Opened Files page and confirmed the file table renders with no console errors.
- Browser console had no errors; only the existing Tailwind CDN warning.
- Screenshot artifact: `output/playwright/phase10-2-files.png`.

Live background-pause validation with temporary Markdown files:

- Created 24 temporary files in `~/Documents/DocFlow` using prefix `docflow-phase10-2-temp-`.
- Forced those files to re-index while a foreground streaming Chat request was active.
- `/api/queue` reported:
  - `paused: true`
  - `pause_reason: foreground_active`
  - `queue_size: 24`
  - `progress.stage: paused`
  - `foreground.foreground_active: true`
- Logs showed `[queue] Paused background ingest: foreground_active`.
- After foreground Chat ended, queue stayed paused during the foreground grace window, then resumed with `progress.stage: embedding`.
- Queue returned to empty afterward.
- Deleted all `docflow-phase10-2-temp-*.md` files and restarted the service so startup cleanup removed their DB/vector records.
- Cleanup verification:
  - disk temp files: `0`
  - `/api/files` temp records: `0`
  - queue empty

## Known Limitations

- This phase pauses only before starting a new heavy ingest batch. If a batch is already running, it is allowed to finish.
- Foreground priority uses a short grace period after Chat finishes, so background ingest does not resume immediately between closely spaced user requests.
- A running embedding batch is still allowed to finish; the pause applies before the next heavy batch starts.

## Next Tasks

Continue with Phase 10.3: Safer Model Switching.

Exact next steps:

1. Add model availability and switch state to `/api/llm`.
2. Show clear loading/error state in the model selector.
3. Prevent failed or timed-out model switches from blocking later queries.
4. Add tests for unknown model, switch timeout, and preserving the current model on failure.
5. Browser-verify successful current-model behavior and visible failure behavior.
