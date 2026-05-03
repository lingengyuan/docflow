# Phase 10 Optimization Plan: Daily-Use Stability

Date: 2026-05-03

## Status

Approved and implemented. Phase 10.1 through Phase 10.5 are complete as of 2026-05-03.

## Goal

Make DocFlow reliable during normal daily use:

- One stuck query must not block later queries forever.
- Background indexing must not make foreground Chat feel broken.
- Long-running model work must show a clear state to the user.
- Destructive UI actions must require confirmation.
- Health status must distinguish "core usable" from "optional capability missing".

## Design Options

### Option A: Minimal Reliability Patch

Summary:

- Keep the current architecture, but add request timeouts, cancellation handling, and UI feedback around the existing shared model worker.

Effort:

- Small to medium.

Risk:

- Low implementation risk, but it does not fully solve foreground/background competition.

Builds on:

- Existing `ml_executor`, `query_stream`, `AnswerGenerator`, and frontend disabled-button states.

Best for:

- Fastest path to stop permanent user-visible hangs.

### Option B: Foreground/Background Work Separation

Summary:

- Split model-backed work into foreground query/summarize lanes and background ingest lanes, then add bounded queues and cancellation rules.

Effort:

- Medium.

Risk:

- Moderate. Requires careful tests around concurrency and cancellation.

Builds on:

- Existing `InferenceExecutor`, ingest queue, `ml_executor`, and queue status API.

Best for:

- Real daily stability while preserving the current local-process architecture.

### Option C: External Local Inference Service

Summary:

- Move embedding, reranking, and LLM work behind a separate local service, with explicit health, queues, and kill/restart behavior.

Effort:

- Large.

Risk:

- High. Adds a second service boundary and more startup/launchd complexity.

Builds on:

- Startup checks, service management, and model health reporting.

Best for:

- A later phase if in-process coordination remains fragile.

## Recommendation

Choose Option B, but implement it in two smaller phases:

1. First add bounded foreground protection and UI feedback.
2. Then separate background ingest from foreground query work.

Attack on the recommendation:

- If both foreground query and background ingest still need the same Apple Silicon runtime resources, splitting queues may not eliminate all contention.
- If cancellation cannot interrupt MLX generation promptly, a stuck foreground task may still occupy a worker until the underlying call returns.
- If too many workers run MLX at once, memory use may rise and make the machine less stable.

Deformed version after the attack:

- Keep one actual MLX execution lane for LLM/reranker calls, but add strict ownership, timeouts, and cancellation around it.
- Give foreground requests priority before background ingest.
- Allow background ingest to pause while any foreground request is active.
- Do not add uncontrolled parallel MLX workers until measurements prove it is safe.

## Chosen Approach

Use a priority-based in-process reliability layer:

```text
Browser / API
    |
    | foreground: query, stream, summarize, debug retrieve
    v
Foreground task controller
    |
    | owns bounded MLX calls, timeout, cancellation, busy state
    v
Shared MLX execution lane
    ^
    | low priority, pausable
Background ingest queue
```

This keeps the architecture simple while addressing the current real failure mode.

## Phase 10.1: Stop Permanent Hangs

### Scope

- Add bounded runtime behavior for model-backed requests.
- Make streaming query cancellation reliable.
- Surface timeout and busy states clearly to the browser.

### Tasks

1. Add a model-task controller around the current shared model worker.
2. Give each foreground request a request id, start time, and timeout.
3. For streaming query:
   - If the browser disconnects, mark the task cancelled.
   - If no citation or token is produced within a bounded time, return a visible error event.
   - Always re-enable the send button on `done` or `error`.
4. For `/api/query`, `/api/summarize`, and `/api/debug/retrieve`:
   - Return a clear timeout response instead of letting clients wait indefinitely.
5. Add logging for task start, task end, timeout, cancellation, and duration.

### Acceptance Criteria

- A deliberately cancelled browser query does not block the next query.
- A query that times out returns a visible error message in Chat.
- After a timed-out query, a new query can still run without restarting the service.
- `/api/query`, `/api/summarize`, and `/api/debug/retrieve` return bounded failures instead of hanging.
- Full test suite passes.
- Browser Chat happy path still returns a cited answer.

### Tests

- Unit test: model task timeout returns a structured error.
- Unit test: stream cancellation sets cancellation state and emits no later history write.
- API test: `/api/query` handles worker timeout.
- API test: `/api/debug/retrieve` handles worker timeout.
- Browser test: send a query, receive an answer, button re-enables.
- Browser test: simulated timeout shows a clear message and button re-enables.

## Phase 10.2: Foreground Priority Over Background Ingest

### Scope

- Keep background indexing from starving Chat.
- Make ingest queue visibly pause or slow while foreground work is active.

### Tasks

1. Add foreground-active state to the app.
2. Make ingest embedding/reranking work check that state before starting a new batch.
3. Add configurable background batch limits:
   - max files per batch.
   - max chunks per batch.
   - cooldown between batches when the browser is active.
4. Extend `/api/queue` with paused/backoff state.
5. Show paused/backoff state in the Files page queue banner.

### Acceptance Criteria

- During a large scan, Chat can still answer a short query.
- Files page shows when background ingest is paused for foreground work.
- Background ingest resumes after foreground work completes.
- Queue status remains truthful and does not hide pending work.

### Tests

- Unit test: background queue pauses while foreground-active state is true.
- Unit test: queue status reports paused/backoff state.
- Integration test: foreground query can run while queue has pending files.
- Browser test: Files page shows queue pause/resume state.

## Phase 10.3: Safer Model Switching

### Scope

- Prevent accidental expensive model switching from making the app feel stuck.

### Tasks

1. Add model availability and estimated load state to `/api/llm`.
2. Disable unavailable or expensive model switches unless the model is cached and safe to load.
3. Add loading state to the LLM dropdown.
4. If switch fails or times out, keep the current model and show a visible error.
5. Log model switch duration and result.

### Acceptance Criteria

- Switching to the current model is instant and idempotent.
- Switching to another cached model shows loading state and either succeeds or fails clearly.
- A failed model switch does not block later queries.
- `/api/llm` remains fast while no switch is running.

### Tests

- API test: unknown model returns 400.
- API test: switch timeout preserves current model.
- Frontend test: selector shows loading and error states.

## Phase 10.4: Data-Safe UI Actions

### Scope

- Protect destructive actions and reduce accidental data loss.

### Tasks

1. Add confirmation before `清空历史`.
2. Add confirmation before deleting conversations.
3. Make destructive buttons visually distinct from normal actions.
4. Keep the final API behavior unchanged, but require UI confirmation.

### Acceptance Criteria

- Clicking `清空历史` first opens a confirmation.
- Cancelling confirmation preserves history.
- Confirming deletion still clears history.
- Deleting a conversation requires confirmation.

### Tests

- Browser test: cancel clear-history preserves history count.
- Browser test: confirm clear-history clears only when explicitly confirmed, using test data.
- Browser test: cancel conversation deletion preserves conversation.

## Phase 10.5: Health and Log Polish

### Scope

- Make status easier to understand and reduce harmless noise.

### Tasks

1. Split health display into:
   - Core: query, ingest, SQLite, Qdrant.
   - Optional: OCR, enhanced LLM, VLM, contextual prefix.
2. Change UI copy so missing optional VLM reads as optional, not broken.
3. Add `/favicon.ico` handling or redirect to `/favicon.svg`.
4. Make file preview support `HEAD` or avoid misleading 404s.

### Acceptance Criteria

- A missing optional model does not make the UI look broken.
- `/favicon.ico` no longer logs 404.
- `HEAD /api/file/{id}/preview` gives a truthful result or is intentionally unsupported with a clear response.

### Tests

- API/static test for `/favicon.ico`.
- API test for preview `HEAD` behavior.
- Browser test that health panel distinguishes core and optional checks.

## Rollback Plan

- All changes are code-only and do not require database migration.
- If Phase 10.1 causes regressions, revert the task-controller changes and keep the existing worker behavior.
- If Phase 10.2 causes ingest starvation, disable foreground pause through config and restore current queue behavior.
- If Phase 10.3 model switching causes regressions, hide the selector and keep the configured default model.
- UI confirmation and health copy changes can be reverted without touching stored data.

## External Dependencies

- No new external services are required.
- Existing local dependencies remain:
  - Docker/Qdrant for vector storage.
  - Ollama for OCR.
  - MLX local model cache for LLM/reranker.
  - launchd for background service.

## Validation Before Completion

Run these before marking Phase 10 complete:

```bash
.venv/bin/python -m pytest
.venv/bin/python main.py service status
curl -s http://127.0.0.1:8000/api/health
curl -s http://127.0.0.1:8000/api/queue
curl -s -X POST http://127.0.0.1:8000/api/query \
  -H 'Content-Type: application/json' \
  -d '{"question":"DocFlow 当前状态如何？请简短回答。"}'
```

Browser validation:

- Open `http://localhost:8000`.
- Verify Chat happy path.
- Verify a cancelled or failed query recovers.
- Verify Files queue pause/resume messaging.
- Verify History clear confirmation.
- Verify model switch loading/error state.
- Verify health panel core/optional wording.

## Handoff Requirement

After implementing each Phase 10 subphase, write a handoff document under `docs/` with:

- Completed scope.
- Changed files.
- Validation commands and results.
- Known limitations.
- Exact next tasks.
