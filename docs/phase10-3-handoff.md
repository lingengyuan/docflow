# Phase 10.3 Handoff: Safer Model Switching

Date: 2026-05-03

## Status

Complete.

## Completed Scope

- `/api/llm` now reports model cache and availability state.
- Switching to the current model returns immediately and does not reload the model.
- Unavailable MLX models are rejected before attempting a switch.
- Model switching exposes `idle`, `switching`, and `error` state.
- Timed-out model switches return a visible error and do not block later requests.
- MLX switching now loads a candidate model first and only applies it after a successful load, so timeout work cannot later overwrite the current model.
- The model selector shows:
  - current model.
  - locally available model.
  - switching state.
  - visible failure message.

## Changed Files

- `src/api/app.py`
  - Added model availability metadata to `/api/llm`.
  - Added model switch state.
  - Added idempotent current-model switch behavior.
  - Changed MLX switching to apply a loaded candidate only after success.
- `frontend/index.html`
  - Shows `当前使用`, `本地可用`, `切换中`, and error text.
  - Disables the model selector while a switch is in progress.
- `tests/test_api_llm.py`
  - Added API coverage for model availability, unknown model, current-model idempotence, successful switch, and switch timeout.
- `docs/phase10-optimization-plan.md`
  - Updated status after Phase 10.3 completion.

## Validation

Targeted tests:

```bash
.venv/bin/python -m pytest tests/test_api_llm.py tests/test_model_tasks.py
```

Result:

- `7 passed`

Full test suite:

```bash
.venv/bin/python -m pytest
```

Result:

- `127 passed`
- `5 warnings` from third-party SWIG bindings, unchanged from prior runs.

Live service validation after `launchctl kickstart -k gui/$(id -u)/com.docflow.local`:

```bash
curl -s http://127.0.0.1:8000/api/llm
curl -s -X POST http://127.0.0.1:8000/api/llm \
  -H 'Content-Type: application/json' \
  -d '{"model":"mlx-community/Qwen3-4B-4bit"}'
curl -s -X POST http://127.0.0.1:8000/api/query ...
```

Result:

- `/api/llm` reports both configured MLX models as cached and available.
- Switching to the current 4B model returned immediately with `unchanged: true`.
- Chat still answered normally after model switch operations, with citations.

Browser validation:

- Opened `http://localhost:8000`.
- Opened the model selector.
- Confirmed dropdown shows:
  - `mlx-community/Qwen3-4B-4bit` as `当前使用`.
  - `mlx-community/Qwen3-8B-4bit` as `本地可用`.
- Clicking the current model closes the dropdown and keeps status at `Local LLM`.
- Attempting the 8B switch showed visible `切换中`.
- The 8B switch timed out after 90s in this environment and the UI showed the timeout message.
- `/api/llm` still reported current model as 4B afterward.
- A follow-up Chat query returned normally after the timeout.
- Screenshot artifacts:
  - `output/playwright/phase10-3-llm-timeout.png`
  - `output/playwright/phase10-3-llm-selector.png`

## Known Limitations

- The 8B model switch exceeded the current 90s timeout in live validation, even though the model cache exists. The app now handles this visibly and keeps using 4B.
- Browser devtools logs one expected failed resource entry when `/api/llm` returns HTTP 504 during a timed-out switch.
- The UI shows model availability from local cache checks, not a precise load-time estimate.

## Next Tasks

Continue with Phase 10.4: Data-Safe UI Actions.

Exact next steps:

1. Add confirmation before clearing all history.
2. Add confirmation before deleting conversations.
3. Make destructive UI actions visually distinct and consistent.
4. Browser-test cancel and confirm flows using temporary test data.
