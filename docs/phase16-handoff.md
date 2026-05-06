# Phase 16 Handoff: Runtime Health and Recovery Guidance

Date: 2026-05-06

## Status

Complete.

## Scope

Phase 16 improved model/runtime, health, setup, and recovery maturity.

Implemented:

- Added model runtime readiness to `/api/health` for embedding, reranker, default LLM, enhanced LLM, OCR, and VLM.
- Added safe recommended actions to `/api/health`, including read-only checks, dry-run backup previews, Ollama guidance, OCR model pulls, and model-cache guidance.
- Changed Ollama health handling so a closed Ollama service reports degraded optional capability with clear guidance instead of a generic failed check.
- Expanded `/api/llm` model status with cache details and model-preparation guidance.
- Updated Settings to show core health, model runtime, optional capabilities, watched folders, and recovery suggestions.
- Added copy buttons for suggested commands. The UI does not auto-run repair or destructive actions.
- Improved Settings mobile layout so model controls do not squeeze headings into vertical text.
- Updated README, maturity scoring, fixed evidence questions, and tests.

## Changed Files

- `src/api/app.py`
- `frontend/index.html`
- `tests/test_api_health.py`
- `tests/test_api_llm.py`
- `tests/test_static_assets.py`
- `README.md`
- `eval/phase11_maturity_dimensions.json`
- `eval/phase11_questions.jsonl`
- `docs/phase16-handoff.md`

## Validation Results

Targeted tests:

```bash
.venv/bin/python -m pytest tests/test_api_health.py tests/test_api_llm.py tests/test_static_assets.py tests/test_maturity_eval.py
# 27 passed, 5 warnings
```

Maturity score:

```bash
.venv/bin/python -m json.tool eval/phase11_maturity_dimensions.json >/tmp/docflow-phase16-maturity.json
.venv/bin/python main.py maturity-eval --skip-retrieval
# Overall score: 8.01/9.0
```

Live API validation:

```bash
.venv/bin/python <urllib /api/health check>
# status=ok
# groups=['core', 'runtime', 'optional']
# runtime=['向量模型', '精排模型', '回答模型', '增强回答模型', 'OCR 模型', '图片理解模型']
# actions include python main.py check --json and python main.py backup --dry-run
```

Browser validation:

```bash
.venv/bin/python <one-off Playwright Phase 16 Settings check>
# phase16 browser settings check passed
# screenshots:
# /tmp/docflow-phase16-settings-top.png
# /tmp/docflow-phase16-settings-bottom.png
# /tmp/docflow-phase16-mobile-settings.png
```

Covered in the browser flow:

- Restarted the local app on `http://127.0.0.1:8000` with the current code.
- Opened Settings.
- Verified model runtime labels for embedding, reranker, LLM, OCR, and VLM.
- Verified recovery suggestions render.
- Clicked a copy-command button.
- Checked desktop Settings top and bottom screenshots.
- Checked Settings at a 390px-wide viewport after fixing mobile header squeezing.

Full validation:

```bash
.venv/bin/python -m pytest
# 164 passed, 5 warnings

.venv/bin/python main.py check --json
# status=ok, sqlite_chunks=9850, qdrant_points=9850

.venv/bin/python main.py maturity-eval --skip-retrieval
# Overall score: 8.01/9.0

git diff --check
# clean
```

## Known Limitations

- The Settings page displays and copies recovery commands; it intentionally does not execute repair commands from the browser.
- Model cache guidance is explicit but still manual. There is no model download progress UI yet.
- Real restore rehearsal is still pending; backup and restore commands exist, but Phase 16 did not perform a destructive restore exercise.
- The page still uses Tailwind CDN, which remains a later production packaging task.

## Next Phase

Continue with Phase 17: improve local knowledge workflow maturity.

Recommended Phase 17 tasks:

1. Add reusable knowledge output workflows such as summaries, learning cards, action items, and project briefs.
2. Let saved answers or selected files flow into those templates from the browser UI.
3. Keep generated outputs as local Markdown files with clear collection/tag metadata.
4. Add tests for template generation, saved outputs, and UI entrypoints.
5. Run full tests, browser validation, maturity score update, and write `docs/phase17-handoff.md`.
