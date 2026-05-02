# Phase 3 Handoff: Real Health Checks

Date: 2026-05-02

## Status

Phase 3 is implemented for optimization plan section 2.7: real health checks.

Completed scope:

- Replaced the fixed `/api/health` response with real dependency checks.
- Added SQLite read/write and `PRAGMA quick_check` validation.
- Added Qdrant collection connectivity validation.
- Added Ollama reachability and configured Ollama model availability checks.
- Added lightweight local model cache checks without downloading models.
- Added explicit capability flags for query, ingest, OCR, and contextual prefix.
- Kept non-critical dependency failures as `degraded` instead of blocking query and ingest.
- Updated README health-check documentation.
- Added tests for healthy, degraded, unavailable, and exception cases.

## Files Changed

- `src/api/app.py`
  - Added structured health response.
  - Added SQLite, Qdrant, Ollama, and local model cache checks.
  - Added health aggregation and capability flags.
- `tests/test_api_health.py`
  - Added health endpoint tests for core states.
- `README.md`
  - Updated `/api/health` description in English and Chinese.
- `docs/phase3-handoff.md`
  - This handoff.

## Health Response Shape

`/api/health` now returns:

- `status`: `ok`, `degraded`, or `unavailable`.
- `checks.api`: FastAPI process status.
- `checks.sqlite`: SQLite read/write and quick-check result.
- `checks.qdrant`: Qdrant collection status and point count.
- `checks.ollama`: Ollama reachability and configured Ollama model availability.
- `checks.models`: local Hugging Face cache presence for configured local models.
- `capabilities.query`: true only when SQLite and Qdrant are available.
- `capabilities.ingest`: true only when SQLite and Qdrant are available.
- `capabilities.ocr`: true only when the configured OCR model is available through Ollama.
- `capabilities.contextual_prefix`: true only when contextual prefix is enabled and its configured mode is available.

Critical failures:

- SQLite unavailable.
- Qdrant unavailable.

Non-critical degradation:

- Ollama unavailable.
- Optional Ollama model missing.
- Local model cache missing.

## Validation Results

Commands that passed:

```bash
.venv/bin/python -m pytest
```

Result:

- 82 passed
- 5 warnings from third-party SWIG/PyMuPDF imports

```bash
/Users/hughlin/.codex/skills/readme-maintainer/scripts/check_bilingual_readme.sh README.md
```

Result:

- Bilingual README parity check passed

```bash
git diff --check
```

Result:

- No whitespace errors

Live `/api/health` request:

```bash
.venv/bin/python -m uvicorn src.api.app:app --host 127.0.0.1 --port 8010 --lifespan off
curl -s http://127.0.0.1:8010/api/health
```

Result:

- HTTP 200.
- Overall status was `degraded`.
- SQLite was `ok`.
- Qdrant was `ok`, collection `docflow`, `points_count=5172`.
- Ollama was `ok` after the user started it.
- Configured OCR model `glm-ocr` was available through Ollama.
- Query capability was `true`.
- Ingest capability was `true`.
- OCR capability was `true`.
- Contextual prefix was `false` because it is disabled in config.
- Local cache check found embedding, reranker, and default LLM cached.
- Local cache check reported enhanced LLM and VLM caches missing.

Interpretation:

- The endpoint correctly detects Ollama recovery after the user starts Ollama.
- The endpoint correctly avoids reporting full failure when only optional local model-cache capabilities are missing.
- The current local app can still query and ingest because SQLite and Qdrant are healthy.

## Known Limitations

- The local model check only inspects Hugging Face cache folders. It does not verify that model files can fully load.
- The health endpoint does not start or download models.
- Running the app with normal lifespan still performs startup warmup separately; the live validation above used `--lifespan off` to isolate the health endpoint without loading models.
- If Ollama is not running, the endpoint reports OCR as unavailable while query and ingest remain available.

## Next Phase

Start with optimization plan section 2.8: query path degradation.

Exact next tasks:

1. Add bounded failure handling around Qdrant vector search.
2. Keep FTS retrieval available when vector search fails.
3. Return fused or FTS-only results when reranker fails.
4. Surface degraded retrieval state clearly in debug output and, where useful, API responses.
5. Add tests for Qdrant failure, FTS failure, reranker failure, and partial-result behavior.
6. Run `.venv/bin/python -m pytest`, `git diff --check`, and a real degraded retrieval check before writing the next phase handoff.
