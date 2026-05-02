# Phase 4 Handoff: Query Path Degradation

Date: 2026-05-02

Update: Phase 5 is now documented in `docs/phase5-handoff.md`. The "Next Phase" items below describe what was next at the end of Phase 4, not the current next step.

## Status

Phase 4 is implemented for optimization plan section 2.8: query path degradation.

Completed scope:

- Added failure handling around vector search so FTS can still return results when Qdrant is unavailable.
- Changed FTS retrieval to read raw text and parent context from SQLite, so keyword fallback no longer depends on Qdrant payload fetch.
- Added failure handling around FTS search, with debug output showing when FTS itself is unavailable.
- Added reranker fallback: if reranking fails or returns no results, fused candidates are returned.
- Added answer-generation fallback: if the LLM fails after retrieval succeeds, the response keeps retrieved citations and clearly says the answer model is unavailable.
- Added stream-generation fallback for LLM failures.
- Added frontend interrupted-stream messaging when the connection ends without a `done` or `error` event.
- Added degraded retrieval status and degradation reasons to debug output.
- Updated README degradation behavior in English and Chinese.

## Files Changed

- `src/query/retriever.py`
  - Added vector, FTS, FTS-payload, and reranker degradation handling.
  - Added degraded status and degradation details to `debug_retrieve`.
  - Added fused-candidate fallback when reranking fails.
- `src/ingest/store.py`
  - Returned raw text and parent context fields from FTS queries.
- `src/query/engine.py`
  - Added LLM failure fallback for normal and streaming queries.
- `frontend/index.html`
  - Added visible interrupted-stream messaging.
- `tests/test_retriever.py`
  - Added vector failure, FTS failure, reranker failure, and SQLite-payload fallback tests.
- `tests/test_engine.py`
  - Added LLM failure fallback tests.
- `README.md`
  - Documented degraded retrieval behavior.
- `docs/phase4-handoff.md`
  - This handoff.

## Behavior

When vector search fails:

- Debug status becomes `degraded`.
- The vector stage is empty.
- FTS can still return SQLite-backed results.
- Parent context expansion still runs.

When Qdrant payload fetch fails during FTS:

- FTS results use SQLite `raw_text` and `parent_text`.
- Debug output includes a `fts_payload` degradation reason.

When reranker fails:

- Debug status becomes `degraded`.
- Fused candidates are returned.
- Returned debug items include `rerank_fallback=true`.

When answer generation fails:

- Query response returns a clear fallback message.
- Citations from retrieved chunks are still returned.
- Streaming query returns the fallback message instead of surfacing a raw exception.

## Validation Results

Commands that passed:

```bash
.venv/bin/python -m pytest
```

Result:

- 89 passed
- 5 warnings from third-party SWIG/PyMuPDF imports

```bash
awk '/<script>/{flag=1;next}/<\\/script>/{flag=0}flag' frontend/index.html | node --check
```

Result:

- Frontend script syntax check passed.

Manual degraded retrieval check:

```bash
.venv/bin/python - <<'PY'
import logging
from pathlib import Path
import yaml
from qdrant_client import QdrantClient

from src.embedding_backend import embedding_backend_config_from_dict
from src.query.retriever import HybridRetriever

logging.getLogger().setLevel(logging.CRITICAL)
with open('config.yaml') as f:
    cfg = yaml.safe_load(f)
retriever = HybridRetriever(
    qdrant_host='127.0.0.1',
    qdrant_port=6334,
    db_path=Path(cfg['paths']['db_path']).expanduser(),
    embedding_config=embedding_backend_config_from_dict(cfg, 'config.yaml'),
)
retriever._qdrant = QdrantClient(host='127.0.0.1', port=6334, timeout=1)
result = retriever.debug_retrieve(
    'README 里怎么描述 /api/health 当前的能力状态？',
    file_filter=['README.md'],
    include_rerank=False,
    max_text_chars=80,
)
print('status', result['status'])
print('degradations', [d['stage'] for d in result['degradations']])
print('fts_count', len(result['stages']['fts']))
print('parent_expanded_count', len(result['stages']['parent_expanded']))
if result['stages']['parent_expanded']:
    first = result['stages']['parent_expanded'][0]
    print('first_file', first['file_name'])
    print('first_status', first['retrieval_status'])
PY
```

Result:

- `status degraded`
- `degradations ['vector', 'fts_payload']`
- `fts_count 4`
- `parent_expanded_count 2`
- `first_file README.md`
- `first_status degraded`

Other checks that passed:

```bash
/Users/hughlin/.codex/skills/readme-maintainer/scripts/check_bilingual_readme.sh README.md
git diff --check
```

Result:

- Bilingual README parity check passed.
- No whitespace errors.

## Known Limitations

- The vector search call still relies on the Qdrant client timeout behavior. The fallback works after the client raises.
- FTS fallback depends on files being re-ingested after Phase 2 so SQLite has `raw_text` and `parent_text`.
- The normal `/api/query` response does not add a top-level degradation field; detailed degradation reasons are exposed through `/api/debug/retrieve`.
- The frontend now shows an interrupted-stream message if the stream ends unexpectedly, but there is no automated browser test for this UI state.

## Next Phase

Start with optimization plan section 2.9: consistency check and rebuild.

Exact next tasks:

1. Add a `check` command that compares SQLite chunk records with Qdrant points.
2. Detect missing Qdrant points for SQLite chunks.
3. Detect orphan Qdrant points that no SQLite chunk references.
4. Detect file records whose chunk count no longer matches stored chunks.
5. Add a `rebuild` command that rebuilds SQLite and Qdrant from source files.
6. Add `rebuild --qdrant-only` using SQLite-stored chunk text.
7. Add tests for mismatch detection and rebuild planning.
8. Run `.venv/bin/python -m pytest`, `git diff --check`, and a live consistency check before writing the next phase handoff.
