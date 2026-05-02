# Phase 2 Handoff: Retrieval Quality Improvements

Date: 2026-05-02

## Status

Phase 2 retrieval-quality scope is implemented for sections 2.3 through 2.6 of the optimization plan.

Completed scope:

- Added Markdown chunking regression coverage for nested headings, empty parent headings, long section splitting, and tables immediately after headings.
- Added Obsidian Markdown cleanup coverage to ensure headings survive parser cleanup.
- Added parent context grouping during ingest.
- Stored child raw text, embedding text, parent id, parent text, and contextual prefix data in SQLite.
- Kept Qdrant embeddings on child chunks while storing enough payload data to inspect raw and parent context.
- Expanded retrieved child hits back to parent context before answer generation.
- Added a default-off contextual prefix framework.
- Added adaptive QueryRouter output for query type, retrieval count, rerank count, and retrieval weights.
- Exposed the router decision and parent-expanded stage through retrieval debugging.
- Fixed old-database migration order for the new chunk fields.
- Updated README and config references for the new retrieval behavior.

## Files Changed

- `README.md`
  - Added parent context retrieval and adaptive routing notes.
  - Added contextual prefix and parent context configuration notes.
  - Added retrieval debug output description.
- `config.yaml`
  - Added `ingest.parent_context_chars`.
  - Added default-off contextual prefix settings.
- `src/ingest/chunker.py`
  - Added raw text, embedding text, parent context, and contextual prefix fields to chunks.
- `src/ingest/pipeline.py`
  - Added parent context grouping.
  - Added contextual prefix framework.
  - Switched embedding cache and embedding input to `embedding_text`.
  - Kept FTS indexing on raw text.
- `src/ingest/embedder.py`
  - Stores raw child text plus parent/context metadata in Qdrant payloads.
- `src/ingest/store.py`
  - Added SQLite migration and storage for parent/context fields.
  - Added parent context lookup by Qdrant id.
  - Fixed migration order for old databases.
- `src/query/retriever.py`
  - Added adaptive routing candidate counts.
  - Added parent context expansion before answer generation.
  - Added parent-expanded debug stage.
- `tests/test_chunker.py`
  - Added Markdown boundary tests.
- `tests/test_markdown_parser.py`
  - Added Obsidian cleanup heading-preservation test.
- `tests/test_pipeline.py`
  - Added parent context and contextual prefix tests.
- `tests/test_retriever.py`
  - Added QueryRouter and parent expansion tests.
- `tests/test_store.py`
  - Added parent context storage and old DB migration tests.
- `docs/phase2-handoff.md`
  - This handoff.

## How To Use

Parent context is enabled by the ingest path automatically. Re-ingest files after this phase so existing chunks receive parent metadata:

```bash
.venv/bin/python main.py ingest README.md
```

Contextual prefix remains off by default:

```yaml
ingest:
  contextual_prefix: false
```

To inspect the retrieval route and parent-expanded stage:

```bash
curl -X POST http://localhost:8000/api/debug/retrieve \
  -H "Content-Type: application/json" \
  -d '{"question":"为什么 Parent-Child Chunking 不能只加 parent_id？","include_rerank":false}'
```

## Validation Results

Commands that passed:

```bash
.venv/bin/python -m pytest
```

Result:

- 78 passed
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

Qdrant connectivity:

```bash
curl -s http://localhost:6333/collections
```

Result:

- Qdrant was reachable.
- The `docflow` collection exists.

Manual re-ingest after Phase 2:

```bash
.venv/bin/python main.py ingest README.md
.venv/bin/python main.py ingest AGENTS.md
.venv/bin/python main.py ingest /Users/hughlin/MyNotes/HughLin/Notes/plans/docflow/docflow-optimization-plan.md
```

Result:

- `README.md`: 33 chunks indexed.
- `AGENTS.md`: 8 chunks indexed.
- `docflow-optimization-plan.md`: 38 chunks indexed.

Retrieval debug spot check:

```bash
.venv/bin/python - <<'PY'
from src.query.engine import QueryEngine
engine = QueryEngine.from_config('config.yaml')
result = engine.retriever.debug_retrieve(
    '为什么 Parent-Child Chunking 不能只加 parent_id？',
    file_filter=['docflow-optimization-plan.md'],
    include_rerank=False,
    max_text_chars=160,
)
print(result['router'])
print(len(result['stages']['deduped']), len(result['stages']['parent_expanded']))
print(result['stages']['parent_expanded'][0]['file_name'])
print(result['stages']['parent_expanded'][0]['parent_id'])
print(result['stages']['parent_expanded'][0]['text_length'])
PY
```

Result:

- Router returned `query_type=semantic`, `top_k_retrieval=24`, `top_k_rerank=8`.
- `deduped=11`, `parent_expanded=11`.
- First result came from `docflow-optimization-plan.md`.
- First result had a non-zero parent id and parent text length of 432.

Fixed evaluation after re-ingest:

```bash
.venv/bin/python main.py eval --no-rerank --json
```

Result:

- 15 cases executed.
- 8 passed.
- 7 failed.
- Exit code was 1 because the eval set is a quality baseline and still has failing cases.

Interpretation:

- Re-ingesting the current project docs improved the Phase 1 baseline from 2/15 to 8/15.
- The remaining failures are mostly corpus/ranking quality issues, not runner failures.
- Several fixed cases still expect `config.yaml` or README matches that are not consistently in the top returned files.

## Known Limitations

- Existing indexed files need re-ingest before they receive parent/context fields.
- Contextual prefix is implemented as a default-off framework. The safe local metadata mode is available; Ollama mode exists but should be evaluated before turning it on broadly.
- Parent context grouping is conservative: adjacent chunks are grouped by file path, page number, and section, with `ingest.parent_context_chars` as the cap.
- The eval runner still uses the current mixed local corpus, so failures can reflect older similarly named DocFlow notes outranking current project files.
- Phase 2 did not implement full dependency health checks. That remains the next reliability task.

## Next Phase

Start with optimization plan section 2.7: real health checks.

Exact next tasks:

1. Add `/api/health` checks for Qdrant connectivity.
2. Add SQLite read/write health checks.
3. Add Ollama connectivity status and mark OCR/contextual prefix availability clearly.
4. Check configured model availability in a lightweight way without forcing downloads.
5. Update README health-check documentation to match the real response.
6. Add tests for healthy, degraded, and unavailable dependency states.
7. Run `.venv/bin/python -m pytest`, `git diff --check`, and a live `/api/health` request before writing the next phase handoff.
