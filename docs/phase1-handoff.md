# Phase 1 Handoff: Evaluation and Retrieval Debugging

Date: 2026-05-02

Update: Phase 2 is now documented in `docs/phase2-handoff.md`. The "Next Phase" items below describe what was next at the end of Phase 1, not the current next step.

## Status

Phase 1 is implemented.

Completed scope:

- Added the project rule that every completed phase must produce a handoff document.
- Added a fixed retrieval evaluation dataset.
- Added `python main.py eval` as the one-command entry point for retrieval evaluation.
- Added a file chunk debugging endpoint.
- Added a retrieval pipeline debugging endpoint.
- Added unit coverage for the new store and retriever behavior.
- Updated README command and API references for the new Phase 1 tools.

## Files Changed

- `AGENTS.md`
  - Added the phase handoff rule.
- `README.md`
  - Added `python main.py eval`.
  - Added `/api/file/{id}/chunks`.
  - Added `/api/debug/retrieve`.
- `main.py`
  - Added `eval` command.
- `scripts/run_eval.py`
  - New retrieval evaluation runner.
- `eval/phase1_questions.jsonl`
  - New fixed evaluation cases.
- `src/ingest/store.py`
  - Added `list_file_chunks(file_id)`.
- `src/query/retriever.py`
  - Added `debug_retrieve(...)`.
  - Added `fetch_chunks_by_ids(...)`.
  - Added debug item shaping for retrieval stages.
- `src/api/app.py`
  - Added `GET /api/file/{file_id}/chunks`.
  - Added `POST /api/debug/retrieve`.
- `tests/test_store.py`
  - Added chunk metadata test.
- `tests/test_retriever.py`
  - Added debug retrieval tests.
- `tests/test_api_debug.py`
  - Added endpoint tests for chunk inspection and retrieval debugging.
- `docs/phase1-handoff.md`
  - This handoff.

## How To Use

Run the fixed retrieval evaluation set:

```bash
.venv/bin/python main.py eval
```

Skip reranker for a faster retrieval-only check:

```bash
.venv/bin/python main.py eval --no-rerank
```

Emit JSON:

```bash
.venv/bin/python main.py eval --no-rerank --json
```

Inspect chunks for one indexed file:

```bash
curl "http://localhost:8000/api/file/1/chunks"
```

Inspect the retrieval pipeline:

```bash
curl -X POST http://localhost:8000/api/debug/retrieve \
  -H "Content-Type: application/json" \
  -d '{"question":"DocFlow 支持哪些文件格式？","include_rerank":false}'
```

## Validation Results

Commands that passed:

```bash
.venv/bin/python -m pytest
```

Result:

- 66 passed
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

```bash
.venv/bin/python main.py eval --help
```

Result:

- CLI help printed successfully

Initial command that could not complete before Docker was started:

```bash
.venv/bin/python main.py eval --no-rerank --json
```

Result:

- Failed before completion because Qdrant was not reachable on `localhost:6333`.
- Docker was not running: Docker socket `/Users/hughlin/.docker/run/docker.sock` was unavailable.
- The eval runner returned a clear dependency message instead of a traceback.

After Docker/Qdrant was started, the same command completed and produced a baseline:

```bash
.venv/bin/python main.py eval --no-rerank --json
```

Result:

- 15 cases executed
- 2 passed
- 13 failed
- Exit code was 1 because this is a quality baseline and not all expected cases passed

Interpretation:

- The runner and Qdrant path work.
- The current Qdrant `docflow` collection does not yet reflect the updated README, new AGENTS.md, and updated optimization plan content.
- Many failures missed expected files such as `README.md`, `AGENTS.md`, and `docflow-optimization-plan.md` while matching older DocFlow notes instead.
- Before using this eval as a regression gate, re-ingest the current project docs and the relevant notes plan so the index matches the current workspace.

## Known Limitations

- The eval runner depends on a running Qdrant instance and an already-ingested corpus matching the fixed cases.
- The fixed cases currently target project-local docs plus the external optimization plan under the configured notes directory. If those files are stale or not ingested, the eval fails even though the runner itself works.
- `/api/debug/retrieve` can load the embedding model and, when `include_rerank=true`, the reranker. Use `include_rerank=false` for faster checks.
- At the end of Phase 1, `/api/file/{id}/chunks` read text previews from Qdrant payloads because SQLite stored chunk metadata only. Phase 2 added SQLite raw text and parent context storage.

## Next Phase

Phase 2 should start with Markdown chunking reinforcement:

1. Add tests for multi-level Markdown headings, headings without body text, long sections that need recursive splitting, and tables immediately after headings.
2. Confirm `section` survives through retrieval output, citations, and summaries.
3. Run `.venv/bin/python -m pytest`.
4. Re-ingest the current `README.md`, `AGENTS.md`, and `/Users/hughlin/MyNotes/HughLin/Notes/plans/docflow/docflow-optimization-plan.md`, then run `.venv/bin/python main.py eval --no-rerank --json` as the clean baseline before changing Parent-Child or Contextual Prefix behavior.

After Phase 2 completes, update or create the next handoff document under `docs/`.
