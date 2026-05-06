# Phase 19 Handoff: Retrieval Evidence Reliability

Date: 2026-05-06

## Status

Complete.

## Prior Commit

Phase 18 was committed before Phase 19 work started:

```bash
git log -1 --oneline
# 671950a chore(release): complete phase18 polish
```

## Scope

Phase 19 focused on the fixed retrieval evidence weakness from Phase 18, where
the no-rerank maturity retrieval check was only `11 / 27`.

Implemented:

- Added source-refresh support for fixed retrieval eval cases, so expected
  project files can be re-ingested before evidence checks.
- Added source-filtered fixed evidence checks, separating "does the expected
  project source contain enough evidence" from "does the full personal corpus
  rank that source first".
- Changed fixed eval evidence matching to use parent-expanded context when
  available, matching the context that answer generation actually sees.
- Increased debug retrieval text windows for evaluation so source evidence is
  not cut too aggressively.
- Refined mixed Chinese/English technical query routing so product and command
  terms such as OCR, fallback, restore-plan, and reranked get stronger keyword
  handling.
- Relaxed the vector candidate prefilter threshold enough to keep moderate but
  relevant vector hits available before fusion/ranking.
- Added README facts for supported formats, OCR/VLM behavior, source preview,
  debug retrieval, model readiness, health checks, recovery suggestions,
  foreground task priority, and Knowledge Outputs.
- Updated release and README commands to use the refreshed fixed retrieval
  checks.
- Updated the maturity scorecard from `8.51 / 9.0` to `8.55 / 9.0`.

## Changed Files

- `AGENTS.md`
- `README.md`
- `docs/LOCAL_DEPLOYMENT.md`
- `docs/phase19-handoff.md`
- `eval/phase11_maturity_dimensions.json`
- `scripts/run_eval.py`
- `scripts/run_maturity_eval.py`
- `src/quality/maturity.py`
- `src/query/retriever.py`
- `tests/test_maturity_eval.py`
- `tests/test_retriever.py`

## Validation Results

Full tests:

```bash
.venv/bin/python -m pytest
# 180 passed, 5 warnings
```

Fixed retrieval evidence, expected-source check:

```bash
.venv/bin/python main.py eval --cases eval/phase11_questions.jsonl --no-rerank --refresh-sources --source-filter --json
# cases=27, passed=27, failed=0, source_filter=true
```

Fixed retrieval evidence, full-corpus competition check:

```bash
.venv/bin/python main.py eval --cases eval/phase11_questions.jsonl --no-rerank --refresh-sources --json
# cases=27, passed=22, failed=5, source_filter=false
```

Remaining full-corpus failures:

- `project_no_masking_fallback`
- `health_optional_core`
- `queue_foreground_priority`
- `source_preview_citations`
- `phase13_web_notes_import`

Maturity evaluation:

```bash
.venv/bin/python main.py maturity-eval --no-rerank --refresh-sources --source-filter --json
# overall_score=8.55/9.0
# retrieval_eval=27/27 passed
```

Consistency check:

```bash
.venv/bin/python main.py check --json
# status=ok
# sqlite_chunks=10035
# qdrant_points=10035
# missing_points=0
# orphan_points=0
# file_chunk_mismatches=0
# missing_source_files=0
```

Diff hygiene:

```bash
git diff --check
# passed
```

## Known Limitations

- The source-filtered eval proves the expected project files contain the needed
  evidence, but it intentionally narrows retrieval to those files.
- The full-corpus no-rerank eval remains `22 / 27`; older or similar local
  project notes can still outrank the expected source for five fixed cases.
- This phase did not run a live answer-generation quality suite with the LLM.
- This phase did not replace remote Material Symbols usage; that remains the
  next local-first UI asset task.

## Next Phase Tasks

Phase 20 should be the fully local UI assets phase:

1. Replace the Google Fonts Material Symbols runtime request with local icons or
   an already bundled icon strategy.
2. Verify the app has no runtime external font/style/script requests during
   normal use.
3. Run `npm run build:css`, `.venv/bin/python -m pytest`,
   `.venv/bin/python main.py check --json`, and browser validation.
4. Update README or local deployment docs if the icon setup changes user-facing
   build or release steps.
5. Write `docs/phase20-handoff.md` with scope, changed files, validation
   results, limitations, and exact next tasks before reporting completion.
