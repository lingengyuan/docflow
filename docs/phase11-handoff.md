# Phase 11 Handoff: 9-Point Maturity Baseline

Date: 2026-05-04

## Status

Complete.

## Scope

Phase 11 established the baseline for the 9-point mature personal-local DocFlow target.

Implemented:

- Added a structured maturity scorecard for all 12 dimensions.
- Added an expanded fixed retrieval evidence set.
- Added a `maturity-eval` command that reports dimension scores plus retrieval evidence results.
- Updated README command references.
- Documented the acceptance baseline in `docs/phase11-9-point-acceptance.md`.

## Changed Files

- `main.py`
- `README.md`
- `scripts/run_eval.py`
- `scripts/run_maturity_eval.py`
- `src/quality/__init__.py`
- `src/quality/maturity.py`
- `eval/phase11_maturity_dimensions.json`
- `eval/phase11_questions.jsonl`
- `tests/test_maturity_eval.py`
- `docs/phase11-9-point-acceptance.md`
- `docs/phase11-handoff.md`

The existing `config.yaml` change from the prior image-model installation is still present and should be kept unless the user asks to change the VLM model again.

## Validation Results

Targeted tests:

```bash
.venv/bin/python -m pytest tests/test_maturity_eval.py
# 4 passed

.venv/bin/python main.py maturity-eval --skip-retrieval
# Overall score: 6.96/9.0
```

Live retrieval evidence baseline:

```bash
.venv/bin/python main.py maturity-eval --no-rerank --json
# overall=6.96/9.0
# retrieval=11/20 passed, 9 failed
```

Full test suite:

```bash
.venv/bin/python -m pytest
# 138 passed, 5 warnings
```

Runtime consistency:

```bash
.venv/bin/python main.py check --json
# status=ok, sqlite_chunks=9831, qdrant_points=9831
```

## Known Limitations

- The retrieval evidence baseline currently has failures because the live corpus includes older DocFlow notes that can outrank current project files.
- The maturity scores are a human-curated baseline stored in JSON. Later phases should update them when the corresponding user-facing behavior actually improves.
- The command intentionally exits successfully when scores are below 9; below-target scores are the report content, not a command failure.

## Next Phase

Continue with Phase 12: strengthen file management and knowledge organization.

Recommended Phase 12 tasks:

1. Add collection and user-tag storage.
2. Add collection/tag filtering to the files API.
3. Upgrade Files into a Library view with collection/tag controls and source details.
4. Add batch operations for favorite, summarize, and rebuild where appropriate.
5. Update the maturity dimensions after the library-management behavior is verified.
6. Run targeted tests, full tests, browser validation, and write `docs/phase12-handoff.md`.
