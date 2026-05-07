# Phase 23 Handoff: Vector ID Allocation Hardening

Date: 2026-05-07

## Status

Complete.

## Prior Commit

Phase 22 was committed before Phase 23 work started:

```bash
git log -1 --oneline
# cee0aa4 test(recovery): complete phase22 restore drill
```

## Scope

Phase 23 fixed the root cause behind the duplicate Qdrant IDs found in Phase 22.

Implemented:

- Added file-lock protected vector ID reservation in `Embedder`.
- Changed ID allocation so each ingest reserves IDs from the counter under an
  interprocess lock before writing Qdrant points.
- Changed the counter loader to use the highest Qdrant point ID plus one when
  the counter file is missing or invalid, instead of using point count.
- Added an ingest-time safety floor:
  - SQLite max `qdrant_id`.
  - Qdrant max point ID.
  - Existing counter value.
  - New IDs start at the highest of those values plus one.
- Extended `python main.py check --json` counter reporting so expected minimum
  uses both SQLite and Qdrant.
- Added `python main.py repair-ids --dry-run` and `python main.py repair-ids`.
  - Dry-run reports stale counters, duplicate vector IDs, and affected files.
  - Non-dry-run advances the counter and reingests files affected by duplicate
    vector IDs.
- Updated README, local deployment guide, changelog, and maturity scorecard.

## Changed Files

- `CHANGELOG.md`
- `README.md`
- `docs/LOCAL_DEPLOYMENT.md`
- `docs/phase23-handoff.md`
- `eval/phase11_maturity_dimensions.json`
- `main.py`
- `src/ingest/embedder.py`
- `src/ingest/pipeline.py`
- `src/ingest/store.py`
- `src/maintenance/consistency.py`
- `tests/test_embedder_ids.py`
- `tests/test_pipeline.py`
- `tests/test_consistency.py`
- `tests/test_store.py`

## Validation Results

Targeted tests:

```bash
.venv/bin/python -m pytest tests/test_embedder_ids.py tests/test_pipeline.py tests/test_consistency.py tests/test_store.py
# 34 passed, 5 warnings
```

Repair dry-run on current live project:

```bash
.venv/bin/python main.py repair-ids --dry-run
# status=dry_run
# id_counter.status=ok
# min_next_id=29345
# duplicate_qdrant_ids=[]
# affected_files=[]
# actions=[]
```

Live consistency:

```bash
.venv/bin/python main.py check --json
# status=ok
# sqlite_chunks=10099
# qdrant_points=10099
# id_counter.value=29345
# id_counter.expected_min=29345
# id_counter.status=ok
# duplicate_qdrant_ids=[]
```

Full tests:

```bash
.venv/bin/python -m pytest
# 195 passed, 5 warnings
```

Restore drill:

```bash
.venv/bin/python main.py restore-drill --json
# status=ok
# passed=11
# failed=0
# files=265
# chunks=10099
# chunks_jsonl=10099
# quick_check=ok
# duplicate_qdrant_ids=[]
# id_counter.value=29345
# id_counter.expected_min=29345
```

Maturity score:

```bash
.venv/bin/python main.py maturity-eval --skip-retrieval --json
# overall_score=8.68/9.0
```

README bilingual parity:

```bash
/Users/hughlin/.agents/skills/readme-maintainer/scripts/check_bilingual_readme.sh README.md
# Bilingual README parity check passed.
```

Diff hygiene:

```bash
git diff --check
# passed
```

## Known Limitations

- The lock protects the local counter file on this machine. It is not a
  distributed lock for multiple machines writing the same Qdrant collection.
- `repair-ids` fixes stale counters and duplicate IDs by reingesting affected
  files; it does not repair arbitrary file corruption or missing source files.
- Full overwrite restore in an isolated copied project is still a separate
  rehearsal task.
- `repair-ids` non-dry-run should be used only after reading the dry-run output.

## Next Phase Tasks

Phase 24 can return to the broader product maturity plan:

1. Continue packaging/local installer work now that recovery and ID allocation
   are hardened.
2. Add a browser-visible maintenance entry for restore drill and repair dry-run
   if the Settings workflow should expose these newer commands.
3. Run the full release checklist, including restore drill, sample suite,
   consistency, maturity evaluation, and browser smoke.
4. Write `docs/phase24-handoff.md` before reporting Phase 24 completion.
