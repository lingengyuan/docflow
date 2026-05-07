# Phase 22 Handoff: Restore Drill and Index Safety Checks

Date: 2026-05-07

## Status

Complete.

## Prior Commit

Phase 21 was committed before Phase 22 work started:

```bash
git log -1 --oneline
# 3a184c2 test(samples): complete phase21 sample suite
```

## Scope

Phase 22 added a repeatable restore drill and tightened index safety checks.

Implemented:

- Added `python main.py restore-drill`.
- Added `scripts/run_restore_drill.py`.
- Added `restore_drill()` to the backup helpers.
- The restore drill now:
  - Creates a real backup from the current project.
  - Extracts it into `/tmp/docflow-phase22-restore-drill`.
  - Verifies required archive members.
  - Runs SQLite `quick_check` on the extracted database.
  - Compares manifest file and chunk counts with restored SQLite rows.
  - Checks exported `chunks.jsonl` row count.
  - Checks duplicate Qdrant vector IDs in SQLite.
  - Checks the Qdrant ID counter is not behind the restored index.
  - Checks file chunk metadata and source file paths.
  - Confirms `restore-plan` can read the generated archive.
- Added an output-directory guard so restore drill refuses to clear an existing
  unmarked directory.
- Extended `python main.py check --json` so it reports:
  - Actual SQLite chunk row count, not only unique vector IDs.
  - Duplicate Qdrant vector IDs.
  - Qdrant ID counter status.
- Added tests for restore drill success, missing source files, duplicate vector
  ID detection, and stale ID counter detection.
- Updated README, local deployment guide, changelog, and maturity scorecard.

## Local Data Repair Performed

The new checks exposed a real local index issue:

- SQLite had `10099` chunk rows but only `10091` unique Qdrant IDs.
- The duplicate IDs were `29120` through `29127`.
- The duplicates were shared by:
  - `/Users/hughlin/Projects/docflow/AGENTS.md`
  - `/Users/hughlin/MyNotes/HughLin/Notes/plans/eduspire/eduspire-plan-v2.md`
- `qdrant_id_counter.txt` was `29139`, while SQLite already had IDs up to
  `29272`.

Repair completed:

```bash
# Set qdrant_id_counter.txt to 29273 before repair.
.venv/bin/python main.py ingest /Users/hughlin/Projects/docflow/AGENTS.md
# AGENTS.md -> 8 chunks indexed

.venv/bin/python main.py ingest /Users/hughlin/MyNotes/HughLin/Notes/plans/eduspire/eduspire-plan-v2.md
# eduspire-plan-v2.md -> 64 chunks indexed
```

After repair:

```bash
cat qdrant_id_counter.txt
# 29345

sqlite3 docflow.db "SELECT MAX(qdrant_id), COUNT(*), COUNT(DISTINCT qdrant_id) FROM chunks;"
# 29344|10099|10099
```

`docflow.db` and `qdrant_id_counter.txt` are ignored runtime files, so this
repair affects the local running data but is not part of the git diff.

## Changed Files

- `CHANGELOG.md`
- `README.md`
- `docs/LOCAL_DEPLOYMENT.md`
- `docs/phase22-handoff.md`
- `eval/phase11_maturity_dimensions.json`
- `main.py`
- `scripts/run_restore_drill.py`
- `src/maintenance/backup.py`
- `src/maintenance/consistency.py`
- `tests/test_backup.py`
- `tests/test_consistency.py`

## Validation Results

Targeted restore and consistency tests:

```bash
.venv/bin/python -m pytest tests/test_backup.py tests/test_consistency.py
# 14 passed, 5 warnings
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
# restore_plan_status=ok
```

Live consistency:

```bash
.venv/bin/python main.py check --json
# status=ok
# sqlite_chunks=10099
# qdrant_points=10099
# duplicate_qdrant_ids=[]
# file_chunk_mismatches=[]
# missing_source_files=[]
# id_counter.status=ok
```

Full tests:

```bash
.venv/bin/python -m pytest
# 188 passed, 5 warnings
```

Maturity score:

```bash
.venv/bin/python main.py maturity-eval --skip-retrieval --json
# overall_score=8.66/9.0
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

- The restore drill is intentionally non-destructive. It does not overwrite the
  live project or perform a full restored Qdrant rebuild.
- The backup archive does not include `qdrant_id_counter.txt`; restore guidance
  still expects `rebuild --qdrant-only` to recreate Qdrant and advance the
  counter. The new checks verify the live counter is safe before handoff.
- The observed duplicate-ID issue came from the counter being behind existing
  SQLite IDs. Detection and local repair are now covered, but ID allocation is
  not yet protected by a lock.
- The drill does not yet simulate corrupt archives, permission failures, or
  disk-full restore failures.

## Next Phase Tasks

Phase 23 should harden ID allocation before moving on to packaging:

1. Make Qdrant ID allocation resistant to concurrent ingests. Use a file lock,
   SQLite-backed counter, or another single-writer mechanism.
2. On startup or before ingest, advance the counter to at least
   `max(SQLite qdrant_id, Qdrant point id) + 1`.
3. Add a safe repair command for stale counters and duplicate Qdrant IDs, so the
   manual repair performed in Phase 22 becomes repeatable.
4. Re-run restore drill, consistency check, full tests, and browser smoke after
   the ID allocation hardening.
5. After ID allocation is hardened, continue the prior packaging/local installer
   work.
