# Phase 5 Handoff: Consistency Check and Rebuild

Date: 2026-05-02

## Status

Phase 5 is implemented for optimization plan section 2.9: consistency check and rebuild.

Completed scope:

- Added `python main.py check`.
- Added `python main.py check --json`.
- Added `python main.py rebuild --dry-run`.
- Added `python main.py rebuild`.
- Added `python main.py rebuild --qdrant-only --dry-run`.
- Added `python main.py rebuild --qdrant-only`.
- Added SQLite/Qdrant mismatch detection.
- Added missing Qdrant point detection for SQLite chunks.
- Added orphan Qdrant point detection.
- Added file chunk-count mismatch detection.
- Added missing source-file detection.
- Added qdrant-only rebuild from SQLite-stored `raw_text` / `embedding_text`.
- Added full rebuild from configured watch directories.
- Updated README command references.

## Files Changed

- `main.py`
  - Added `check` command.
  - Added `rebuild` command.
  - Reduced `httpx` log noise so `check --json` emits clean JSON.
- `src/maintenance/consistency.py`
  - New consistency and rebuild module.
- `src/maintenance/__init__.py`
  - New package marker.
- `src/ingest/store.py`
  - Added chunk index listing.
  - Added file chunk-count listing.
  - Added index-clearing helper for full rebuild.
- `tests/test_consistency.py`
  - Added mismatch detection and rebuild dry-run tests.
- `README.md`
  - Added check and rebuild commands in English and Chinese.
- `docs/phase5-handoff.md`
  - This handoff.

## Commands

Check current consistency:

```bash
.venv/bin/python main.py check
.venv/bin/python main.py check --json
```

Plan rebuilds without changing data:

```bash
.venv/bin/python main.py rebuild --dry-run
.venv/bin/python main.py rebuild --qdrant-only --dry-run
```

Execute rebuilds:

```bash
.venv/bin/python main.py rebuild
.venv/bin/python main.py rebuild --qdrant-only
```

Important behavior:

- `check` exits with code 0 when consistent and 1 when inconsistent.
- `rebuild --dry-run` only lists planned source files.
- `rebuild` clears the indexed files/chunks and recreates Qdrant from configured watch directories.
- `rebuild --qdrant-only` recreates Qdrant from SQLite chunk rows and preserves SQLite records.

## Validation Results

Commands that passed:

```bash
.venv/bin/python -m pytest
```

Result:

- 93 passed
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

Dry-run qdrant-only rebuild:

```bash
.venv/bin/python main.py rebuild --qdrant-only --dry-run
```

Result:

- `status=dry_run`
- `mode=qdrant_only`
- `chunks=4765`
- `min_qdrant_id=0`
- `max_qdrant_id=7740`

Full rebuild dry-run:

```bash
.venv/bin/python main.py rebuild --dry-run
```

Result:

- `status=dry_run`
- `mode=full`
- `files=259`

Live consistency check:

```bash
.venv/bin/python main.py check
```

Result:

- `status: inconsistent`
- `sqlite_chunks: 4765`
- `qdrant_points: 5172`
- `missing_qdrant_points: 0`
- `orphan_qdrant_points: 407`
- `file_chunk_mismatches: 0`
- `missing_source_files: 20`

Interpretation:

- SQLite does not reference every point currently in Qdrant.
- The live Qdrant collection has 407 orphan points.
- 20 source files tracked in SQLite no longer exist on disk.
- I did not run a destructive rebuild automatically. Use `rebuild --qdrant-only` to remove orphan Qdrant points while preserving SQLite, or use full `rebuild` to rebuild from currently available source files.

## Known Limitations

- Full rebuild can be slow because it re-parses and re-embeds all files in configured watch directories.
- Full rebuild clears indexed file/chunk metadata but preserves query history and embedding cache.
- `rebuild --qdrant-only` depends on Phase 2+ SQLite fields (`raw_text`, `embedding_text`, parent context). Very old rows without stored text cannot be faithfully rebuilt.
- The live check currently reports inconsistency in the existing local corpus. That is expected until a rebuild or cleanup is run.
- Rebuild commands are intentionally explicit and are not run automatically by `check`.

## Next Phase

Start with optimization plan section 2.10: backup and export.

Exact next tasks:

1. Add export of SQLite, `config.yaml`, and needed metadata into an archive.
2. Add chunk export as JSONL.
3. Add restore or restore-plan support that can trigger qdrant-only rebuild after import.
4. Add retention for automatic backups, keeping the newest N archives.
5. Add tests for archive contents, JSONL chunk export, and retention behavior.
6. Run `.venv/bin/python -m pytest`, `git diff --check`, and a real export dry-run before writing the next phase handoff.
