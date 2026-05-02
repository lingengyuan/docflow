# Phase 6 Handoff: Backup and Export

Date: 2026-05-02

## Status

Phase 6 is implemented for optimization plan section 2.10: backup and export.

Completed scope:

- Added `python main.py backup`.
- Added `python main.py backup --dry-run`.
- Added `python main.py backup --output backups --keep 5`.
- Added `python main.py export-chunks --output backups/chunks.jsonl`.
- Added `python main.py restore-plan <backup.tar.gz>`.
- Added archive export for `config.yaml`, a SQLite database snapshot, chunk JSONL, and a manifest.
- Added chunk export as JSONL.
- Added backup retention that keeps the newest N backup archives.
- Added a non-destructive restore plan that points to `rebuild --qdrant-only` and `check`.
- Updated README command references in English and Chinese.
- Ignored generated backup files from git.

## Files Changed

- `main.py`
  - Added backup, chunk export, and restore-plan commands.
  - Added lightweight command argument parsing for the new maintenance commands.
- `src/maintenance/backup.py`
  - New backup and export module.
  - Creates archive files with `manifest.json`, `config.yaml`, `docflow.db`, and `chunks.jsonl`.
  - Exports chunks as standalone JSONL.
  - Reads backup archives and returns restore steps without modifying local files.
  - Applies retention to old backup archives.
- `tests/test_backup.py`
  - Added JSONL export tests.
  - Added backup archive content tests.
  - Added restore-plan tests.
  - Added dry-run and retention tests.
- `.gitignore`
  - Added local backup output ignores.
- `README.md`
  - Added backup, export, and restore-plan commands in English and Chinese.
- `docs/phase6-handoff.md`
  - This handoff.

## Commands

Preview backup without writing an archive:

```bash
.venv/bin/python main.py backup --dry-run --output backups --keep 3
```

Create a backup archive:

```bash
.venv/bin/python main.py backup --output backups --keep 3
```

Export chunks only:

```bash
.venv/bin/python main.py export-chunks --output /tmp/docflow-phase6-chunks.jsonl
```

Inspect restore steps:

```bash
.venv/bin/python main.py restore-plan backups/docflow-backup-20260502-143904-821331.tar.gz
```

Important behavior:

- `backup --dry-run` reports what would be included and does not write an archive.
- `backup` writes a local archive under the chosen output directory.
- Backup archives include a SQLite snapshot, not a raw live file copy.
- `export-chunks` writes one chunk per JSONL line.
- `restore-plan` only reads the archive and prints recovery steps; it does not overwrite local files.
- Backup retention applies only to files named `docflow-backup-*.tar.gz` in the selected backup directory.

## Validation Results

Commands that passed:

```bash
.venv/bin/python -m pytest tests/test_backup.py
```

Result:

- 5 passed

```bash
.venv/bin/python -m pytest
```

Result:

- 98 passed
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

Backup dry-run:

```bash
.venv/bin/python main.py backup --dry-run --output backups --keep 3
```

Result:

- `status=dry_run`
- `would_include=manifest.json, config.yaml, docflow.db, chunks.jsonl`
- `files=162`
- `chunks=4765`
- `keep=3`

Real backup:

```bash
.venv/bin/python main.py backup --output backups --keep 3
```

Result:

- `status=done`
- `path=backups/docflow-backup-20260502-143904-821331.tar.gz`
- `files=162`
- `chunks=4765`
- `deleted=[]`

Chunk export:

```bash
.venv/bin/python main.py export-chunks --output /tmp/docflow-phase6-chunks.jsonl
wc -l /tmp/docflow-phase6-chunks.jsonl
```

Result:

- `status=done`
- `chunks=4765`
- Exported file line count: 4765

Restore-plan check:

```bash
.venv/bin/python main.py restore-plan backups/docflow-backup-20260502-143904-821331.tar.gz
```

Result:

- `status=ok`
- Archive members: `chunks.jsonl`, `config.yaml`, `docflow.db`, `manifest.json`
- Manifest file count: 162
- Manifest chunk count: 4765
- Restore steps include `rebuild --qdrant-only` and `check`

## Known Limitations

- Restore is intentionally manual for now. The command gives exact steps but does not overwrite local files.
- Backup archives do not include Qdrant vector storage. After restoring SQLite and config, run `rebuild --qdrant-only`.
- Backup archives do not include the original watched source files.
- Retention only manages backup archives in the selected output directory.
- The current live corpus still has the Phase 5 consistency issue until a rebuild or cleanup is run.

## Next Phase

Start with optimization plan section 2.11: multi-turn conversations.

Exact next tasks:

1. Add conversation and message persistence.
2. Add commands or API behavior to create, switch, and delete conversations.
3. Make the newest user question drive retrieval.
4. Pass recent conversation turns into answer generation as context.
5. Add follow-up query rewriting for questions like "expand the second point".
6. Ensure conversations survive app restart.
7. Add tests for persistence, switching/deleting conversations, and follow-up behavior.
8. Run `.venv/bin/python -m pytest`, relevant app flow checks, README parity check, `git diff --check`, and write the next phase handoff before reporting completion.
