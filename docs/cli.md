# CLI

DocFlow exposes commands through `docflow` after `pip install -e .`.

## Daily Commands

```bash
docflow serve
docflow doctor
docflow doctor --offline
docflow platform --json
docflow demo
docflow demo --create-only
docflow scan
docflow ingest /path/to/file.pdf
docflow eval public
docflow eval retrieval
docflow eval parsing
```

`docflow doctor --offline` checks startup, a local ingest probe, query fallback, model status, and source preview for unexpected outbound connections.

`docflow eval public` runs the small committed public-domain smoke benchmark without source filtering. `docflow eval retrieval` runs the larger internal source-filtered project regression set.

## Maintenance Commands

```bash
docflow check
docflow rebuild --dry-run
docflow repair-ids --dry-run
docflow backup --dry-run
docflow restore-plan backups/docflow-backup.tar.gz
docflow restore-drill
```

## Browser Verification

```bash
docflow browser-acceptance
```
