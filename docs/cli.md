# CLI

DocFlow exposes commands through `docflow` after `pip install -e .`.

## Daily Commands

```bash
docflow serve
docflow doctor
docflow doctor --offline
docflow demo
docflow demo --create-only
docflow scan
docflow ingest /path/to/file.pdf
```

`docflow doctor --offline` checks startup, a local ingest probe, query fallback, model status, and source preview for unexpected outbound connections.

## Maintenance Commands

```bash
docflow admin platform --json
docflow admin check
docflow admin rebuild --dry-run
docflow admin repair-ids --dry-run
docflow admin backup --dry-run
docflow admin restore-plan backups/docflow-backup.tar.gz
docflow admin restore-drill
```

## Browser Verification

```bash
docflow dev eval public
docflow dev eval retrieval
docflow dev eval parsing
docflow dev eval performance
docflow dev browser-acceptance
docflow dev dead-code-audit
```

`docflow dev eval public` runs the committed public-domain smoke benchmark without source filtering. `docflow dev eval retrieval` runs the larger internal source-filtered project regression set. `docflow dev eval performance` runs a local parser/chunker smoke check for one long synthetic note and a synthetic many-note library.
