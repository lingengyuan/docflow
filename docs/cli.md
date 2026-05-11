# CLI

DocFlow exposes commands through `docflow` after `pip install -e .`.

## Daily Commands

```bash
docflow serve
docflow doctor
docflow doctor --offline
docflow scan
docflow ingest /path/to/file.pdf
docflow eval
```

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

`python main.py ...` remains available for local development.
