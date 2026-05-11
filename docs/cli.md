# CLI

DocFlow currently exposes commands through `python main.py`.

## Daily Commands

```bash
python main.py serve
python main.py doctor
python main.py scan
python main.py ingest /path/to/file.pdf
python main.py eval
```

## Maintenance Commands

```bash
python main.py check
python main.py rebuild --dry-run
python main.py repair-ids --dry-run
python main.py backup --dry-run
python main.py restore-plan backups/docflow-backup.tar.gz
python main.py restore-drill
```

## Browser Verification

```bash
python main.py browser-acceptance
```

The roadmap includes replacing the flat command list with grouped `docflow` commands.
