# DocFlow Local Deployment Guide

Date: 2026-05-06

This guide is for running DocFlow as a long-lived personal local knowledge assistant on an Apple Silicon Mac.

## What Stays Local

- Source files stay in watched folders listed in `config.yaml`.
- SQLite data stays at `~/Projects/docflow/docflow.db` by default.
- Qdrant vectors stay in the local Qdrant collection configured in `config.yaml`.
- Default answer generation uses the local MLX model from `llm.mlx_model`.
- Optional OCR uses local Ollama with `glm-ocr`.
- Optional image understanding uses the local VLM cache named in `vlm.model`.

Visible external network use:

- Python and Node dependencies are downloaded during setup.
- Hugging Face and Ollama are contacted when pulling model files.
- The browser UI serves styles and icons locally. It should not request Tailwind, icon fonts, or Google Fonts during normal page use.

## First Run

```bash
cd ~/Projects/docflow
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
docker run -d --name qdrant -p 6333:6333 qdrant/qdrant
python main.py doctor
python main.py start
```

Open `http://localhost:8000`.

## Daily Use

```bash
cd ~/Projects/docflow
source .venv/bin/activate
python main.py service status
python main.py check --json
python main.py repair-ids --dry-run
```

If the service is not installed:

```bash
python main.py service install --dry-run
python main.py service install
```

## Rebuild Browser CSS

The committed `frontend/styles.css` is enough to run the app. Rebuild it only after changing Tailwind classes or theme tokens.

```bash
npm install
npm run build:css
```

## Model Preparation

Check current model status in Settings or with:

```bash
python main.py doctor --json
```

For scanned PDF OCR:

```bash
ollama pull glm-ocr
```

For image understanding, make sure the configured Hugging Face model is cached locally:

```bash
python main.py start --check-only
```

## Backup and Restore Rehearsal

Create a backup:

```bash
python main.py backup --output backups --keep 5
```

Inspect restore steps without changing local files:

```bash
python main.py restore-plan backups/docflow-backup-YYYYmmdd-HHMMSS-ffffff.tar.gz
```

Run a disposable restore drill without changing live files:

```bash
python main.py restore-drill
```

The drill checks the backup archive, extracted SQLite, chunk export, duplicate
vector IDs, ID counter safety, source paths, and restore-plan readiness.

Manual restore outline:

1. Stop DocFlow.
2. Extract the archive into a temporary folder.
3. Back up the current `config.yaml` and `docflow.db`.
4. Copy restored `config.yaml` and `docflow.db` into place.
5. Run `python main.py rebuild --qdrant-only`.
6. Run `python main.py check --json`.

## Troubleshooting

Qdrant is unavailable:

```bash
docker start qdrant
python main.py check --json
```

Port 8000 is busy:

```bash
lsof -nP -iTCP:8000 -sTCP:LISTEN
python main.py start --check-only
```

Ollama OCR is unavailable:

```bash
open -a Ollama
ollama pull glm-ocr
```

Index counts do not match:

```bash
python main.py check --json
python main.py repair-ids --dry-run
python main.py rebuild --qdrant-only --dry-run
```

The browser page looks unstyled after frontend edits:

```bash
npm run build:css
python main.py start
```

## Release Checklist

Before handing off a release-like change:

```bash
npm run build:css
.venv/bin/python -m pytest
.venv/bin/python main.py sample-suite
.venv/bin/python main.py restore-drill
.venv/bin/python main.py check --json
.venv/bin/python main.py repair-ids --dry-run
.venv/bin/python main.py maturity-eval --no-rerank --refresh-sources --source-filter
git diff --check
```

Use `.venv/bin/python main.py maturity-eval --skip-retrieval` only for a
scorecard-only check when Qdrant or the local retrieval index is unavailable.

For UI changes, also run a browser validation pass against `http://127.0.0.1:8000`.
