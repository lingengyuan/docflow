# Privacy

DocFlow is built as a local-first personal knowledge assistant.

## Default Promise

- No telemetry.
- No analytics.
- No automatic error reporting.
- No upload of documents for product analytics.
- Local documents are stored and indexed on the user's machine.

## Local Services

DocFlow normally talks to local services such as:

- FastAPI app on the local machine.
- SQLite database on disk.
- Qdrant on `localhost`.
- Ollama on `localhost` when OCR or local model features use it.

## Network Access

Current known network-related cases:

- Model downloads may contact model hosting providers when a configured model is not already cached.
- Optional cloud LLM backends require explicit user configuration.
- Development and browser testing tools may download or launch browser dependencies.

The 90-point roadmap adds an auditable offline doctor command so users can verify unexpected outbound connections instead of relying on this document alone.

## User Responsibility

DocFlow creates `config.yaml` from `config.example.yaml` on first run. Review the generated local `config.yaml` before changing model backends, watched folders, or optional cloud settings.
