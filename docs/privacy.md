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

DocFlow keeps a small registry of runtime network cases:

- Local services: Qdrant, Ollama, and the DocFlow web app on `localhost` or loopback addresses.
- User-triggered webpage import: only when the user explicitly imports a URL.
- Model downloads: blocked by default when a configured Hugging Face style model is not already cached.
- Cloud LLM backends: off by default and only active when explicitly configured.

Run the offline doctor to check for unexpected outbound connections across startup, local ingest, local query fallback, model status checks, and source preview:

```bash
docflow doctor --offline
```

Expected result:

```text
DocFlow offline network check: ok
0 unexpected outbound connections
Covered local paths: startup, ingest, query, model status, source preview
Registered network cases: local_services, user_web_import, model_download, cloud_llm
```

User-triggered webpage import is the main intentionally external runtime feature. Optional model downloads require `privacy.allow_model_download: true`. Cloud model backends require an explicit backend and key. When a cloud answer backend is active, the Settings page shows a plain-language notice that questions may be sent to the configured external model service.

If `privacy.allow_model_download: false` and a configured local model is missing from cache, DocFlow fails clearly before loading that model instead of silently downloading it.

## User Responsibility

DocFlow creates `config.yaml` from `config.example.yaml` on first run. Review the generated local `config.yaml` before changing model backends, watched folders, or optional cloud settings.
