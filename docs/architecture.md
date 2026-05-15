# Architecture

DocFlow has four main layers. For the *why* behind these boundaries see [ADR 0001](adr/0001-module-boundaries.md); for the privacy / no-fallback contract see [ADR 0002](adr/0002-local-first-no-telemetry.md). Public direction is tracked in the root [ROADMAP](../ROADMAP.md), while measured status lives in [docs/status.md](status.md).

## API

`src/api/` serves the FastAPI browser app, API routes, streaming responses, model task state, and runtime status.
Runtime dependencies are held in one application context. Routes should read shared objects from that context instead of creating parallel module-level state.

## Ingest

`src/ingest/` parses files, chunks content, writes metadata, creates embeddings, and keeps indexes in sync with watched folders.

## Query

`src/query/` handles retrieval, reranking, scoped search, evidence selection, and answer generation.
Query thresholds, answer chunk limits, research step limits, table keywords, and the Qdrant collection name are configuration-backed settings, not scattered constants.

## Storage

DocFlow uses SQLite for local metadata and history. Qdrant stores vectors. Runtime files such as databases, caches, backups, and Qdrant storage are local data and should not be committed.
SQLite schema changes must be handled through store migrations. Config changes must keep old keys readable or provide a documented default so existing local libraries do not require a rebuild without warning.

## Browser UI

The browser UI lives in `frontend/`. It is intentionally a local product interface, not a developer console. Normal UI pages must not expose shell commands, script names, maintenance commands, or recovery instructions.

## Failure Modes

- Qdrant is unavailable: startup and health checks should report the vector store as unavailable instead of silently falling back to stale results.
- Local model server is unavailable: the app should keep the library usable and explain that answering needs a configured local model.
- Model cache is missing while downloads are blocked: startup should show that the model is missing and respect `privacy.allow_model_download: false`.
- SQLite schema is older: migrations should run at open time. If a migration cannot be applied safely, the error should name the database and stop before writing partial state.
- Source file was moved or deleted: source preview should explain the file is missing and avoid showing old text as current evidence.
- Webpage import or cloud model use is external by design: the action must remain explicit and must not be counted as offline behavior.

## Upgrade Boundaries

- Config changes must preserve existing keys or provide documented defaults.
- SQLite schema changes belong in store migrations and need tests against older database shapes.
- Qdrant collection changes must state whether users can keep the existing collection, run a vector-only rebuild, or reindex all files.
- Embedding model changes are not automatically compatible with old vectors; release notes must call out any required reindex.
- Browser asset changes are bundled with the app. Package builds must include `frontend/`, `config.example.yaml`, and public docs required by the app.
