# Architecture

DocFlow has four main layers.

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
