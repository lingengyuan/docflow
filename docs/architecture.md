# Architecture

DocFlow has four main layers.

## API

`src/api/` serves the FastAPI browser app, API routes, streaming responses, model task state, and runtime status.

## Ingest

`src/ingest/` parses files, chunks content, writes metadata, creates embeddings, and keeps indexes in sync with watched folders.

## Query

`src/query/` handles retrieval, reranking, scoped search, evidence selection, and answer generation.

## Storage

DocFlow uses SQLite for local metadata and history. Qdrant stores vectors. Runtime files such as databases, caches, backups, and Qdrant storage are local data and should not be committed.

## Browser UI

The browser UI lives in `frontend/`. It is intentionally a local product interface, not a developer console. Normal UI pages must not expose shell commands, script names, maintenance commands, or recovery instructions.
