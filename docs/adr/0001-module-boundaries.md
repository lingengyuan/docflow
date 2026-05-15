# ADR 0001 — Module boundaries: ingest / storage / retrieval / api / ui kept orthogonal

- **Status**: Accepted
- **Date**: 2026-05-15
- **Deciders**: maintainers

## Context

DocFlow has five concerns that historically tend to bleed into each other in RAG codebases:

1. **Ingest** — parse files, chunk, normalize.
2. **Storage** — SQLite for metadata, Qdrant for vectors.
3. **Retrieval** — query rewrite, hybrid search (BM25 + dense), rerank, citation building.
4. **API** — FastAPI HTTP surface and streaming.
5. **UI** — browser frontend.

Mixing them produces the typical RAG bug pattern: a UI change forces a retrieval change, a retrieval change forces a parsing change, and tests can't isolate any layer.

The project rule "Orthogonality: keep ingestion, storage, retrieval, API, and UI separate" (see `AGENTS.md`) is the long-standing intent.

## Decision

The repository layout enforces this split:

- `src/ingest/` — only ingestion, parsing, chunking, watched-folder logic.
- `src/vector_store.py` + SQLite glue under `src/api/services/` — storage adapters.
- `src/query/` — retrieval and answer composition only. May call storage adapters; **must not** call ingest internals.
- `src/api/` — FastAPI app, routes, schemas, runtime. May orchestrate query & ingest; **must not** contain retrieval logic.
- `frontend/` — UI. Communicates with backend only via documented HTTP endpoints.

Cross-cutting helpers (`src/domain_types.py`, `src/net.py`, `src/resources.py`, `src/model_cache.py`) are allowed but must be **stateless** with respect to the five layers above.

## Consequences

**Positive**
- Each layer testable in isolation. The current 55-file test suite reflects this.
- New contributors can locate "where does X live" without grepping the whole repo.
- Refactors stay local (e.g., switching reranker shouldn't touch ingest).

**Negative**
- Sometimes forces an extra adapter when a quick cross-call would do.
- Larger PRs when a feature genuinely spans multiple layers (acceptable cost).

## Compliance check

Any PR that introduces a new import edge across these boundaries must justify it in the PR description, or refactor to keep the edge inside the appropriate layer. There is no automated linter for this today; adding one belongs in the public [ROADMAP](../../ROADMAP.md) or the relevant issue before implementation.
