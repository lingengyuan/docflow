# Project Status

DocFlow is now beyond the original prototype phase and has a clearer public project surface.

## Current State

- README is concise and points to focused public docs.
- New users can start from `config.example.yaml`, `docker-compose.yml`, and the `docflow` command.
- Docker Compose now defines both the DocFlow app and Qdrant service for first-run startup.
- First-run users can create a small demo library from the CLI or browser empty state.
- Runtime dependencies are smaller and optional image support is split out.
- CI, CodeQL, Dependabot, ruff, pre-commit, issue templates, and PR template are present.
- CI now runs full ruff, mypy, and pytest on Ubuntu and macOS.
- Local privacy has an offline doctor check covering startup, local ingest, query fallback, model status, and source preview.
- Model downloads are blocked by default when the configured cache is missing and `privacy.allow_model_download` is false.
- Citations include chunk identity and source span metadata.
- Source preview can highlight the cited source range when citation span metadata is available.
- Browser UI now presents model choices, collections, watched folders, queue stages, and source labels in user-facing language.
- The Library page now derives topic views, similar documents, and knowledge cards from indexed local content.
- Local answer generation uses deterministic defaults.
- Retrieval evaluation now covers 84 committed questions and reports Recall@5, MRR@5, nDCG@5, pass rate, and latency summary.
- Parsing regression now covers 31 committed files across Markdown, TXT, code-like text, PDF, and DOCX fixtures.
- Incremental indexing has a regression test for add, modify, and delete behavior.
- Release guidance now covers validation, status updates, tagging, release notes, screenshots, and known limitations.

## Latest Local Validation

- Unit/integration tests: 304 passed.
- Ruff: passed.
- Mypy: passed.
- Browser acceptance: 73 checks passed.
- Retrieval eval: 84/84 passed, Recall@5 1.0, MRR@5 1.0, nDCG@5 1.0, P50 297.48 ms, P95 793.96 ms.
- Parsing eval: 31/31 passed, 42 chunks checked, 11,351 text characters checked.
- Offline doctor from Phase55: 0 unexpected outbound connections across startup, ingest, query, model status, and source preview.

## Remaining Gaps

- The offline doctor now covers local use paths, but user-triggered webpage import and configured cloud model backends still need explicit user review because they are intentionally external.
- Citation source opening carries chunk identity and span metadata, and source preview highlights the cited range when the matching chunk is available.
- Large-file and large-library benchmarks are not yet part of the standard CI path.
- Retrieval eval currently uses source filtering for project regression checks; a broader unfiltered public benchmark is still needed before external quality claims.
- Public API, storage, and retrieval entry points are now small facades, but their implementation modules still need deeper internal splitting before the codebase feels fully mature to outside contributors.

## Status Update Rule

Update this page only from measured command output. When validation numbers change, update this page and the README verification lines in the same commit. Use `docs/release.md` before tagging a release, and do not use subjective maturity scores as quality proof.
