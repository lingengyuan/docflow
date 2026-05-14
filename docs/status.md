# Project Status

DocFlow is now beyond the original prototype phase and has a clearer public project surface.

## Current State

- README is concise and points to focused public docs.
- New users can start from `config.example.yaml`, `docker-compose.yml`, and the `docflow` command.
- Docker Compose now defines both the DocFlow app and Qdrant service for first-run startup.
- First-run users can create a small demo library from the CLI or browser empty state.
- Runtime dependencies are smaller and optional image support is split out.
- CI, CodeQL, Dependabot, dependency audit, ruff, pre-commit, issue templates, and PR template are present.
- CI now runs full ruff, mypy, and pytest on Ubuntu and macOS.
- Local privacy has an offline doctor check covering startup, local ingest, query fallback, model status, and source preview.
- Model downloads are blocked by default when the configured cache is missing and `privacy.allow_model_download` is false.
- Citations include chunk identity and source span metadata.
- Source preview can highlight the cited source range when citation span metadata is available.
- Browser UI now presents model choices, collections, watched folders, queue stages, and source labels in user-facing language.
- The Library page now derives topic views, similar documents, and knowledge cards from indexed local content.
- Local answer generation uses deterministic defaults.
- Retrieval evaluation now covers 84 committed questions and reports Recall@5, MRR@5, nDCG@5, pass rate, and latency summary.
- Public retrieval smoke evaluation now covers 6 committed public-domain cases without source filtering. It is reproducible from `eval/public_corpus/`, but it is intentionally not a BEIR, MTEB, or C-MTEB score.
- Parsing regression now covers 31 committed files across Markdown, TXT, code-like text, PDF, and DOCX fixtures.
- Incremental indexing has a regression test for add, modify, and delete behavior.
- Release guidance now covers validation, status updates, tagging, release notes, screenshots, and known limitations.
- Release packaging now has a GHCR Docker image workflow, a Python package artifact workflow, and a Docker Compose image file for tagged releases.
- Storage is split into focused database, file, vector, history, and library metadata modules. Retrieval routing and MLX reranking now live outside the main retriever implementation. API health checks now live outside the main app implementation.
- Runtime dependencies now keep Apple Silicon MLX support in an optional requirements file, and code hygiene tests prevent silent broad exception handlers and non-maintenance print calls from creeping back into `src/`.
- The latest dependency review raised `python-multipart`, `pillow`, `onnx`, `pytest`, `mlx`, and `mlx-lm` above the current Dependabot fixed versions. Local `pip-audit` and `npm audit` both report no known vulnerabilities.
- CI now includes Ubuntu Python 3.11/3.12, macOS Python 3.12, Windows Python 3.12, and a dedicated offline doctor job.
- The browser shell now has a language toggle foundation, keyboard skip link, active navigation state, and PWA shell files.
- Internal planning notes now live outside the public project surface; the repository no longer ships a public `plans/` directory.

## Latest Local Validation

- Unit/integration tests: 361 passed.
- Ruff: passed.
- Mypy: passed.
- Browser acceptance: 79 checks passed.
- Public eval: 6/6 passed, Recall@5 1.0, MRR@5 1.0, nDCG@5 1.0, P50 803.56 ms, P95 4898.86 ms. This is a small public-domain smoke check, not a broad public benchmark.
- Retrieval eval: 84/84 passed, Recall@5 1.0, MRR@5 1.0, nDCG@5 1.0, P50 300.18 ms, P95 753.75 ms.
- Parsing eval: 31/31 passed, 42 chunks checked, 11,351 text characters checked.
- Offline doctor: 0 unexpected outbound connections across startup, ingest, query, model status, and source preview.

## Remaining Gaps

- The offline doctor now covers local use paths, but user-triggered webpage import and configured cloud model backends still need explicit user review because they are intentionally external.
- Citation source opening carries chunk identity and span metadata, and source preview highlights the cited range when the matching chunk is available.
- Large-file and large-library benchmarks are not yet part of the standard CI path.
- Retrieval eval currently uses source filtering for project regression checks; do not present it as an external benchmark.
- Public eval is currently a small smoke set. It improves reproducibility, but a broad BEIR, MTEB, C-MTEB, or domain-specific benchmark is still needed before making external quality claims.
- API route handlers and retrieval orchestration are still larger than ideal. Storage is now split, but the app layer still needs more handler/service extraction before outside contributors will find it easy to review.
- DocFlow is not published to PyPI yet. The supported public install paths are source checkout and Docker; tagged Docker images are prepared through GHCR release automation.

## Status Update Rule

Update this page only from measured command output. When validation numbers change, update this page and the README verification lines in the same commit. Use `docs/release.md` before tagging a release, and do not use subjective maturity scores as quality proof.
