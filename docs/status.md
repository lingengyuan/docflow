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
- Answer generation now asks models to cite exact chunk identifiers and filters displayed
  source cards down to verified chunks that were actually cited.
- Streaming answers now finalize the displayed answer and saved history against the same
  chunk-level citation validation used by non-streamed answers.
- Answers now include a deterministic sentence-level source check that flags conclusions
  without verified source markers or with cited snippets that do not share meaningful
  content terms in both normal and streaming responses.
- Source preview can highlight the cited source range when citation span metadata is available.
- Browser UI now presents model choices, collections, watched folders, queue stages, and source labels in user-facing language.
- The Library page now derives topic views, similar documents, and knowledge cards from indexed local content.
- The Notes page now surfaces answer-to-source relationship activity in the active review panel.
- The active review panel now includes data-backed knowledge-depth signals: active concepts, question-to-source trails, coverage gaps, and next actions.
- The active review panel now suggests unlinked but related local sources so users can turn similar documents into explicit knowledge relationships.
- Local answer generation uses deterministic defaults.
- Retrieval evaluation now covers 84 committed questions and reports Recall@5, MRR@5, nDCG@5, pass rate, and latency summary.
- Public retrieval regression now covers 547 committed public-domain, United States government, NASA, literature, and civic-history cases without source filtering. It is reproducible from `eval/public_corpus/`, but it is intentionally not a BEIR, MTEB, or C-MTEB score.
- Parsing regression now covers 120 committed files across Markdown, wikilink/frontmatter/callout-style Markdown, TXT, noisy OCR-like text, code-like text, PDF, and DOCX fixtures.
- Performance smoke now covers parser/chunker behavior for a synthetic long note and a synthetic many-note library in the standard local CI script.
- Incremental indexing has a regression test for add, modify, and delete behavior.
- Release guidance now covers validation, status updates, tagging, release notes, screenshots, and known limitations.
- Release packaging now has a GHCR Docker image workflow, a Python package artifact workflow, and a Docker Compose image file for tagged releases.
- Python package artifacts now include browser assets, config templates, and public docs, with an installed-wheel smoke test in the local CI script.
- Storage is split into focused database, file, vector, history, and library metadata modules. Retrieval routing and MLX reranking now live outside the main retriever implementation. API health checks now live outside the main app implementation.
- Runtime dependencies now keep Apple Silicon MLX support in an optional requirements file, and code hygiene tests prevent silent broad exception handlers and non-maintenance print calls from creeping back into `src/`.
- The latest dependency review raised `python-multipart`, `pillow`, `onnx`, `pytest`, `mlx`, and `mlx-lm` above the current Dependabot fixed versions. Local `pip-audit` and `npm audit` both report no known vulnerabilities.
- CI now includes Ubuntu Python 3.11/3.12, macOS Python 3.12, Windows Python 3.12, and a dedicated offline doctor job.
- The browser shell now has a language toggle foundation, keyboard skip link, and active navigation state.
- Browser actions now pass failures through user-facing messages before rendering them, with a static regression check that blocks raw service errors from leaking into normal pages.
- Public screenshots are regenerated from the bundled demo library so they do not expose local personal paths or private notes.
- Internal planning notes now live outside the public project surface; the repository no longer ships a public `plans/` directory.
- Release surface checks now verify public docs, README/status validation counts, Docker Compose files, workflows, package data, and ignored internal handoff/output paths before package smoke testing.
- GitHub CI now runs the release surface check, package smoke test, parser/chunker performance smoke, and parsing eval in addition to ruff, mypy, pytest, frontend checks, dependency audit, and offline doctor.
- A scheduled evaluation workflow runs the full public retrieval eval with Qdrant and model download enabled for that isolated benchmark job.
- Mypy now covers API schemas, the API runtime access layer, API services, and query modules; API handlers now read runtime state through `src/api/runtime.py` instead of reaching back into `src.api.app_impl`.
- Runtime configuration now has a typed settings loader for core paths, Qdrant, ingest, LLM, query, and privacy settings. The source and Docker config templates share the same answer-quality keys so Docker does not drift into different thresholds.
- External benchmark tracking now lists BEIR, MTEB, and C-MTEB separately from DocFlow's committed regression sets. No external benchmark score has been archived yet.

## Latest Local Validation

- Unit/integration tests: 446 passed.
- Ruff: passed.
- Mypy: passed.
- Browser acceptance: 81 checks passed.
- Public eval: 547/547 passed, Recall@5 0.9982, MRR@5 0.9145, nDCG@5 0.9357, P50 213.92 ms, P95 271.26 ms. This is a committed public-domain regression check, not a broad public benchmark.
- External benchmark catalog: valid; 0 archived external scores.
- Retrieval eval: 84/84 passed, Recall@5 1.0, MRR@5 1.0, nDCG@5 1.0, P50 310.27 ms, P95 775.24 ms.
- Parsing eval: 120/120 passed, 147 chunks checked, 26,613 text characters checked.
- Performance smoke: passed; long note 73,947 bytes, 192 chunks, 3.33 ms total; many-note library 80 files, 80 chunks, 9.73 ms total.
- Release surface check: passed.
- Offline doctor: 0 unexpected outbound connections across startup, ingest, query, model status, and source preview.

## Remaining Gaps

- The offline doctor now covers local use paths, but user-triggered webpage import and configured cloud model backends still need explicit user review because they are intentionally external.
- Citation source opening carries chunk identity and span metadata, and source preview highlights the cited range when the matching chunk is available.
- The answer-level source check verifies citation coverage and source-content overlap, not deep semantic truth. A broader factuality benchmark is still needed before treating it as full answer-grounding proof.
- Parser/chunker performance smoke and parsing eval are now in the standard GitHub CI path, but full large-library retrieval, embedding, and model-answer benchmarks are still not part of every pull-request CI run.
- Retrieval eval currently uses source filtering for project regression checks; do not present it as an external benchmark.
- Public eval is still a committed regression set. It now has a scheduled GitHub workflow, which improves repeatability, but a broad BEIR, MTEB, C-MTEB, or domain-specific benchmark is still needed before making external quality claims.
- API route handlers and retrieval orchestration are still larger than ideal. Storage is now split, but the app layer still needs more handler/service extraction before outside contributors will find it easy to review.
- DocFlow is not published to PyPI yet. Source checkout and Docker remain the recommended public install paths; wheel artifacts are built and smoke-tested for releases, but PyPI publishing is not enabled.

## Status Update Rule

Update this page only from measured command output. When validation numbers change, update this page and the README verification lines in the same commit. Use `docs/release.md` before tagging a release, and do not use subjective maturity scores as quality proof.
