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
- Release packaging now has a GHCR Docker image workflow, a Python package artifact workflow, and a Docker Compose image file for the no-build public image path.
- The GHCR image workflow now publishes `ghcr.io/lingengyuan/docflow:edge` from `main` and versioned tags from releases.
- Release candidates now produce package checksums, a release candidate manifest, and release notes template before publishing anything.
- OpenSSF Scorecard now runs as a scheduled and main-branch open-source health baseline.
- Public security docs now include a maintainer threat model and model-license boundary.
- Dependency files now document core, local-model, vision, Apple Silicon MLX, and development layers separately.
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
- A scheduled evaluation workflow runs the full public retrieval eval with Qdrant and model download enabled for that isolated benchmark job, and also runs the larger 10,000-document large-library benchmark outside normal push CI.
- Mypy now covers API schemas, the API runtime access layer, API services, and query modules; API handlers now read runtime state through `src/api/runtime.py` instead of reaching back into `src.api.app_impl`.
- Runtime configuration now has a typed settings loader for core paths, Qdrant, ingest, LLM, query, and privacy settings. The source and Docker config templates share the same answer-quality keys so Docker does not drift into different thresholds.
- Legacy internal quality-baseline tooling is hidden from public help and is not release evidence; public quality claims must come from measured checks.
- The Settings page now mounts from a Vite-built Preact component with an explicit design contract, while the existing browser shell keeps the other desktop pages stable during gradual migration.
- Saved notes, source snippets, generated knowledge outputs, and user-confirmed related documents now write back into the same relationship graph used by the Library and active review panels.
- The active review panel now presents a single knowledge loop across sources, questions, citations, saved notes, relationships, review prompts, and feedback.
- The Notes active-review panel now mounts from the Vite-built Preact component path, reducing the legacy browser script to a compatibility entry point.
- External benchmark tracking now lists BEIR, MTEB, and C-MTEB separately from DocFlow's committed regression sets. BEIR SciFact-lite and NFCorpus-lite subset results are now archived with result artifacts and claim boundaries.
- Faithfulness checks now cover supported claims, uncited claims, fabricated source markers, wrong-source citations, no-evidence answers, partial citations, mismatched pages, conflicting sources, stale sources, multi-citation support, and insufficient-evidence answers.
- A desktop large-library benchmark now records synthetic 10,000-document indexing, direct lookup, full-text retrieval orchestration, and deterministic answer assembly separately from embedding, vector search, MLX reranking, and live model generation.
- Release hardening now pins GitHub Actions to commit SHAs, pins the Docker base image and Qdrant service image by digest, tightens workflow token permissions, generates Python package checksums, writes a release candidate manifest, and enables Docker SBOM/provenance output. The public `edge` app image remains a moving no-build smoke tag, and release artifacts are not signed yet.

## Latest Local Validation

- Unit/integration tests: 489 passed.
- Ruff: passed.
- Mypy: passed.
- Browser acceptance: 82 checks passed.
- Public eval: 547/547 passed, Recall@5 0.9982, MRR@5 0.9145, nDCG@5 0.9357, P50 213.92 ms, P95 271.26 ms. This is a committed public-domain regression check, not a broad public benchmark.
- External benchmark: BEIR SciFact-lite subset with 20 questions, Recall@5 0.95, MRR@5 0.95, nDCG@5 0.95, P50 315.66 ms, P95 700.93 ms, 79 indexed documents, no source filtering. Archived subset only; not a full BEIR leaderboard score.
- External benchmark: BEIR NFCorpus-lite subset with 20 questions, Recall@5 0.30, MRR@5 0.50, nDCG@5 0.3246, P50 187.99 ms, P95 412.77 ms, 139 indexed documents, no source filtering, max 5 relevant documents per query. Archived subset only; not a full BEIR leaderboard score.
- External benchmark catalog: valid; 2 archived external scores.
- Retrieval eval: 84/84 passed, Recall@5 1.0, MRR@5 1.0, nDCG@5 1.0, P50 310.27 ms, P95 775.24 ms.
- Parsing eval: 120/120 passed, 147 chunks checked, 26,613 text characters checked.
- Faithfulness eval: 14/14 passed across supported, uncited, fabricated-source, wrong-source, no-evidence, partial-citation, mismatched-page, conflict, stale-source, multi-citation, and insufficient-evidence cases.
- Performance smoke: passed; long note 73,947 bytes, 192 chunks, 3.33 ms total; many-note library 80 files, 80 chunks, 9.73 ms total.
- Large-library benchmark: passed; 10,000 synthetic Markdown documents, 10,000 chunks, all 20 queries returned the expected note in direct lookup, retrieval orchestration, and deterministic answer-path checks. The 20 query targets are spread from note 250 through note 9,750. Indexing took 5,726.40 ms total, or 0.5726 ms per document. Stage P95 timings were 11.75 ms for direct lookup, 13.17 ms for retrieval orchestration, and 14.77 ms for deterministic answer assembly. No smoke thresholds failed. This still does not measure embedding, Qdrant vector search, MLX reranking, live model generation, or first-token latency.
- Release surface check: passed.
- Offline doctor: 0 unexpected outbound connections across startup, ingest, query, model status, and source preview.
- OpenSSF Scorecard: latest reviewed pre-Phase115 baseline was 4.5/10 on commit `672b4e0` (2026-05-17). This is not a mature open-source security score. Phase115 addressed workflow token permissions, GitHub Action pins, Docker base/Qdrant service image pins, package checksums, and Docker SBOM/provenance preparation; branch protection, enforced review policy, signed releases, PyPI publishing, fuzzing, contributor diversity, CII Best Practices, and hash-pinned Python installs remain open.

## Remaining Gaps

- The offline doctor now covers local use paths, but user-triggered webpage import and configured cloud model backends still need explicit user review because they are intentionally external.
- Citation source opening carries chunk identity and span metadata, and source preview highlights the cited range when the matching chunk is available.
- The answer-level source check verifies citation coverage and source-content overlap, not deep semantic truth. A broader factuality benchmark is still needed before treating it as full answer-grounding proof.
- Parser/chunker performance smoke, parsing eval, faithfulness eval, and a thresholded 200-document large-library smoke are now in the standard GitHub CI path. The scheduled workflow covers the larger 10,000-document synthetic path. Embedding, Qdrant vector search, MLX reranking, and live model-answer latency are still not part of every pull-request CI run.
- Retrieval eval currently uses source filtering for project regression checks; do not present it as an external benchmark.
- Public eval is still a committed regression set. It now has a scheduled GitHub workflow, which improves repeatability, and the archived BEIR SciFact-lite and NFCorpus-lite results give two external subset checks. The NFCorpus-lite result also exposes a weak external medical short-query score. A full BEIR, MTEB, C-MTEB, or domain-specific benchmark is still needed before making broad external quality claims.
- API route handlers and retrieval orchestration are still larger than ideal. Storage is now split, but the app layer still needs more handler/service extraction before outside contributors will find it easy to review.
- Open-source security posture is not mature yet. The latest reviewed OpenSSF Scorecard baseline is 4.5/10. Repository-setting gaps still require GitHub configuration or project maturity outside this commit: branch protection, enforced code review, CI-on-PR history, contributor diversity, signed releases, PyPI publishing, fuzzing, CII Best Practices, and hash-pinned Python installation commands.
- DocFlow is not published to PyPI yet. Source checkout and GHCR image startup remain the recommended public install paths; wheel artifacts are built and smoke-tested for releases, but PyPI publishing is not enabled.

## Status Update Rule

Update this page only from measured command output. When validation numbers change, update this page and the README verification lines in the same commit. Use `docs/release.md` before tagging a release, and do not use subjective maturity scores as quality proof.
