# Changelog

## Unreleased

- Moved internal critique, improvement-planning, and scoring notes out of the public docs surface.
- Added an Architecture Decision Record directory (`docs/adr/`) seeded with module-boundary, local-first/no-telemetry, and third-party integration scope ADRs.
- Added PEP 621 `[project.optional-dependencies]` for `dev`, `vision`, and `mlx`, enabling `pip install "docflow[dev]"` while keeping the existing `requirements-*.txt` files as the canonical CI path.
- Linked the ADR index and public roadmap from the README and `docs/architecture.md`.
- Added release packaging automation for GHCR Docker images and Python package artifacts.
- Added a Docker image Compose file for tagged releases.
- Clarified installation costs, upgrade boundaries, and failure modes in public docs.
- Added the public-domain retrieval smoke evaluation path and public/internal evaluation separation.
- Added answer feedback and saved-answer backlinks to close more of the personal knowledge loop.
- Raised vulnerable dependency pins and added Python/frontend dependency audit coverage.
- Added frontend script checks so browser UI changes have a quick syntax gate.
- Grouped maintenance and contributor-only commands under `docflow admin ...` and `docflow dev ...`, leaving the public help focused on daily use.
- Added an archived BEIR SciFact-lite external retrieval subset, answer faithfulness checks, and a synthetic 10,000-document local lookup benchmark.

## 0.58.0 - 2026-05-12

- Added Dependabot and CodeQL coverage for dependency and security maintenance.
- Added feature and question issue templates, strengthened the pull request checklist, and documented the release process.
- Updated the project version and status rules so README and status validation numbers stay tied to real checks.
- Declared the multipart upload and tokenizer dependencies required by upload and retrieval routes.

## 0.57.0 - 2026-05-12

- Added a Library knowledge view derived from indexed local content.
- Added topic groups, similar-document suggestions, and reusable knowledge cards to the Library context panel.
- Kept source review state stable across Library refreshes so citation snippets do not disappear during list updates.

## 0.56.0 - 2026-05-12

- Reworked normal browser UI wording for model choices, default collections, queue stages, and source labels.
- Hid full watched-folder paths from the Settings page by default.
- Added static UI copy coverage to prevent technical model labels and internal English collection names from returning.

## 0.55.0 - 2026-05-12

- Expanded the offline doctor check across startup, local ingest, query fallback, model status, and source preview.
- Added a central runtime network registry for local services, user webpage imports, model downloads, and cloud model backends.
- Blocked silent Hugging Face style model downloads when `privacy.allow_model_download` is false.
- Added a plain Settings notice when a cloud answer backend is active.

## 0.54.0 - 2026-05-12

- Expanded retrieval evaluation from 8 to 84 committed cases.
- Expanded parsing regression from 2 to 31 committed files across Markdown, TXT, code-like files, PDF, and DOCX fixtures.
- Added retrieval latency summaries, parsing performance summaries, and an incremental add/modify/delete indexing regression test.
- Tightened source-filtered evals so same-named files in other local folders do not pollute project benchmark results.

## 0.53.0 - 2026-05-12

- Added exact source-range highlighting from citation metadata in the source preview.
- Marked inline model citations that do not match retrieved evidence as unverified.
- Expanded citation alignment tests to cover verified source cases and fabricated inline citations.

## 0.52.0 - 2026-05-12

- Split the public API, storage, and retrieval entry points into small facade modules.
- Moved the existing implementations behind stable import paths to reduce public file size.
- Added a structure test to prevent the public entry points from growing back into large files.

## 0.51.0 - 2026-05-12

- Made full `ruff check .` pass across the repository.
- Made the configured `mypy` check pass.
- Updated CI to run full ruff, mypy, and pytest instead of only a small ruff safety subset.

## 0.50.0 - 2026-05-12

- Added a full Docker Compose app service alongside Qdrant for one-command startup.
- Added a demo library command and browser entry point for first-run users.
- Improved empty-library guidance with real actions for demo data, uploads, and folder scanning.

## 0.49.0 - 2026-05-12

- Upgraded the public README with a clearer first screen, verification baseline, privacy promise, and contributor links.
- Added public contribution, security, code of conduct, and roadmap documents.
- Kept the public docs surface focused while leaving internal planning notes out of GitHub.

## 0.29.0 - 2026-05-08

- Added final checked-in screenshots for Chat, Library, Notes, and Settings.
- Closed the current UI redesign plan with browser acceptance evidence.
- Documented that remaining maturity gaps move to the post-review roadmap after the UI plan.

## 0.28.0 - 2026-05-08

- Reworked Settings into a normal user-facing page with status hints, local model state, watched folders, and daily-use preferences.
- Removed command-line recovery, repair, dry-run, and copy-command wording from the normal browser UI.
- Added browser acceptance and static checks to prevent developer-only wording from reappearing in Settings.

## 0.27.0 - 2026-05-08

- Added real Library groups for all files, favorites, recent imports, PDFs, Markdown, images, and code.
- Upgraded Library file details with source chunk review, recent citation history, open-original actions, save-as-note, and file-scoped question shortcuts.
- Extended browser acceptance to click Library groups and verify source review states.

## 0.26.0 - 2026-05-07

- Redesigned the browser UI into a quieter personal knowledge workspace with a wider sidebar, global search, unified toolbars, and right-side context panels.
- Added real context surfaces for chat citations, Library file details, Notes recent captures, and Settings recovery guidance.
- Updated browser acceptance checks and README screenshots for the workspace.

## 0.25.0 - 2026-05-07

- Added `docflow browser-acceptance` for repeatable Chromium checks across Chat, Library, Notes, and Settings.
- Added screenshot artifacts for browser acceptance.
- Documented browser acceptance in README and local deployment guidance.

## 0.24.0 - 2026-05-07

- Added `docflow install-local` and `scripts/install_local.sh` as a safe local install plan.
- Kept local install dry-run by default; use `--apply` to execute the plan.
- Surfaced install, restore drill, and vector ID repair commands in Settings maintenance guidance.

## 0.23.0 - 2026-05-07

- Hardened Qdrant ID allocation with an interprocess file lock and live index floor checks.
- Added `docflow repair-ids --dry-run` to inspect stale ID counters and duplicate vector IDs before repair.
- Extended consistency checks so the ID counter floor uses both SQLite and Qdrant point IDs.

## 0.22.0 - 2026-05-07

- Added `docflow restore-drill` for disposable backup recovery rehearsal.
- Validated restored SQLite integrity, manifest counts, exported chunks, source paths,
  chunk count metadata, duplicate vector IDs, ID counter safety, and restore-plan readiness.

## 0.21.0 - 2026-05-07

- Added `docflow sample-suite` for generated real sample validation.
- Covered deterministic scanned PDF OCR, VLM image parsing, table chunking,
  source preview, and knowledge output workflows.

## 0.20.0 - 2026-05-07

- Replaced the runtime Google Fonts Material Symbols request with local inline SVG icons.
- Added static coverage to prevent remote icon font regressions.
- Updated local deployment guidance to reflect fully local browser UI assets.

## 0.18.0 - 2026-05-06

- Added local production CSS build assets and removed the runtime Tailwind CDN dependency.
- Added final local deployment guidance, troubleshooting notes, and acceptance reporting.
- Added the standalone MIT `LICENSE`.
- Added a validation path for backup and restore-plan rehearsal.

## 0.17.0 - 2026-05-06

- Added reusable knowledge output workflows for summaries, learning cards, action items, and project briefs.
- Added `/api/knowledge-output` and saved generated outputs into `Knowledge Outputs`.
- Added Notes and Library UI entry points for selected-file knowledge outputs.

## 0.16.0 - 2026-05-06

- Added grouped runtime health, model readiness, optional capability status, and copyable recovery guidance.
- Added clearer Ollama, OCR, VLM, and local model cache status in Settings.

## 0.15.0 - 2026-05-06

- Reorganized the browser UI into Chat, Library, Notes, and Settings.
- Added dedicated Notes and Settings workspaces.

## 0.11.0 - 2026-05-04

- Added the rolling 9-point maturity scorecard and fixed retrieval evidence checks.
