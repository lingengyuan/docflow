# Changelog

## 0.49.0 - 2026-05-12

- Upgraded the public README with a clearer first screen, verification baseline, privacy promise, and contributor links.
- Added public contribution, security, code of conduct, and roadmap documents.
- Kept the public docs surface focused while leaving internal phase handoffs out of GitHub.

## 0.29.0 - 2026-05-08

- Added final checked-in screenshots for Chat, Library, Notes, and Settings.
- Closed the current UI redesign plan with browser acceptance evidence and a Phase 29 handoff.
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
- Updated browser acceptance checks and README screenshots for the Phase 26 workspace.

## 0.25.0 - 2026-05-07

- Added `python main.py browser-acceptance` for repeatable Chromium checks across Chat, Library, Notes, and Settings.
- Added screenshot artifacts under `output/playwright/phase25-browser-acceptance`.
- Documented browser acceptance in README and local deployment guidance.

## 0.24.0 - 2026-05-07

- Added `python main.py install-local` and `scripts/install_local.sh` as a safe local install plan.
- Kept local install dry-run by default; use `--apply` to execute the plan.
- Surfaced install, restore drill, and vector ID repair commands in Settings maintenance guidance.

## 0.23.0 - 2026-05-07

- Hardened Qdrant ID allocation with an interprocess file lock and live index floor checks.
- Added `python main.py repair-ids --dry-run` to inspect stale ID counters and duplicate vector IDs before repair.
- Extended consistency checks so the ID counter floor uses both SQLite and Qdrant point IDs.

## 0.22.0 - 2026-05-07

- Added `python main.py restore-drill` for disposable backup recovery rehearsal.
- Validated restored SQLite integrity, manifest counts, exported chunks, source paths,
  chunk count metadata, duplicate vector IDs, ID counter safety, and restore-plan readiness.

## 0.21.0 - 2026-05-07

- Added `python main.py sample-suite` for generated real sample validation.
- Covered deterministic scanned PDF OCR, VLM image parsing, table chunking,
  source preview, and knowledge output workflows.

## 0.20.0 - 2026-05-07

- Replaced the runtime Google Fonts Material Symbols request with local inline SVG icons.
- Added static coverage to prevent remote icon font regressions.
- Updated local deployment guidance to reflect fully local browser UI assets.

## 0.18.0 - 2026-05-06

- Added local production CSS build assets and removed the runtime Tailwind CDN dependency.
- Added final local deployment guidance, troubleshooting notes, and Phase 18 acceptance reporting.
- Added the standalone MIT `LICENSE`.
- Added the Phase 18 validation path for backup and restore-plan rehearsal.

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
