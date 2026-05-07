# Changelog

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
