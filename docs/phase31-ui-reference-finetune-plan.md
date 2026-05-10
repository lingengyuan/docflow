# Phase 31 UI Reference Fine-Tuning Plan

Date: 2026-05-10

## Rule

Phase 31 is the required Phase30.1 blocker before the old Phase31 work can continue. It is complete only when a new visual comparison shows that the browser UI matches the saved reference images in structure, state, information density, wording, and meaningful controls.

If any page is still visibly inconsistent with its reference, Phase 31 is not complete. Continue fixing and rerun the comparison before moving on.

## Inputs

Reference images:

- `docs/ui-concepts/phase26/01-overview-workspace.png`
- `docs/ui-concepts/phase26/02-library.png`
- `docs/ui-concepts/phase26/03-notes.png`
- `docs/ui-concepts/phase26/04-settings.png`
- `docs/ui-concepts/phase26/05-source-preview.png`

Latest comparison evidence:

- `output/playwright/phase30-reference-comparison/chat.png`
- `output/playwright/phase30-reference-comparison/library.png`
- `output/playwright/phase30-reference-comparison/notes.png`
- `output/playwright/phase30-reference-comparison/settings.png`
- `output/playwright/phase30-reference-comparison/source-open.png`

## Required Repairs

### Phase31.2 Supplemental Polish

This supplemental pass is part of Phase 31 and must be completed before Phase 32. It covers the residual visual differences found after the missing-information audit. These are not new roadmap items; they are the remaining Phase31 reference-polish tasks.

Scope:

- Reduce the extra vertical space between the global top bar, page title row, and main workspace so each page matches the reference first-screen density more closely.
- Keep the Library upload control available, but make it a compact secondary drop strip instead of a dominant first-screen block.
- Make the Chat right-side model/status area denser and more truthful by showing citation/source count when an answer has sources.
- Compact the Settings system-status area into reference-like cards instead of a tall grouped list.
- Let the Notes recent-capture table appear earlier in the first screen by reducing oversized editor and source-input heights.
- Improve Source Preview's left result rail so each item names the selected section or text snippet instead of repeating the same file name on every row.
- Re-run browser screenshots and visual comparison after these changes. If any of the five pages still shows obvious spacing, density, or missing-information drift from the references, keep Phase31 open and continue polishing.

### 1. Chat / Overview

Current problem:

- Current screen is still a welcome empty state, while the reference is a real answered conversation.
- Main workspace has too much blank space.
- Right panel shows placeholders instead of real citation/model/task density.

Required final state:

- Default UI comparison must use a real answered state.
- Show user question bubble, answer card, structured content, answer action row, citations, selected source excerpt, model details, and task progress.
- Match reference spacing and density; no large empty center area.

### 2. Library

Current problem:

- The default list is still shaped around a compact upload strip and a simplified detail panel.
- Left side lacks the reference's persistent collection/tag browsing density.
- Right panel is useful but does not match the reference tabbed detail structure.
- Pagination or large-list control is missing.

Required final state:

- Toolbar, grouping, collections, tags, table density, and right detail structure must match the reference.
- Upload remains available but should not dominate the default file-manager view.
- Detail panel must include reference-like sections for details, preview, content, related/recent references, status, and suggestions.
- Counts, statuses, collections, tags, pages, chunks/片段, and timestamps must be real.

### 3. Notes

Current problem:

- The editor is still lighter than the reference.
- Import options are incomplete.
- Right panel lacks today's knowledge outputs, saved answers, and a real processing timeline.
- Recent captures table is not restored.

Required final state:

- Full Markdown editor area with metadata, toolbar, word count/mode state, and usable controls.
- Web import must include extraction options and save destination controls that map to real behavior or honest local state.
- Knowledge output cards, model selector, generation controls, today outputs, saved answers, processing progress, and recent captures table must match the reference.

### 4. Settings

Current problem:

- Current page is a simplified user-facing settings page, not the reference layout.
- System status, model list, watched folders, and storage overview are not structured like the reference.

Required final state:

- Restore reference-like system cards, local model table, watched-folder table, storage visualization, and right-side suggestion panels.
- Do not bring back developer/prototype language. If the reference contains recovery/maintenance-looking areas, rewrite them as ordinary product guidance without commands.
- Storage and model data must be truthful; no fake charts or fake numbers.

### 5. Source Preview

Current problem:

- Current page loads real chunks but is still a chunk browser, not a document reading experience.
- It lacks the reference's original-document center view, highlighted evidence, relation explanation, keywords, and source timeline.

Required final state:

- Left side source list, center document reader, right citation explanation, evidence strength, related context, keywords, save/open/recheck actions, and timeline/location controls must match the reference.
- Chat and Library source actions must both enter this complete Source Preview page.

## Validation

Required commands and checks:

```bash
npm run build:css
.venv/bin/python -m pytest
.venv/bin/python main.py browser-acceptance --base-url http://127.0.0.1:8000 --screenshots-dir output/playwright/phase31-browser-acceptance
```

Required visual evidence:

- Generate new comparison sheets under `output/playwright/phase31-reference-comparison/`.
- The comparison must cover all five reference pages.
- Write the final checklist into `docs/phase31-handoff.md`.

Completion gate:

- If the new comparison still shows mismatch in any of the five pages, keep Phase 31 open and continue fixing.
- Only after the comparison is visually consistent, tests pass, and `docs/phase31-handoff.md` is written may the project proceed to Phase 32.
