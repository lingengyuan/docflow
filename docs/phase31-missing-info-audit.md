# Phase 31 Missing Information Audit

Date: 2026-05-10

## Scope

This audit rechecked the five saved UI reference images after the storage-usage repair. The goal was to find problems similar to the missing storage block: reference-visible, user-valuable information or controls that the current UI either omitted or replaced with misleading placeholders.

Reference images:

- `/Users/hughlin/MyNotes/HughLin/Notes/plans/docflow/assets/docflow-ui-redesign/01-overview-workspace.png`
- `/Users/hughlin/MyNotes/HughLin/Notes/plans/docflow/assets/docflow-ui-redesign/02-library.png`
- `/Users/hughlin/MyNotes/HughLin/Notes/plans/docflow/assets/docflow-ui-redesign/03-notes.png`
- `/Users/hughlin/MyNotes/HughLin/Notes/plans/docflow/assets/docflow-ui-redesign/04-settings.png`
- `/Users/hughlin/MyNotes/HughLin/Notes/plans/docflow/assets/docflow-ui-redesign/05-source-preview.png`

Current evidence:

- `output/playwright/phase31-missing-info-audit/browser-acceptance/01-chat.png`
- `output/playwright/phase31-missing-info-audit/browser-acceptance/02-library.png`
- `output/playwright/phase31-missing-info-audit/browser-acceptance/03-source.png`
- `output/playwright/phase31-missing-info-audit/browser-acceptance/04-notes.png`
- `output/playwright/phase31-missing-info-audit/browser-acceptance/05-settings.png`
- `output/playwright/phase31-missing-info-audit/all-pages-side-by-side-updated.png`

## Findings

### Fixed: Library table could show an empty result after rapid filter changes

The first comparison sheet showed a serious false state: the Library side rail said 268 files existed, while the main table showed no matching files.

Root cause:

- The Library page can issue overlapping refreshes when a user switches groups quickly.
- A stale empty result, for example from the PDF group, could arrive after the later `All files` refresh and overwrite the table.

Fix:

- `refreshFiles()` now tracks the latest request and ignores stale responses.
- Browser acceptance now waits for each view to finish loading before screenshots.
- Library acceptance now requires real rows after returning to `All files` when the library count is greater than zero.

Validation result:

- Library screenshot now shows 14 rendered rows and `显示 268 / 268 个文件`.
- Browser acceptance passed: 73 passed, 0 failed.

### Confirmed: Settings storage block is now present and real

The earlier missing storage block is now covered:

- Sidebar shows local storage usage.
- Settings shows total storage, available space, file count, collection count, model cache, app data, library files, and other local usage.
- Values come from the local runtime instead of static mock numbers.

### Confirmed: No other same-class missing blocks found in this pass

The remaining differences are not the same class as the missing storage block. They are mostly live-data differences, density differences, or future polish:

- Chat has an answered state, citation list, source preview, model status, and task panel.
- Library has group counts, collection filtering, table rows, detail tabs, source preview actions, and pagination.
- Notes has editor controls, web import, knowledge-output templates, saved answers, processing progress, and recent capture area.
- Settings has status cards, model table, monitored folder table, storage usage, and user-facing suggestions without developer commands.
- Source Preview has source list, document preview, citation detail, highlighted evidence, keywords, timeline, and source actions.

## Residual Non-Blocking Differences

These are still worth improving later, but they are not hidden missing-data bugs:

- Some reference screenshots contain richer mock data than the user's current local data.
- Some runtime metrics from the reference, such as speed or temperature, should only be shown if the app can report truthful values.
- The current UI remains less pixel-identical than the references in spacing and first-screen density, but the major information blocks now exist.

## Next Step

Proceed only after committing and pushing this audit fix. Later UI work should use the stricter browser screenshots so loaded data is captured instead of transient empty states.
