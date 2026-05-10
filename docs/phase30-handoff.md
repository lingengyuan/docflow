# Phase 30 Handoff - UI Reference Gap Repair

Date: 2026-05-08

## Completed Scope

Phase 30 moved the one-to-one UI reference gap analysis into the roadmap as the next required phase before later review-driven engineering work.

Implemented first-pass repairs for the most visible differences:

- Added a dedicated Source Preview navigation item and three-column source review page.
- Connected Library source actions to the new Source Preview page with real file chunks.
- Reduced Library upload-zone visual weight and removed raw absolute file paths from default table rows.
- Localized Library file status labels and replaced technical `Chunks` wording with `片段`.
- Fixed Library type counts in the running UI by deriving counts from the actual file list when the current server metadata is incomplete.
- Localized Notes and Settings titles.
- Added real Notes Markdown toolbar actions.
- Replaced the knowledge-output dropdown surface with selectable template cards while keeping the existing request contract.
- Added Settings storage overview cards and made monitored-folder labels more user friendly.
- Removed visible command/path/developer wording introduced by earlier prototype-style surfaces.
- Updated the post-review roadmap so this work is Phase 30 and later phases are shifted.

## Changed Files

- `frontend/index.html`
- `frontend/styles.css`
- `src/quality/browser_acceptance.py`
- `tests/test_browser_acceptance.py`
- `tests/test_static_assets.py`
- `docs/phase29-handoff.md`
- `docs/phase30-ui-reference-gap-analysis.md`
- `docs/phase30-handoff.md`
- `/Users/hughlin/MyNotes/HughLin/Notes/plans/docflow/docflow-post-review-roadmap.md`

## Validation

Commands run:

```bash
npm run build:css
node - <<'NODE'
const fs = require('fs');
const html = fs.readFileSync('frontend/index.html', 'utf8');
const scripts = [...html.matchAll(/<script>([\s\S]*?)<\/script>/g)].map(m => m[1]).join('\n');
new Function(scripts);
console.log('inline script syntax ok');
NODE
.venv/bin/python -m pytest tests/test_static_assets.py tests/test_browser_acceptance.py tests/test_store.py::TestDocStore::test_file_metadata_filters_and_facets
.venv/bin/python main.py browser-acceptance --base-url http://127.0.0.1:8000 --screenshots-dir output/playwright/phase30-browser-acceptance
.venv/bin/python -m pytest
```

Results:

- CSS build passed.
- Inline script syntax check passed.
- Focused tests: `23 passed`.
- Browser acceptance: `65/65 passed`.
- Full test suite: `206 passed`.
- Manual source-preview browser flow: opening Source Preview from the first Library row loaded `20 个片段` and exposed `保存片段`.

Screenshots reviewed:

- `output/playwright/phase30-browser-acceptance/02-library.png`
- `output/playwright/phase30-browser-acceptance/03-source.png`
- `output/playwright/phase30-browser-acceptance/04-notes.png`
- `output/playwright/phase30-browser-acceptance/05-settings.png`
- `output/playwright/phase30-source-open.png`

## Known Limitations

- Chat still needs a reference-style answered-state screenshot and richer answered-state polish.
- Source Preview now exists and loads real chunks, but its empty default state is still sparse until entered from Chat or Library.
- Library detail tabs and pagination are not fully restored yet.
- Notes import options and right-panel saved-answer/timeline depth are still lighter than the reference.
- Settings storage overview uses truthful available browser/library statistics, not full local disk analytics.
- Direct Codex in-app Browser automation was attempted but its runtime connection timed out; validation used project browser acceptance plus Playwright screenshots against the same running local app.

## Next Phase

Proceed to Phase 31: UI Reference Fine-Tuning. This is the Phase30.1 blocker requested after reviewing the latest reference comparison.

Phase 31 must finish the UI reference restoration before any low-cost fixes or engineering debt work continues.

Exact next tasks:

1. Use `docs/phase31-ui-reference-finetune-plan.md` as the Phase 31 scope.
2. Repair all remaining reference mismatches across Chat, Library, Notes, Settings, and Source Preview.
3. Generate a new `reference vs current` UI comparison set after the repairs.
4. If any page still differs from its reference, Phase 31 is not complete; continue fixing before moving on.
5. Keep ordinary-user UI rules intact: no command-line, maintenance, recovery, script, or copyable terminal wording in the browser UI.
6. Run full validation, write `docs/phase31-handoff.md`, commit, and push before moving to Phase 32.
