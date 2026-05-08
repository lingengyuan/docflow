# Phase 27 Handoff - Library and Source Preview Upgrade

Date: 2026-05-08

## Completed Scope

- Added real Library groups: all files, favorites, recent imports, PDF, Markdown, images, and code.
- Wired the group controls into `/api/files`, so the table is filtered by backend data instead of only changing UI state.
- Added Library facet counts for file types and recent imports.
- Upgraded the Library right detail panel with:
  - source chunk review,
  - recent citation history,
  - open-original action,
  - save chunk as a local note,
  - file-scoped question shortcut,
  - maintenance actions for knowledge output and rebuild.
- Extended browser acceptance so Library validation clicks group filters and opens source review.
- Updated README, CHANGELOG, and the maturity scorecard for Phase 27.

## Changed Files

- `frontend/index.html`
  - Added Library group controls and active group state.
  - Added source review, recent citations, save-as-note, and file-scoped question actions.
- `frontend/styles.css`
  - Rebuilt from Tailwind after UI changes.
- `src/api/app.py`
  - Added `kind` and `recent` query params for `/api/files`.
- `src/ingest/store.py`
  - Added backend file-kind filtering, recent filtering, and Library type facets.
  - Preserved tag filtering correctness when combined with recent imports.
- `src/quality/browser_acceptance.py`
  - Added Library group and source review checks.
- `tests/test_api_files.py`
  - Covered new `/api/files` filter params.
- `tests/test_store.py`
  - Covered kind/recent filtering, type facets, and recent-with-tag ordering.
- `README.md`
  - Updated current release target and Library/Source Preview feature facts.
- `CHANGELOG.md`
  - Added `0.27.0`.
- `eval/phase11_maturity_dimensions.json`
  - Raised UI usability to `8.9` and Library management to `8.6`.

## Validation

```bash
npm run build:css
```

Result: passed. Existing Browserslist caniuse-lite warning remains.

```bash
.venv/bin/python -m pytest tests/test_store.py tests/test_api_files.py tests/test_api_debug.py tests/test_static_assets.py
```

Result: 48 passed, 5 warnings.

```bash
.venv/bin/python main.py browser-acceptance --json
```

Result: 55 passed, 0 failed.

```bash
.venv/bin/python -m pytest
```

Result: 206 passed, 5 warnings.

```bash
/Users/hughlin/.agents/skills/readme-maintainer/scripts/check_bilingual_readme.sh README.md
```

Result: passed.

```bash
.venv/bin/python main.py maturity-eval --skip-retrieval --json
```

Result: overall score is `8.77 / 10`.

The first browser run exposed a source review timing/state mismatch. The UI now reports source review failures with a clear `读取失败` state, and the browser check waits for the final rendered source review state before judging the result.

## Known Limitations

- Browser acceptance still writes screenshots into `output/playwright/phase25-browser-acceptance`; the checks are current, but the default directory name is older.
- Phase 27 covers Library group filters and source review, but it does not yet create temporary files inside the browser flow to test upload, scan, batch metadata, or batch rebuild end to end.
- Source chunk review shows indexed chunks and text previews; it does not yet highlight exact matching spans inside the original source preview.
- Saved filters are not persisted across browser sessions.

## Current Score

- Overall maturity after the scorecard update: `8.77 / 10`.
- Biggest remaining gaps:
  - model runtime progress and cancellation,
  - full real-workflow browser coverage for upload/scan/import/batch actions,
  - stronger saved Library workflows.

## Next Phase

Proceed to Phase 28: User-Facing UI Cleanup.

Recommended Phase 28 tasks:

1. Remove maintenance commands, recovery-advice wording, dry-run wording, and copyable terminal commands from the normal browser UI.
2. Rework Settings into an ordinary personal-knowledge settings page: local status, model status, monitored folders, usage preferences, and clear user-facing state hints.
3. Update browser acceptance and static checks so the normal UI fails if command-line or developer-console language becomes visible again.
4. Update README/CHANGELOG and the maturity scorecard to reflect the product/CLI separation.
5. Update the Phase 28 handoff, rerun validation, commit the phase, and push to GitHub before starting Phase 29.

Project rule update for later sessions:

- Every phase must now end with a handoff document, a focused commit, and a successful push to GitHub before the next phase begins.
