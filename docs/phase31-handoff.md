# Phase 31 Handoff - UI Reference Fine-Tuning

Date: 2026-05-10

## Completed Scope

Phase 31 was treated as the Phase30.1 UI reference gate. The browser UI was repaired against the five saved reference images before continuing to the later roadmap phases.

Completed repairs:

- Chat now opens in a real answered state when history exists, with a user question, answer card, answer actions, citations, source preview, model status, and task panel.
- Library now has persistent collection/tag browsing, page controls, a denser table, and a tab-like detail panel with detail, preview, content, and related sections.
- Notes now matches the reference layout more closely: editor, web import, knowledge-output controls, saved answers, processing timeline, and recent capture table are present.
- Settings now uses compact system cards, a model table, monitored-folder table, true local storage usage, and user-facing guidance without command-line or recovery wording.
- The reference storage block gap was repaired after review: the sidebar and Settings page now show real disk usage, local file usage, model cache usage, app data usage, and other local usage instead of only file counts.
- A follow-up missing-information audit found and fixed a Library refresh race where rapid group switching could leave the main table empty even though the side rail showed 268 files.
- Source Preview now opens to a complete evidence-reading page, waits for real content before acceptance screenshots, and shows document text, highlighted evidence, relationship context, keywords, actions, and timeline controls.

## Changed Files

- `frontend/index.html`
- `frontend/styles.css`
- `src/api/app.py`
- `src/quality/browser_acceptance.py`
- `tests/test_api_storage.py`
- `tests/test_browser_acceptance.py`
- `tests/test_static_assets.py`
- `docs/phase31-missing-info-audit.md`
- `docs/phase31-handoff.md`

## Visual Evidence

Browser screenshots:

- `output/playwright/phase31-browser-acceptance/01-chat.png`
- `output/playwright/phase31-browser-acceptance/02-library.png`
- `output/playwright/phase31-browser-acceptance/03-source.png`
- `output/playwright/phase31-browser-acceptance/04-notes.png`
- `output/playwright/phase31-browser-acceptance/05-settings.png`
- `output/playwright/phase31-storage-browser-acceptance/05-settings.png`
- `output/playwright/phase31-missing-info-audit/browser-acceptance/02-library.png`
- `output/playwright/phase31-missing-info-audit/all-pages-side-by-side-updated.png`

Reference comparison sheets:

- `output/playwright/phase31-reference-comparison/chat.png`
- `output/playwright/phase31-reference-comparison/library.png`
- `output/playwright/phase31-reference-comparison/notes.png`
- `output/playwright/phase31-reference-comparison/settings.png`
- `output/playwright/phase31-reference-comparison/source-open.png`

## Validation

Commands run:

```bash
npm run build:css
.venv/bin/python -m pytest
.venv/bin/python main.py browser-acceptance --base-url http://127.0.0.1:8000 --screenshots-dir output/playwright/phase31-browser-acceptance --json
.venv/bin/python main.py browser-acceptance --base-url http://127.0.0.1:8010 --screenshots-dir output/playwright/phase31-storage-browser-acceptance --json
.venv/bin/python main.py browser-acceptance --base-url http://127.0.0.1:8010 --screenshots-dir output/playwright/phase31-missing-info-audit/browser-acceptance --json
curl -s http://127.0.0.1:8010/api/storage/usage
```

Results:

- CSS build passed.
- Full test suite passed: 208 passed, 5 warnings.
- Browser acceptance passed on the clean Phase31 storage server: 68 passed, 0 failed.
- Missing-information audit browser acceptance passed after the Library refresh-race fix: 73 passed, 0 failed.
- Library audit confirmed the table now renders real rows after rapid group switching: 14 rows shown, 268 / 268 files counted.
- Storage API returned real local values during validation: 248 GB used of 926 GB, 268 existing files, about 38 GB model cache, about 1.0 GB app data, and no missing source files.
- Settings acceptance confirmed no developer or command-line language is exposed in the normal UI.
- A separate check against `127.0.0.1:8000` initially hit an older already-running service and returned a storage API 404, so final browser validation was rerun on `127.0.0.1:8010` with the current code.

## Known Limitations

- The screenshots use the user's current local data, so document names and answer content do not match the reference mock data exactly.
- The UI now matches the reference structure and product tone closely enough to pass the Phase31 gate, but exact pixel identity is still limited by live data differences and the existing single-file frontend structure.
- Source Preview shows real chunk data. When a source has no retrieval score, it shows a truthful verifiable-source state instead of a fake confidence value.
- Storage usage is read from the local machine at runtime, so exact GB numbers will change as files, models, and system storage change.

## Next Phase

Proceed to Phase 32 only after committing and pushing this Phase 31 checkpoint.

Recommended next work:

1. Continue the post-review roadmap phases that were deferred by the UI reference gate.
2. Keep the browser acceptance screenshots as the regression baseline for future UI work.
3. Avoid reintroducing developer-only wording into the normal browser UI.
