# Phase 28 Handoff - User-Facing UI Cleanup

Date: 2026-05-08

## Completed Scope

- Updated the project phase rule: every completed phase now requires a handoff document, a focused commit, and a successful GitHub push before reporting completion or starting the next phase.
- Reworked Settings from a maintenance-style surface into an ordinary user-facing settings page:
  - local system status,
  - local model status,
  - monitored folders,
  - daily-use preferences,
  - status hints,
  - source-scope context.
- Removed visible command-line, recovery, dry-run, repair, install, doctor, and copy-command wording from the normal browser UI.
- Added a UI safety layer so command-like health details from `/api/health` are rendered as user-friendly state text in Settings.
- Updated browser acceptance so Settings fails if developer-only wording becomes visible again.
- Updated README, CHANGELOG, and the maturity scorecard for the product/CLI separation.

## Changed Files

- `AGENTS.md`
  - Added the rule that every phase must be committed and pushed to GitHub before moving on.
- `docs/phase27-handoff.md`
  - Updated the next phase from workflow coverage to the UI cleanup required by the latest plan.
- `frontend/index.html`
  - Removed Settings maintenance commands and recovery wording.
  - Added Settings status hints, source-scope context, and safer health-detail rendering.
  - Adjusted Settings layout after screenshot review.
- `frontend/styles.css`
  - Rebuilt after UI changes.
- `src/quality/browser_acceptance.py`
  - Updated Settings expectations and added a forbidden-language check.
- `tests/test_static_assets.py`
  - Updated static UI tests to require the new Settings surface and prevent old command wording from returning.
- `tests/test_browser_acceptance.py`
  - Updated acceptance-plan expectations for Settings.
- `README.md`
  - Updated English and Chinese product facts so the browser UI no longer claims to show maintenance commands.
- `CHANGELOG.md`
  - Added `0.28.0`.
- `eval/phase11_maturity_dimensions.json`
  - Raised UI usability to `9.0` and documented the Phase 28 cleanup.

## Validation

```bash
npm run build:css
```

Result: passed. Existing Browserslist caniuse-lite warning remains.

```bash
.venv/bin/python -m pytest tests/test_static_assets.py tests/test_browser_acceptance.py
```

Result: 22 passed, 5 warnings.

```bash
.venv/bin/python main.py browser-acceptance --screenshots-dir output/playwright/phase28-ui-cleanup --json
```

Result: 56 passed, 0 failed. Screenshots were written to `output/playwright/phase28-ui-cleanup/`.

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

Result: overall score is `8.78 / 10`.

The first browser acceptance run caught a real Settings leak: the SQLite health detail still displayed `python main.py doctor --strict`. The Settings renderer now replaces command-like health details with ordinary status text, and the browser acceptance check now guards against that regression.

## Known Limitations

- `/api/health` still includes diagnostic actions for CLI and documentation use. Phase 28 only prevents those details from being exposed in the normal browser UI.
- Phase 28 did not add new temporary-file upload, scan, batch metadata, or batch rebuild browser workflows.
- The Settings health card can still be tall when many local capabilities are enabled; the page scrolls correctly, but Phase 29 should decide whether to compact the health display further.

## Current Score

- Overall maturity: `8.78 / 10`.
- UI usability: `9.0 / 10`.
- Remaining largest gaps:
  - model task cancellation and clearer progress,
  - real browser workflow coverage for upload, scan, import, and batch actions,
  - richer Knowledge Output customization and preview,
  - stronger Library saved workflows.

## Next Phase

Proceed to Phase 29: Final UI Plan Acceptance and Transition.

Recommended Phase 29 tasks:

1. Capture final browser screenshots for Chat, Library, Notes, and Settings, and update README screenshots if needed.
2. Run final browser acceptance and full tests after the Phase 28 UI cleanup.
3. Update the maturity scorecard and handoff with the final state of the UI redesign plan.
4. Decide whether the current UI plan is complete and explicitly hand off remaining work to the post-review roadmap.
5. Commit Phase 29 and push it to GitHub before moving to any post-review optimization phase.
