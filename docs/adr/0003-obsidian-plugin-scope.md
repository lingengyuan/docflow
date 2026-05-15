# ADR 0003 — Scope of the in-repo Obsidian plugin

- **Status**: Proposed (decision pending)
- **Date**: 2026-05-15
- **Deciders**: maintainers

## Context

The repository contains `obsidian-plugin/docflow-assistant/` (4 files, ~32KB: `manifest.json`, `main.js`, `styles.css`, `README.md`).

The main `README.md`, `ROADMAP.md`, `CHANGELOG.md`, and `docs/features.md` make **no mention** of this plugin. There is no CI step that builds or tests it. There is no versioning relationship documented between the plugin and the Python package.

This is a positioning problem:

- If DocFlow's value is "personal knowledge workspace," then Obsidian is a competitor in the same category. Bundling a plugin for a competitor signals that DocFlow is really a Q&A back-end *for* Obsidian, not a workspace of its own.
- If DocFlow's value is "best local RAG, integrate anywhere," then the Obsidian plugin is fine — but then we should also have plugins for Logseq, VSCode, Raycast, etc., or at least say so.

The project cannot occupy both positions credibly. See `docs/critique-2026-05.md` §1.3.

## Options

### Option A — Split it out
Move `obsidian-plugin/docflow-assistant/` into a separate repository `docflow-obsidian`. Reference it from main `README.md` and `docs/features.md` as "third-party integration: Obsidian."

- ✅ Main repo's identity is clearly "standalone workspace."
- ✅ Plugin can have its own release cadence, CI, marketplace publishing.
- ⚠️ Cross-repo coordination cost when the HTTP API changes.

### Option B — Adopt it as a first-class integration
Keep `obsidian-plugin/` in the monorepo. Add:
- Build / lint step to CI.
- A line in `docs/features.md` ("Use DocFlow from Obsidian via the bundled plugin").
- A versioning rule: plugin major version tracks DocFlow major version.
- Treat the plugin as evidence that DocFlow's value extends *beyond* its own UI.

- ✅ Honest about today's reality (the plugin already exists in-tree).
- ✅ Reduces user friction (one repo, one install).
- ⚠️ Tells users "DocFlow is a backend service" more than "DocFlow is a workspace." Mitigation: ship both — strong standalone workspace **and** integration points.

### Option C — Delete it
Remove `obsidian-plugin/` outright. Stop offering this integration.

- ✅ Maximum focus.
- ❌ Loses whatever existing users this plugin has, with no migration path.
- ❌ Doesn't address the underlying "are we a workspace or a backend?" question.

## Decision

**Pending.** The intent of this ADR is to force a decision in a follow-up PR rather than let the current ambiguous state persist.

Recommended default if no other input arrives: **Option B**, because it matches today's reality and lets the main UI carry the "standalone workspace" claim while the plugin carries the "integrate anywhere" claim. But this requires the standalone UI to actually become workspace-grade (see Phase 4-6 of [`../improvement-roadmap.md`](../improvement-roadmap.md)) — otherwise Option B leaves DocFlow looking like an Obsidian backend with a half-finished frontend.

## Consequences

To be filled in once decided.
