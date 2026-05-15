# ADR 0003 — Scope of the in-repo Obsidian plugin

- **Status**: Accepted
- **Date**: 2026-05-15
- **Deciders**: maintainers

## Context

The repository contains `obsidian-plugin/docflow-assistant/` (4 files: `manifest.json`, `main.js`, `styles.css`, `README.md`) plus `/api/obsidian/*` endpoints and tests.

The main product scope is now a standalone local desktop browser knowledge workspace. The project is not pursuing third-party application integrations in the current quality push: no Obsidian plugin, VS Code plugin, browser extension, Notion/Readwise/email connector, mobile app, or PWA install path.

This makes the in-repo Obsidian plugin a positioning and maintenance problem:

- It tells users DocFlow may be a back-end for another app, not its own workspace.
- It expands the supported surface without matching CI, docs, release, or product ownership.
- It distracts from the current goal: making the standalone desktop product credible.

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
- ✅ Directly answers the current scope question: DocFlow is a standalone workspace first.

## Decision

Choose **Option C — Delete it** from the main repository during Phase102.

DocFlow may keep support for local Markdown syntax commonly found in personal notes, such as wikilinks, frontmatter, and callouts. That is file-format compatibility, not an Obsidian integration.

## Consequences

- Phase102 removes `obsidian-plugin/`, `/api/obsidian/*`, and Obsidian-specific tests.
- Public docs should describe DocFlow as a standalone local desktop knowledge workspace.
- Any future Obsidian work must be proposed as a separate project or a later explicit scope expansion.
- Markdown parsing tests should use neutral names such as wikilink/frontmatter/callout instead of Obsidian integration wording.
