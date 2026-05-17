# ADR 0003 — Scope of third-party app integrations

- **Status**: Accepted
- **Date**: 2026-05-15
- **Deciders**: maintainers

## Context

DocFlow previously carried an in-repository plugin and backend route for a third-party note app. That expanded the project surface beyond the current goal: a standalone local desktop browser knowledge workspace.

The current desktop productization track explicitly excludes third-party application integrations, mobile apps, and installable browser shells. Keeping those surfaces in the main repository would make the product boundary unclear and increase maintenance without improving the standalone workspace.

## Decision

Third-party app integrations do not live in the main repository during the current productization track.

The existing integration plugin, API route, tests, and packaging references have been removed from the main repository. Future integrations must be proposed separately with their own ownership, CI, release path, and product rationale.

DocFlow may keep support for local Markdown syntax commonly found in personal notes, such as wikilinks, frontmatter, embeds, and callouts. That is file-format compatibility, not third-party app integration.

## Consequences

- Public docs describe DocFlow as a standalone local desktop knowledge workspace.
- The main package does not ship third-party app plugins.
- The normal browser UI remains the primary product surface.
- Markdown parsing tests use neutral file-format names.
