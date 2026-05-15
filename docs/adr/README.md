# Architectural Decision Records (ADR)

This directory captures DocFlow's architectural decisions in a lightweight format inspired by Michael Nygard's original ADR template.

## Index

| # | Title | Status |
| --- | --- | --- |
| [0001](0001-module-boundaries.md) | Module boundaries: ingest / storage / retrieval / api / ui kept orthogonal | Accepted |
| [0002](0002-local-first-no-telemetry.md) | Local-first, no telemetry, no SaaS fallback | Accepted |
| [0003](0003-third-party-integration-scope.md) | Scope of third-party app integrations | Accepted |

## How to add a new ADR

1. Pick the next sequence number (`NNNN`).
2. Copy an existing file as a template.
3. Fill in **Context**, **Decision**, **Consequences**, **Status**.
4. Add a row to the index above in the same PR.
5. Reference the ADR from `docs/architecture.md` if the decision changes the picture there.

ADRs are **decisions**, not designs. They explain *why*, not *how*. They are immutable once accepted — supersede by adding a new ADR that references the old one.
