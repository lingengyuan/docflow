# ADR 0002 — Local-first, no telemetry, no SaaS fallback

- **Status**: Accepted
- **Date**: 2026-05-15
- **Deciders**: maintainers

## Context

DocFlow is positioned as a personal document Q&A and knowledge workspace. The product's value claim depends on a hard boundary: a user's documents, metadata (SQLite), vectors (Qdrant), indexes, and queries stay on their machine unless the user **explicitly** opts into an external feature.

Competing products in the space frequently:

- Send telemetry "for product improvement."
- Add a SaaS fallback path "for when the local model is down."
- Upload document text or embeddings to a third-party service "for better retrieval."

Each of these breaks the local-first contract silently. The project rule "no masking fallbacks: fallback behavior must not hide failures, data loss, stale data, or reduced answer quality" (see `AGENTS.md`) is the long-standing intent.

## Decision

DocFlow ships with:

- **Zero telemetry.** No analytics, no error reporting, no usage metrics phoned home.
- **Zero automatic upload.** Documents, SQLite metadata, Qdrant vectors, backups, indexes never leave the machine by default.
- **No SaaS fallback.** When the user-selected local model is unavailable, the answer path **fails loudly**. It does not silently call a hosted model.
- **Opt-in external surfaces.** Webpage import, model weight downloads, and OpenAI-compatible cloud endpoints exist but are explicit and configured per-feature.

The `docflow doctor --offline` command exists to give users a verifiable check that the covered paths produce no unexpected outbound connections.

## Consequences

**Positive**
- Clear story for privacy-sensitive users (legal, medical, research notes).
- Easier to reason about failure modes: a broken local backend produces a broken answer, never a degraded-but-cloudy answer.
- Cuts out an entire class of supply-chain risk.

**Negative**
- Users hitting "my Ollama is down" get an error instead of a fallback. This is the explicit trade-off.
- Onboarding requires the user to install a local model runtime; we can't paper over that step.
- New contributors must internalize: anything that could become a silent fallback in a future PR is unacceptable. ADR 0001 + this ADR together make that reviewable.

## Compliance check

- Any new code path that opens a network connection must be:
  1. opt-in via `config.yaml` or explicit user gesture, **and**
  2. visible in `docflow doctor --offline` so the user can audit it.
- Any new "fallback" behavior must surface the underlying failure to the user (no swallowed exceptions, no silent degraded answer).

