# ADR 0004 — Engineering hardening strategy (mypy white-list, coverage rollout, module-size budget)

- **Status**: Accepted
- **Date**: 2026-05-15
- **Deciders**: maintainers
- **Supersedes / amends**: none

## Context

`docs/improvement-roadmap.md` Phase 1 targets Engineering ≥ 90 and lists five workstreams: mypy
full coverage, coverage in CI, requirements/extras sync, splitting large modules, and ADR
discipline. The honest situation today (recorded in `docs/scoring-2026-05.md`):

- `mypy` `files=` covered only **4 paths** under `src/` (`api/schemas.py`, `api/runtime.py`,
  `api/services`, `query/`). Roughly two thirds of the `src/` tree was invisible to the type
  checker. A scan of the full tree (`mypy --ignore-missing-imports src`) surfaces **35 errors
  across 11 modules**.
- CI runs `pytest` but **does not measure coverage**. Without a baseline number the roadmap's
  "set fail-under to baseline − 2" plan cannot start.
- Five modules under `src/` are ≥ 400 LOC and two are ≥ 700 LOC, contradicting the project
  principle "keep modules small, clear, and localized" (AGENTS.md → Project Principles).
  There is no mechanism preventing further drift.

A naive "fix everything at once" PR would either (a) silently hide errors with broad ignores,
or (b) blow up into a multi-thousand-line refactor that nobody can review. We need a strategy
that **expands the safety net without freezing development on the legacy hotspots**, and that
turns "we should refactor this someday" into a forcing function.

## Decision

We adopt a three-pronged hardening strategy, each part designed to land in a small PR.

### 1. Mypy: scope-first, fix-later

- `tool.mypy.files` is set to `["src", "main.py"]` — the **entire** Python source tree is in
  scope for type checking.
- A single `[[tool.mypy.overrides]]` block with `ignore_errors = true` white-lists the **exact
  set of modules** that fail mypy today. This list is the engineering backlog made concrete.
- **Invariant: the list must shrink, never grow.** Adding a new module to it is a regression
  and must be justified in the PR description.
- Trivial fixes that fall out of writing the override list (e.g. shadowed-name, simple
  annotation gaps) are taken inline as proof the mechanism works.

This gives us 100% scope coverage (`mypy` now checks 100 source files vs. 24 before) while
keeping CI green and giving us a concrete shrinking-list to drive follow-up PRs.

### 2. Coverage: collect first, gate later

- Add `pytest-cov` to dev dependencies (`pyproject.toml` extras + `requirements-dev.txt`).
- CI runs `pytest --cov=src --cov-report=xml --cov-report=term-missing` on the Linux/3.12 leg
  only (other matrix legs keep the existing fast `pytest`).
- The XML report is uploaded as a GitHub Actions artifact (`coverage-xml`). No
  `--cov-fail-under` gate is set in this PR — that's a follow-up once we have a stable
  baseline across a few PRs.

This avoids the trap of guessing a fail-under number, locking it in, and then having every
unrelated PR fight the gate.

### 3. Module size: budget + grandfathering

- New script `scripts/check_module_sizes.py` enforces a default budget of **800 LOC per
  module** under `src/`.
- Existing modules over budget are grandfathered with explicit ceilings (current LOC +
  ~5% headroom). They are allowed to exist but **not allowed to grow**.
- New modules over budget fail CI immediately, forcing the author to split.
- The script is wired into the main CI job after `mypy`.

Today's grandfathered list:

| Module | Current LOC | Grandfathered ceiling |
| --- | --- | --- |
| `src/quality/browser_acceptance.py` | 999 | 1010 |
| `src/maintenance/startup.py` | 728 | 740 |

Every other module under `src/` is within the 800-line budget. Each grandfathered entry
implies a future refactor PR (Phase 1 item 4); landing those PRs lowers the ceiling toward 800
until the override can be removed.

## Consequences

**Positive**

- Every new `.py` file under `src/` is type-checked from day one.
- The "we should fix mypy in those modules" backlog is now a literal list in `pyproject.toml`
  that shrinks with each follow-up PR — easy to review and impossible to forget.
- Coverage is measurable for the first time without taking on the political cost of a gate.
- Large-module drift is now visible and self-enforcing.

**Negative**

- The mypy override list looks like "we cheated" until people read this ADR. The
  `[[tool.mypy.overrides]]` block carries a comment pointing here.
- The grandfathered ceilings are a per-file rachet; if a legitimate feature needs a tiny
  bump on a grandfathered file, the bump must be argued in the PR. This is intentional friction.
- One existing structural test (`test_phase98_…`) pinned the **narrow** old mypy scope. It is
  updated to pin the **broader** new scope; the original assertion is strictly weaker.

## Out of scope (deliberately deferred)

- Actually splitting `browser_acceptance.py` and `startup.py` — one risky refactor PR each,
  with full test runs before/after.
- Fixing the 33 mypy errors that remain inside white-listed modules — each in its own PR per
  module, removing the override line when done.
- Setting `--cov-fail-under` — pick this once 3–4 PRs have stabilized the baseline.
- Replacing `requirements-*.txt` with `pyproject.toml` extras as the canonical install path
  (touches CI install commands; needs its own PR).
