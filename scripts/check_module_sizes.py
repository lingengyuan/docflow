#!/usr/bin/env python3
"""Module-size budget check.

The project principle (AGENTS.md / Project Principles) is to keep modules small,
clear, and localized. To make drift visible, this script enforces a soft LOC
budget for every Python module under ``src/``.

Rules:
- Default budget: ``DEFAULT_BUDGET`` lines (counting all lines, blank + comment
  included — same as ``wc -l``, so the number matches what reviewers see).
- Modules in ``GRANDFATHERED`` are allowed to exceed the budget but **MUST NOT
  grow beyond their recorded ceiling**. The ceiling is the LOC at the time the
  module was grandfathered, rounded up to a small headroom.
- Any new module that crosses the budget fails CI immediately.
- Any grandfathered module that grows beyond its ceiling fails CI.

This forcing function lets us refactor the existing god modules in dedicated
follow-up PRs without freezing development on them in the meantime, while
preventing the situation from getting worse.

Run locally::

    python scripts/check_module_sizes.py

Run with JSON output (for CI / dashboards)::

    python scripts/check_module_sizes.py --json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "src"

DEFAULT_BUDGET = 800

# Grandfathered modules: existing god modules that exceed the budget today.
# Each ceiling is current LOC + ~5% headroom, rounded up to the next 10.
# Refactoring tickets per module belong in the engineering hardening roadmap.
GRANDFATHERED: dict[str, int] = {
    "src/quality/browser_acceptance.py": 1010,
    "src/maintenance/startup.py": 740,
}


def loc(path: Path) -> int:
    with path.open("rb") as fh:
        return sum(1 for _ in fh)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json", action="store_true", help="Emit machine-readable JSON.")
    parser.add_argument(
        "--budget",
        type=int,
        default=DEFAULT_BUDGET,
        help=f"Default LOC budget per module (default: {DEFAULT_BUDGET}).",
    )
    args = parser.parse_args(argv)

    violations: list[dict[str, object]] = []
    report: list[dict[str, object]] = []

    for path in sorted(SRC.rglob("*.py")):
        rel = path.relative_to(ROOT).as_posix()
        lines = loc(path)
        ceiling = GRANDFATHERED.get(rel, args.budget)
        entry: dict[str, object] = {
            "path": rel,
            "loc": lines,
            "ceiling": ceiling,
            "grandfathered": rel in GRANDFATHERED,
        }
        report.append(entry)
        if lines > ceiling:
            violations.append(entry)

    if args.json:
        json.dump(
            {"budget": args.budget, "violations": violations, "report": report},
            sys.stdout,
            indent=2,
            sort_keys=True,
        )
        sys.stdout.write("\n")
    else:
        for entry in report:
            mark = "!" if entry in violations else ("g" if entry["grandfathered"] else " ")
            print(f"  {mark} {entry['loc']:>5}  {entry['path']}  (ceiling {entry['ceiling']})")
        if violations:
            print()
            print(f"FAIL: {len(violations)} module(s) over budget:")
            for entry in violations:
                print(
                    f"  - {entry['path']}: {entry['loc']} > {entry['ceiling']}"
                    + (" (grandfathered ceiling)" if entry["grandfathered"] else "")
                )
            print(
                "\nTo fix: split the module, or — only as a last resort — raise its "
                "grandfathered ceiling in scripts/check_module_sizes.py and justify "
                "the bump in the PR description."
            )
        else:
            print(f"\nOK: all {len(report)} modules within budget.")

    return 1 if violations else 0


if __name__ == "__main__":
    raise SystemExit(main())
