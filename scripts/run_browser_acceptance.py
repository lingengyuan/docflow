#!/usr/bin/env python3
"""Run the Phase 25 browser acceptance checks."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.quality.browser_acceptance import (
    DEFAULT_BASE_URL,
    DEFAULT_SCREENSHOT_DIR,
    format_browser_acceptance_report,
    run_browser_acceptance,
)


def main() -> int:
    parser = argparse.ArgumentParser(description="Run DocFlow browser acceptance checks.")
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL, help="Running DocFlow web app URL")
    parser.add_argument(
        "--screenshots-dir",
        default=str(DEFAULT_SCREENSHOT_DIR),
        help="Directory where browser screenshots are written",
    )
    parser.add_argument("--no-screenshots", action="store_true", help="Do not write screenshots")
    parser.add_argument("--headed", action="store_true", help="Run browser with a visible window")
    parser.add_argument("--timeout-ms", type=int, default=8000, help="Per-check timeout in milliseconds")
    parser.add_argument("--with-mutation-flow", action="store_true", help="Also create, query, and clean up a temporary note")
    parser.add_argument("--json", action="store_true", help="Emit JSON only")
    args = parser.parse_args()

    report = run_browser_acceptance(
        base_url=args.base_url,
        screenshot_dir=None if args.no_screenshots else args.screenshots_dir,
        headless=not args.headed,
        timeout_ms=args.timeout_ms,
        include_mutation_flow=args.with_mutation_flow,
    )
    if args.json:
        print(json.dumps(report, ensure_ascii=False, indent=2))
    else:
        print(format_browser_acceptance_report(report))
    return 0 if report["failed"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
