#!/usr/bin/env python3
"""Run the Phase 21 real sample suite."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.quality.sample_suite import DEFAULT_SAMPLE_DIR, run_sample_suite  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description="Run DocFlow generated real sample checks.")
    parser.add_argument(
        "--output-dir",
        default=str(DEFAULT_SAMPLE_DIR),
        help="Directory where generated sample files and outputs are written",
    )
    parser.add_argument("--json", action="store_true", help="Emit JSON only")
    args = parser.parse_args()

    report = run_sample_suite(args.output_dir)
    if args.json:
        print(json.dumps(report, ensure_ascii=False, indent=2))
    else:
        print(f"DocFlow sample suite: {report['passed']}/{len(report['checks'])} passed")
        print(f"Output dir: {report['output_dir']}")
        for check in report["checks"]:
            mark = "PASS" if check["passed"] else "FAIL"
            reason = "" if check["passed"] else f" :: {check.get('error', '')}"
            print(f"[{mark}] {check['id']}{reason}")
    return 0 if report["failed"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
