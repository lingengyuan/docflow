#!/usr/bin/env python3
"""Run the Phase 22 backup restore drill."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.maintenance.backup import DEFAULT_RESTORE_DRILL_DIR, restore_drill


def main() -> int:
    parser = argparse.ArgumentParser(description="Run DocFlow backup restore drill checks.")
    parser.add_argument(
        "--output-dir",
        default=str(DEFAULT_RESTORE_DRILL_DIR),
        help="Disposable directory for backup and extraction outputs",
    )
    parser.add_argument("--keep", type=int, default=2, help="Backup archives to retain")
    parser.add_argument("--json", action="store_true", help="Emit JSON only")
    args = parser.parse_args()

    report = restore_drill(output_dir=args.output_dir, keep=args.keep)
    if args.json:
        print(json.dumps(report, ensure_ascii=False, indent=2))
    else:
        print(f"DocFlow restore drill: {report['passed']}/{len(report['checks'])} passed")
        print(f"Output dir: {report['output_dir']}")
        print(f"Archive: {report['archive']}")
        for check in report["checks"]:
            mark = "PASS" if check["passed"] else "FAIL"
            print(f"[{mark}] {check['id']}")
    return 0 if report["status"] == "ok" else 1


if __name__ == "__main__":
    raise SystemExit(main())
