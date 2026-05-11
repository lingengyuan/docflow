#!/usr/bin/env python3
"""Run DocFlow parsing regression checks."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.quality.parsing_eval import (  # noqa: E402
    DEFAULT_CORPUS_DIR,
    DEFAULT_EXPECTED_PATH,
    run_parsing_eval,
)

DEFAULT_RESULTS_DIR = Path("eval/results")


def current_git_sha() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=PROJECT_ROOT,
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except Exception:
        return "unknown"


def write_results(report: dict, results_dir: Path = DEFAULT_RESULTS_DIR) -> Path:
    git_sha = report.get("git_sha") or current_git_sha()
    results_dir.mkdir(parents=True, exist_ok=True)
    output_path = results_dir / f"parsing-{git_sha}.json"
    payload = json.dumps(report, ensure_ascii=False, indent=2)
    output_path.write_text(payload, encoding="utf-8")
    (results_dir / "parsing-latest.json").write_text(payload, encoding="utf-8")
    return output_path


def main() -> int:
    parser = argparse.ArgumentParser(description="Run DocFlow parsing regression checks.")
    parser.add_argument("--config", default="config.example.yaml", help="Path to config file")
    parser.add_argument(
        "--corpus", default=str(DEFAULT_CORPUS_DIR), help="Parsing corpus directory"
    )
    parser.add_argument(
        "--expected", default=str(DEFAULT_EXPECTED_PATH), help="Expected checks JSON"
    )
    parser.add_argument(
        "--write-results", action="store_true", help="Write output under eval/results"
    )
    parser.add_argument(
        "--results-dir", default=str(DEFAULT_RESULTS_DIR), help="Directory for --write-results"
    )
    parser.add_argument("--json", action="store_true", help="Emit JSON only")
    args = parser.parse_args()

    report = run_parsing_eval(args.corpus, args.expected, args.config)
    report["generated_at"] = datetime.now(UTC).isoformat()
    report["git_sha"] = current_git_sha()
    if args.write_results:
        report["results_path"] = str(write_results(report, Path(args.results_dir)))

    if args.json:
        print(json.dumps(report, ensure_ascii=False, indent=2))
    else:
        print(f"DocFlow parsing eval: {report['passed']}/{report['cases']} passed")
        if report.get("results_path"):
            print(f"Results written: {report['results_path']}")
        for result in report["results"]:
            mark = "PASS" if result["passed"] else "FAIL"
            print(f"[{mark}] {result['id']} :: {result['path']}")
            if not result["passed"]:
                print(f"  failures={result['failures']}")
    return 0 if report["failed"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
