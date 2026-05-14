#!/usr/bin/env python3
"""Run DocFlow's committed public-domain retrieval regression benchmark."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import UTC, datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.run_eval import (  # noqa: E402
    DEFAULT_RESULTS_DIR,
    current_git_sha,
    evaluate_case,
    load_cases,
    performance_summary,
    refresh_eval_sources,
    retrieval_metrics,
    write_results,
)
from src.query.engine import QueryEngine  # noqa: E402
from src.resources import resource_path  # noqa: E402

DEFAULT_PUBLIC_CASES = resource_path("eval", "public_retrieval_v1.jsonl")
DEFAULT_PUBLIC_RESULTS_DIR = DEFAULT_RESULTS_DIR / "public"


def public_benchmark_metadata(cases_path: Path) -> dict:
    return {
        "name": "DocFlow public-domain retrieval regression",
        "kind": "public_reproducible_regression",
        "cases_file": str(cases_path),
        "corpus_dir": "eval/public_corpus",
        "source_filter": False,
        "scope_note": "Committed public-domain regression set; not a BEIR, MTEB, or C-MTEB score.",
    }


def build_public_summary(
    *,
    cases_path: Path,
    results: list[dict],
    include_rerank: bool,
    source_refresh: dict | None = None,
) -> dict:
    passed = sum(1 for result in results if result["passed"])
    summary = {
        "schema": "docflow.public_retrieval_eval.v1",
        "generated_at": datetime.now(UTC).isoformat(),
        "git_sha": current_git_sha(),
        "benchmark": public_benchmark_metadata(cases_path),
        "cases": len(results),
        "passed": passed,
        "failed": len(results) - passed,
        "include_rerank": include_rerank,
        "metrics": retrieval_metrics(results),
        "performance": performance_summary(results),
        "results": results,
    }
    if source_refresh is not None:
        summary["source_refresh"] = source_refresh
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description="Run DocFlow public-domain retrieval eval.")
    parser.add_argument("--config", default="config.yaml", help="Path to config.yaml")
    parser.add_argument("--cases", default=str(DEFAULT_PUBLIC_CASES), help="JSONL case file")
    parser.add_argument("--no-rerank", action="store_true", help="Skip reranker")
    parser.add_argument(
        "--no-refresh-sources",
        action="store_true",
        help="Skip ingesting public corpus sources before running checks",
    )
    parser.add_argument(
        "--write-results",
        action="store_true",
        help="Write eval output to eval/results/public/<git-sha>.json",
    )
    parser.add_argument(
        "--results-dir",
        default=str(DEFAULT_PUBLIC_RESULTS_DIR),
        help="Directory for --write-results",
    )
    parser.add_argument("--json", action="store_true", help="Emit JSON only")
    args = parser.parse_args()

    cases_path = Path(args.cases)
    cases = load_cases(cases_path)
    source_refresh = None
    try:
        if not args.no_refresh_sources:
            source_refresh = refresh_eval_sources(cases, args.config)
        engine = QueryEngine.from_config(args.config)
        results = [
            evaluate_case(
                engine,
                case,
                include_rerank=not args.no_rerank,
                source_filter=False,
            )
            for case in cases
        ]
    except Exception as exc:
        print("Public eval failed before completion.", file=sys.stderr)
        print(f"{type(exc).__name__}: {exc}", file=sys.stderr)
        return 2

    summary = build_public_summary(
        cases_path=cases_path,
        results=results,
        include_rerank=not args.no_rerank,
        source_refresh=source_refresh,
    )
    if args.write_results:
        summary["results_path"] = str(write_results(summary, Path(args.results_dir)))

    if args.json:
        print(json.dumps(summary, ensure_ascii=False, indent=2))
    else:
        print(f"DocFlow public retrieval eval: {summary['passed']}/{summary['cases']} passed")
        metrics = summary["metrics"]
        print(
            "Metrics: "
            f"Recall@5={metrics['recall_at_5']} "
            f"MRR@5={metrics['mrr_at_5']} "
            f"nDCG@5={metrics['ndcg_at_5']}"
        )
        if summary.get("results_path"):
            print(f"Results written: {summary['results_path']}")
        for result in results:
            mark = "PASS" if result["passed"] else "FAIL"
            print(f"[{mark}] {result['id']} :: {result['question']}")

    return 0 if summary["failed"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
