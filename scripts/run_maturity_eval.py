#!/usr/bin/env python3
"""Generate DocFlow's internal planning baseline."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.quality.maturity import build_report, format_report, load_dimensions  # noqa: E402

DEFAULT_DIMENSIONS = Path("eval/internal_quality_baseline_dimensions.json")
DEFAULT_CASES = Path("eval/qa_v1.jsonl")


def configure_output_encoding() -> None:
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is not None:
            reconfigure(encoding="utf-8")


def run_retrieval_eval(
    config: str,
    cases_path: Path,
    include_rerank: bool,
    refresh_sources: bool = False,
    source_filter: bool = False,
) -> dict:
    from scripts.run_eval import evaluate_case, load_cases, refresh_eval_sources, retrieval_metrics
    from src.query.engine import QueryEngine

    cases = load_cases(cases_path)
    source_refresh = refresh_eval_sources(cases, config) if refresh_sources else None
    engine = QueryEngine.from_config(config)
    results = [
        evaluate_case(
            engine,
            case,
            include_rerank=include_rerank,
            source_filter=source_filter,
        )
        for case in cases
    ]
    passed = sum(1 for result in results if result["passed"])
    report = {
        "schema": "docflow.retrieval_eval.v1",
        "cases": len(results),
        "passed": passed,
        "failed": len(results) - passed,
        "include_rerank": include_rerank,
        "source_filter": source_filter,
        "case_file": str(cases_path),
        "metrics": retrieval_metrics(results),
        "results": results,
    }
    if source_refresh is not None:
        report["source_refresh"] = source_refresh
    return report


def main() -> int:
    configure_output_encoding()
    parser = argparse.ArgumentParser(
        description=(
            "Run DocFlow's internal planning baseline. This is not release readiness "
            "and must not be used as a public quality claim."
        )
    )
    parser.add_argument("--config", default="config.yaml", help="Path to config.yaml")
    parser.add_argument(
        "--dimensions",
        default=str(DEFAULT_DIMENSIONS),
        help="Internal planning dimension JSON file",
    )
    parser.add_argument(
        "--cases", default=str(DEFAULT_CASES), help="Retrieval evidence eval JSONL file"
    )
    parser.add_argument(
        "--no-rerank", action="store_true", help="Skip reranker in retrieval evidence eval"
    )
    parser.add_argument(
        "--skip-retrieval", action="store_true", help="Skip retrieval evidence checks"
    )
    parser.add_argument(
        "--skip-parsing", action="store_true", help="Skip parsing regression checks"
    )
    parser.add_argument(
        "--parsing-config",
        default="config.example.yaml",
        help="Config used for parsing regression checks",
    )
    parser.add_argument(
        "--refresh-sources",
        action="store_true",
        help="Ingest expected source files before running retrieval evidence checks",
    )
    parser.add_argument(
        "--source-filter",
        action="store_true",
        help="For must-find cases, restrict retrieval to expected source files",
    )
    parser.add_argument("--json", action="store_true", help="Emit JSON only")
    args = parser.parse_args()

    dimensions = load_dimensions(args.dimensions)
    retrieval_eval = None
    parsing_eval = None
    if not args.skip_retrieval:
        try:
            retrieval_eval = run_retrieval_eval(
                args.config,
                Path(args.cases),
                include_rerank=not args.no_rerank,
                refresh_sources=args.refresh_sources,
                source_filter=args.source_filter,
            )
        except Exception as exc:
            print(
                "Maturity eval failed during retrieval evidence checks. "
                "Check that Qdrant is running and the expected documents are ingested.",
                file=sys.stderr,
            )
            print(f"{type(exc).__name__}: {exc}", file=sys.stderr)
            return 2
    if not args.skip_parsing:
        try:
            from src.quality.parsing_eval import run_parsing_eval

            parsing_eval = run_parsing_eval(config_path=args.parsing_config)
        except Exception as exc:
            print("Maturity eval failed during parsing regression checks.", file=sys.stderr)
            print(f"{type(exc).__name__}: {exc}", file=sys.stderr)
            return 2

    report = build_report(dimensions, retrieval_eval=retrieval_eval, parsing_eval=parsing_eval)
    if args.json:
        print(json.dumps(report, ensure_ascii=False, indent=2))
    else:
        print(format_report(report))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
