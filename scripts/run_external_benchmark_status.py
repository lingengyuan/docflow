#!/usr/bin/env python3
"""Report DocFlow external benchmark readiness and claim boundaries."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CATALOG = PROJECT_ROOT / "eval" / "external_benchmarks.json"


def load_catalog(path: Path = DEFAULT_CATALOG) -> dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if data.get("schema") != "docflow.external_benchmark_catalog.v1":
        raise ValueError(f"Unexpected external benchmark catalog schema in {path}")
    benchmarks = data.get("benchmarks")
    if not isinstance(benchmarks, list) or not benchmarks:
        raise ValueError("External benchmark catalog must list at least one benchmark")
    required_ids = {"beir", "mteb", "cmteb"}
    found_ids = {str(item.get("id")) for item in benchmarks if isinstance(item, dict)}
    missing = sorted(required_ids - found_ids)
    if missing:
        raise ValueError(f"External benchmark catalog is missing: {', '.join(missing)}")
    for item in benchmarks:
        if not isinstance(item, dict):
            raise ValueError("External benchmark entries must be objects")
        for key in ("id", "name", "scope", "primary_source", "docflow_status"):
            if not item.get(key):
                raise ValueError(f"Benchmark entry is missing {key}: {item!r}")
        if item["docflow_status"] not in {"not_run", "running", "archived"}:
            raise ValueError(f"Invalid docflow_status for {item['id']}: {item['docflow_status']}")
    return data


def summary(catalog: dict[str, Any]) -> dict[str, Any]:
    benchmarks = catalog["benchmarks"]
    benchmark_summaries = []
    archived_score_count = 0
    for item in benchmarks:
        benchmark: dict[str, Any] = {
            "id": item["id"],
            "name": item["name"],
            "status": item["docflow_status"],
            "source": item["primary_source"],
        }
        archived_results = item.get("archived_results")
        if isinstance(archived_results, list) and archived_results:
            benchmark["archived_results"] = archived_results
            archived_score_count += len(archived_results)
        elif item["docflow_status"] == "archived":
            archived_score_count += 1
        for optional in (
            "archived_result",
            "archived_scope",
            "archived_model_config",
            "archived_runtime",
            "archived_metrics",
            "claim_note",
        ):
            if optional in item:
                benchmark[optional] = item[optional]
        benchmark_summaries.append(benchmark)

    return {
        "schema": catalog["schema"],
        "updated_at": catalog.get("updated_at"),
        "external_benchmark_scores": archived_score_count,
        "benchmarks": benchmark_summaries,
        "claim_policy": catalog.get("claim_policy", []),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Show DocFlow external benchmark status.")
    parser.add_argument("--catalog", default=str(DEFAULT_CATALOG))
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    try:
        report = summary(load_catalog(Path(args.catalog)))
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        print(f"External benchmark status check failed: {exc}", file=sys.stderr)
        return 1

    if args.json:
        print(json.dumps(report, ensure_ascii=False, indent=2))
    else:
        print("DocFlow external benchmark status")
        for item in report["benchmarks"]:
            print(f"- {item['name']}: {item['status']} ({item['source']})")
        print(f"Archived external scores: {report['external_benchmark_scores']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
