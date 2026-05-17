#!/usr/bin/env python3
"""Run deterministic answer faithfulness checks from committed fixtures."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.api.services.evidence_service import EvidenceService  # noqa: E402
from src.query.answer_quality import grounded_quality, quality_with_claim_support  # noqa: E402
from src.query.claim_support import audit_answer_claim_support  # noqa: E402
from src.resources import resource_path  # noqa: E402

DEFAULT_CASES = resource_path("eval", "answer_faithfulness_v1.jsonl")
DEFAULT_RESULTS_DIR = Path("eval/results/faithfulness")


def current_git_sha() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=PROJECT_ROOT,
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def load_cases(path: Path) -> list[dict[str, Any]]:
    cases: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            if "id" not in row or "answer" not in row:
                raise ValueError(f"Invalid faithfulness case at line {line_no}")
            cases.append(row)
    if not cases:
        raise ValueError(f"No faithfulness cases found in {path}")
    return cases


def evaluate_cases(cases: list[dict[str, Any]]) -> dict[str, Any]:
    results = []
    evidence_service = EvidenceService()
    for case in cases:
        audit = audit_answer_claim_support(case["answer"], case.get("citations", []))
        quality = quality_with_claim_support(grounded_quality(), audit)
        evidence = evidence_service.summarize(case.get("citations", []), claim_support=audit)
        expected = {
            "level": case.get("expected_level"),
            "unsupported_claims": int(case.get("expected_unsupported", 0)),
            "unverified_claims": int(case.get("expected_unverified", 0)),
            "weak_source_claims": int(case.get("expected_weak", 0)),
            "quality_status": case.get(
                "expected_quality_status",
                (
                    "grounded"
                    if case.get("expected_level") in {"supported", "none"}
                    else "citation_needs_review"
                ),
            ),
            "evidence_level": case.get(
                "expected_evidence_level",
                "none" if not case.get("citations") else None,
            ),
        }
        actual = {
            "level": audit["level"],
            "unsupported_claims": audit["unsupported_claims"],
            "unverified_claims": audit["unverified_claims"],
            "weak_source_claims": audit["weak_source_claims"],
            "quality_status": quality["status"],
            "evidence_level": evidence["level"],
        }
        if expected["evidence_level"] is None:
            expected["evidence_level"] = actual["evidence_level"]
        expected_quality_label = case.get("expected_quality_label")
        expected_evidence_label = case.get("expected_evidence_label")
        expected_recommendation_contains = case.get("expected_recommendation_contains")
        expected_conflict_type = case.get("expected_conflict_type")
        passed = actual == expected
        for expected_key, actual_value in (
            ("expected_total_claims", audit["total_claims"]),
            ("expected_supported", audit["supported_claims"]),
        ):
            if expected_key in case:
                passed = passed and int(case[expected_key]) == actual_value
        if "expected_coverage" in case:
            passed = passed and float(case["expected_coverage"]) == audit["coverage"]
        if expected_quality_label is not None:
            passed = passed and quality.get("label") == expected_quality_label
        if expected_evidence_label is not None:
            passed = passed and evidence.get("label") == expected_evidence_label
        if expected_recommendation_contains is not None:
            recommendations = " ".join(str(item) for item in evidence.get("recommendations", []))
            passed = passed and str(expected_recommendation_contains) in recommendations
        if expected_conflict_type is not None:
            conflict_types = {str(item.get("type") or "") for item in evidence.get("conflicts", [])}
            passed = passed and str(expected_conflict_type) in conflict_types
        results.append(
            {
                "id": case["id"],
                "passed": passed,
                "expected": expected,
                "actual": actual,
                "quality": {
                    "status": quality["status"],
                    "label": quality.get("label"),
                    "reason": quality.get("reason"),
                },
                "evidence": {
                    "level": evidence["level"],
                    "label": evidence.get("label"),
                    "summary": evidence.get("summary"),
                    "recommendations": evidence.get("recommendations", []),
                    "conflicts": evidence.get("conflicts", []),
                },
                "coverage": audit["coverage"],
                "total_claims": audit["total_claims"],
            }
        )
    passed = sum(1 for item in results if item["passed"])
    return {
        "schema": "docflow.answer_faithfulness_eval.v1",
        "generated_at": datetime.now(UTC).isoformat(),
        "git_sha": current_git_sha(),
        "source_tree": _source_tree_state(),
        "cases": len(results),
        "passed": passed,
        "failed": len(results) - passed,
        "results": results,
    }


def write_results(report: dict[str, Any], results_dir: Path = DEFAULT_RESULTS_DIR) -> Path:
    results_dir.mkdir(parents=True, exist_ok=True)
    output_path = results_dir / f"faithfulness-{report['git_sha']}.json"
    payload = json.dumps(report, ensure_ascii=False, indent=2)
    output_path.write_text(payload, encoding="utf-8")
    (results_dir / "faithfulness-latest.json").write_text(payload, encoding="utf-8")
    return output_path


def _source_tree_state() -> dict[str, Any]:
    try:
        status = subprocess.check_output(
            ["git", "status", "--porcelain"],
            cwd=PROJECT_ROOT,
            text=True,
            stderr=subprocess.DEVNULL,
        ).splitlines()
    except (OSError, subprocess.CalledProcessError):
        return {"worktree_dirty": None, "note": "Git status unavailable."}
    return {
        "worktree_dirty": bool(status),
        "status_entries": len(status),
        "note": (
            "Result was generated from the current local source tree. "
            "If worktree_dirty is true, the artifact records a pre-commit working-tree run."
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Run DocFlow answer faithfulness eval.")
    parser.add_argument("--cases", default=str(DEFAULT_CASES))
    parser.add_argument("--write-results", action="store_true")
    parser.add_argument("--results-dir", default=str(DEFAULT_RESULTS_DIR))
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    try:
        report = evaluate_cases(load_cases(Path(args.cases)))
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        print(f"Faithfulness eval failed: {exc}", file=sys.stderr)
        return 2
    if args.write_results:
        report["results_path"] = str(write_results(report, Path(args.results_dir)))

    if args.json:
        print(json.dumps(report, ensure_ascii=False, indent=2))
    else:
        print(f"DocFlow faithfulness eval: {report['passed']}/{report['cases']} passed")
        if report.get("results_path"):
            print(f"Results written: {report['results_path']}")
    return 0 if report["failed"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
