from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from statistics import mean


@dataclass(frozen=True)
class MaturityDimension:
    id: str
    name: str
    target_score: float
    current_score: float
    phase: str
    evidence: list[str]
    next_steps: list[str]

    @property
    def gap(self) -> float:
        return round(max(self.target_score - self.current_score, 0), 2)

    @property
    def status(self) -> str:
        if self.current_score >= self.target_score:
            return "at_target"
        if self.current_score >= self.target_score - 1:
            return "near_target"
        return "below_target"


def load_dimensions(path: str | Path) -> list[MaturityDimension]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    dimensions = data.get("dimensions", data)
    return [
        MaturityDimension(
            id=item["id"],
            name=item["name"],
            target_score=float(item.get("target_score", 9)),
            current_score=float(item["current_score"]),
            phase=item.get("phase", ""),
            evidence=list(item.get("evidence", [])),
            next_steps=list(item.get("next_steps", [])),
        )
        for item in dimensions
    ]


def summarize_dimensions(dimensions: list[MaturityDimension]) -> dict:
    if not dimensions:
        return {
            "dimensions": 0,
            "overall_score": 0.0,
            "target_score": 9.0,
            "at_target": 0,
            "near_target": 0,
            "below_target": 0,
            "largest_gaps": [],
        }
    target_score = mean(d.target_score for d in dimensions)
    rows = [dimension_to_dict(d) for d in dimensions]
    return {
        "dimensions": len(dimensions),
        "overall_score": round(mean(d.current_score for d in dimensions), 2),
        "target_score": round(target_score, 2),
        "at_target": sum(1 for d in dimensions if d.status == "at_target"),
        "near_target": sum(1 for d in dimensions if d.status == "near_target"),
        "below_target": sum(1 for d in dimensions if d.status == "below_target"),
        "largest_gaps": sorted(rows, key=lambda row: row["gap"], reverse=True)[:5],
    }


def dimension_to_dict(dimension: MaturityDimension) -> dict:
    return {
        "id": dimension.id,
        "name": dimension.name,
        "target_score": dimension.target_score,
        "current_score": dimension.current_score,
        "gap": dimension.gap,
        "status": dimension.status,
        "phase": dimension.phase,
        "evidence": dimension.evidence,
        "next_steps": dimension.next_steps,
    }


def build_report(dimensions: list[MaturityDimension], retrieval_eval: dict | None = None) -> dict:
    return {
        "schema": "docflow.maturity.v1",
        "summary": summarize_dimensions(dimensions),
        "dimensions": [dimension_to_dict(d) for d in dimensions],
        "retrieval_eval": retrieval_eval,
    }


def format_report(report: dict) -> str:
    summary = report["summary"]
    lines = [
        "DocFlow maturity baseline",
        f"Overall score: {summary['overall_score']}/{summary['target_score']}",
        f"Dimensions: {summary['dimensions']} total, {summary['at_target']} at target, "
        f"{summary['near_target']} near target, {summary['below_target']} below target",
        "",
        "Dimension scores:",
    ]
    for item in report["dimensions"]:
        lines.append(
            f"- {item['name']}: {item['current_score']}/{item['target_score']} "
            f"(gap {item['gap']}, {item['phase']})"
        )
    if report.get("retrieval_eval") is not None:
        retrieval = report["retrieval_eval"]
        lines.extend(
            [
                "",
                "Retrieval evidence eval:",
                f"- cases: {retrieval['cases']}, passed: {retrieval['passed']}, failed: {retrieval['failed']}",
                f"- include_rerank: {retrieval['include_rerank']}",
                f"- source_filter: {retrieval.get('source_filter', False)}",
            ]
        )
        for result in retrieval["results"]:
            mark = "PASS" if result["passed"] else "FAIL"
            reason = result.get("failure_reason") or "ok"
            lines.append(f"  [{mark}] {result['id']} :: {result['evidence_status']} :: {reason}")
    return "\n".join(lines)
