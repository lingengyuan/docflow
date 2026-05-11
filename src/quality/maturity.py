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


def build_report(
    dimensions: list[MaturityDimension],
    retrieval_eval: dict | None = None,
    parsing_eval: dict | None = None,
) -> dict:
    return {
        "schema": "docflow.maturity.v1",
        "summary": summarize_dimensions(dimensions),
        "measurements": summarize_measurements(retrieval_eval, parsing_eval),
        "dimensions": [dimension_to_dict(d) for d in dimensions],
        "retrieval_eval": retrieval_eval,
        "parsing_eval": parsing_eval,
    }


def summarize_measurements(
    retrieval_eval: dict | None = None,
    parsing_eval: dict | None = None,
) -> dict:
    signals: list[dict] = []
    if retrieval_eval is not None:
        metrics = retrieval_eval.get("metrics", {})
        signals.append(
            {
                "id": "retrieval_eval",
                "name": "Retrieval eval",
                "passed": retrieval_eval.get("failed", 0) == 0,
                "cases": retrieval_eval.get("cases", 0),
                "passed_cases": retrieval_eval.get("passed", 0),
                "failed_cases": retrieval_eval.get("failed", 0),
                "metrics": metrics,
            }
        )
    if parsing_eval is not None:
        signals.append(
            {
                "id": "parsing_eval",
                "name": "Parsing eval",
                "passed": parsing_eval.get("failed", 0) == 0,
                "cases": parsing_eval.get("cases", 0),
                "passed_cases": parsing_eval.get("passed", 0),
                "failed_cases": parsing_eval.get("failed", 0),
                "metrics": {
                    "pass_rate": _ratio(parsing_eval.get("passed", 0), parsing_eval.get("cases", 0))
                },
            }
        )
    return {
        "signals": signals,
        "passed_signals": sum(1 for signal in signals if signal["passed"]),
        "failed_signals": sum(1 for signal in signals if not signal["passed"]),
    }


def _ratio(numerator: int, denominator: int) -> float:
    return 0.0 if denominator == 0 else round(numerator / denominator, 4)


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
    measurements = report.get("measurements", {})
    signals = measurements.get("signals", [])
    if signals:
        lines.extend(
            [
                "",
                "Measured signals:",
            ]
        )
        for signal in signals:
            mark = "PASS" if signal["passed"] else "FAIL"
            metric_text = _format_metric_text(signal.get("metrics", {}))
            suffix = f" ({metric_text})" if metric_text else ""
            lines.append(
                f"- [{mark}] {signal['name']}: "
                f"{signal['passed_cases']}/{signal['cases']} cases{suffix}"
            )
        lines.append("")
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
                f"- cases: {retrieval['cases']}, passed: {retrieval['passed']}, "
                f"failed: {retrieval['failed']}",
                f"- include_rerank: {retrieval['include_rerank']}",
                f"- source_filter: {retrieval.get('source_filter', False)}",
            ]
        )
        for result in retrieval["results"]:
            mark = "PASS" if result["passed"] else "FAIL"
            reason = result.get("failure_reason") or "ok"
            lines.append(f"  [{mark}] {result['id']} :: {result['evidence_status']} :: {reason}")
    if report.get("parsing_eval") is not None:
        parsing = report["parsing_eval"]
        lines.extend(
            [
                "",
                "Parsing eval:",
                f"- cases: {parsing['cases']}, passed: {parsing['passed']}, "
                f"failed: {parsing['failed']}",
            ]
        )
    return "\n".join(lines)


def _format_metric_text(metrics: dict) -> str:
    parts = []
    for key in ("recall_at_5", "mrr_at_5", "ndcg_at_5", "pass_rate"):
        if key in metrics:
            parts.append(f"{key}={metrics[key]}")
    return ", ".join(parts)
