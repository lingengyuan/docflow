from __future__ import annotations

import json

from scripts.run_eval import EvalCase, evaluate_case
from src.quality.maturity import build_report, load_dimensions, summarize_dimensions


def test_maturity_summary_tracks_gaps(tmp_path):
    path = tmp_path / "dimensions.json"
    path.write_text(
        json.dumps(
            {
                "dimensions": [
                    {
                        "id": "a",
                        "name": "A",
                        "target_score": 9,
                        "current_score": 8,
                        "phase": "Phase X",
                    },
                    {
                        "id": "b",
                        "name": "B",
                        "target_score": 9,
                        "current_score": 5,
                        "phase": "Phase Y",
                    },
                ]
            }
        ),
        encoding="utf-8",
    )

    dimensions = load_dimensions(path)
    summary = summarize_dimensions(dimensions)

    assert summary["overall_score"] == 6.5
    assert summary["near_target"] == 1
    assert summary["below_target"] == 1
    assert summary["largest_gaps"][0]["id"] == "b"


def test_maturity_report_includes_retrieval_eval():
    report = build_report(
        [],
        retrieval_eval={
            "cases": 1,
            "passed": 1,
            "failed": 0,
            "include_rerank": False,
            "results": [],
        },
    )

    assert report["schema"] == "docflow.maturity.v1"
    assert report["retrieval_eval"]["cases"] == 1


class FakeRetriever:
    def __init__(self, final_stage):
        self.final_stage = final_stage

    def debug_retrieve(self, *args, **kwargs):
        return {
            "stages": {
                "deduped": self.final_stage,
                "reranked": self.final_stage,
            },
            "timings": {"total_ms": 1.0},
        }


class FakeEngine:
    def __init__(self, final_stage):
        self.retriever = FakeRetriever(final_stage)

    @staticmethod
    def _is_table_query(question):
        return False


def test_eval_case_reports_grounded_evidence():
    case = EvalCase(
        id="readme",
        category="qa_loop",
        question="What does DocFlow support?",
        expected_files=["README.md"],
        expected_terms=["Markdown"],
        must_find=True,
    )
    engine = FakeEngine(
        [
            {
                "file_name": "README.md",
                "file_path": "/tmp/README.md",
                "section": "Features",
                "text_preview": "DocFlow supports Markdown files.",
                "qdrant_id": 1,
            }
        ]
    )

    result = evaluate_case(engine, case, include_rerank=False)

    assert result["passed"] is True
    assert result["evidence_status"] == "grounded"
    assert result["failure_reason"] == ""
    assert result["top_sources"][0]["file_name"] == "README.md"


def test_eval_case_reports_correct_no_match():
    case = EvalCase(
        id="cloud",
        question="Does it support cloud collaboration?",
        expected_files=[],
        expected_terms=["cloud collaboration"],
        must_find=False,
    )
    engine = FakeEngine([])

    result = evaluate_case(engine, case, include_rerank=False)

    assert result["passed"] is True
    assert result["evidence_status"] == "correctly_no_match"

