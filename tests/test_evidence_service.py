from __future__ import annotations

import os
import time

import pytest

from src.api.services.evidence_service import EvidenceService


@pytest.mark.parametrize(
    ("score", "snippet", "expected"),
    [
        (0.99, "highly relevant source text", "strong"),
        (0.75, "threshold source text", "strong"),
        (0.74, "nearly strong source text", "medium"),
        (0.61, "medium source text", "medium"),
        (0.45, "medium threshold source text", "medium"),
        (0.44, "weak threshold source text", "weak"),
        (0.2, "weak source text", "weak"),
        (0.0, "zero score source text", "weak"),
        (1.0, "", "medium"),
        (0.8, None, "medium"),
        (0.46, "", "medium"),
        (0.1, "", "weak"),
    ],
)
def test_evidence_level_cases(score, snippet, expected):
    citation = EvidenceService().enrich_citations(
        [{"file_name": "source.md", "snippet": snippet, "score": score}]
    )[0]

    assert citation["evidence_level"] == expected
    assert citation["evidence_label"]
    assert citation["evidence_reason"]


@pytest.mark.parametrize(
    ("positive", "negative", "conflict_type"),
    [
        ("approved", "rejected", "status"),
        ("enabled", "disabled", "status"),
        ("true", "false", "status"),
        ("批准", "拒绝", "status"),
        ("开启", "关闭", "status"),
        ("increase", "decrease", "direction"),
        ("higher", "lower", "direction"),
        ("提升", "下降", "direction"),
        ("latest", "deprecated", "freshness"),
        ("当前", "过期", "freshness"),
    ],
)
def test_conflict_detection_cases(positive, negative, conflict_type):
    conflicts = EvidenceService().detect_conflicts(
        [
            {"file_name": "left.md", "snippet": f"the plan is {positive}"},
            {"file_name": "right.md", "snippet": f"the plan is {negative}"},
        ]
    )

    assert conflicts
    assert conflicts[0]["type"] == conflict_type
    assert "left.md" in conflicts[0]["files"]
    assert "right.md" in conflicts[0]["files"]


def test_no_conflict_when_opposing_words_are_in_same_file():
    conflicts = EvidenceService().detect_conflicts(
        [
            {"file_name": "same.md", "snippet": "approved"},
            {"file_name": "same.md", "snippet": "rejected"},
        ]
    )

    assert conflicts == []


def test_fresh_source_is_marked_current(tmp_path):
    path = tmp_path / "fresh.md"
    path.write_text("fresh", encoding="utf-8")

    citation = EvidenceService().enrich_citations(
        [{"file_name": path.name, "file_path": str(path), "snippet": "fresh", "score": 0.9}]
    )[0]

    assert citation["freshness"] == "current"
    assert citation["source_age_days"] == 0


def test_stale_source_is_marked_for_review(tmp_path):
    path = tmp_path / "stale.md"
    path.write_text("stale", encoding="utf-8")
    old = time.time() - 400 * 86400
    os.utime(path, (old, old))

    citation = EvidenceService().enrich_citations(
        [{"file_name": path.name, "file_path": str(path), "snippet": "stale", "score": 0.9}]
    )[0]

    assert citation["freshness"] == "stale"
    assert citation["source_age_days"] >= 399
    assert "来源较旧" in citation["evidence_reason"]


@pytest.mark.parametrize(
    ("citations", "expected"),
    [
        ([], "none"),
        ([{"evidence_level": "weak", "snippet": "thin"}], "weak"),
        ([{"evidence_level": "medium", "snippet": "ok"}], "medium"),
        ([{"evidence_level": "strong", "snippet": "solid"}], "strong"),
        (
            [
                {"file_name": "a.md", "evidence_level": "strong", "snippet": "approved"},
                {"file_name": "b.md", "evidence_level": "strong", "snippet": "rejected"},
            ],
            "conflict",
        ),
    ],
)
def test_evidence_summary_cases(citations, expected):
    summary = EvidenceService().summarize(citations)

    assert summary["level"] == expected
    assert summary["label"]
    assert summary["summary"]
