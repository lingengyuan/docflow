from __future__ import annotations

import pytest

from src.api.services.evidence_service import EvidenceService
from src.api.services.query_service import QueryService
from src.query.answer_quality import grounded_quality, quality_with_claim_support
from src.query.claim_support import (
    audit_answer_claim_support,
    meaningful_terms,
    source_support_score,
    split_answer_claims,
)
from src.query.generator import Citation


def test_claim_support_marks_fully_cited_answer_supported():
    citations = [Citation(file_name="alpha.md", page_num=1, snippet="Alpha 结论成立", score=0.9)]

    audit = audit_answer_claim_support("Alpha 结论成立 [来源: alpha.md, 第1页]。", citations)

    assert audit["level"] == "supported"
    assert audit["total_claims"] == 1
    assert audit["supported_claims"] == 1
    assert audit["coverage"] == 1.0


def test_claim_support_flags_uncited_claims():
    citations = [Citation(file_name="alpha.md", page_num=1, snippet="Alpha 结论成立", score=0.9)]

    audit = audit_answer_claim_support(
        "Alpha 结论成立 [来源: alpha.md, 第1页]。Beta 结论缺少来源。",
        citations,
    )

    assert audit["level"] == "partial"
    assert audit["total_claims"] == 2
    assert audit["supported_claims"] == 1
    assert audit["unsupported_claims"] == 1
    assert audit["unsupported_examples"] == ["Beta 结论缺少来源。"]


def test_claim_support_flags_unverified_source_markers():
    audit = audit_answer_claim_support("编造结论 [未验证来源]。", [])

    assert audit["level"] == "unsupported"
    assert audit["unverified_claims"] == 1
    assert audit["unverified_examples"] == ["编造结论 [未验证来源]。"]


def test_claim_support_flags_verified_marker_with_wrong_source_text():
    citations = [
        Citation(
            file_name="alpha.md",
            page_num=1,
            snippet="Alpha roadmap approved local indexing",
            score=0.9,
        )
    ]

    audit = audit_answer_claim_support(
        "Beta retention policy blocks cloud sync [来源: alpha.md, 第1页]。",
        citations,
    )

    assert audit["level"] == "unsupported"
    assert audit["weak_source_claims"] == 1
    assert audit["claims"][0]["status"] == "weak_source"


def test_source_support_score_uses_shared_content_terms():
    score = source_support_score(
        "Local indexing keeps private notes searchable [来源: alpha.md, 第1页]。",
        [
            Citation(
                file_name="alpha.md",
                page_num=1,
                snippet="Private notes stay searchable through local indexing.",
                score=0.9,
            )
        ],
    )

    assert score["supported"] is True
    assert {"local", "indexing", "private", "searchable"} & set(score["shared_terms"])


def test_meaningful_terms_ignores_generic_words():
    terms = meaningful_terms("This source claim says fact 42 about local indexing.")

    assert "source" not in terms
    assert "claim" not in terms
    assert "fact" not in terms
    assert "local" in terms
    assert "indexing" in terms


def test_claim_support_ignores_no_answer_message():
    audit = audit_answer_claim_support("在现有文档中未找到相关信息。", [])

    assert audit["level"] == "none"
    assert audit["total_claims"] == 0
    assert audit["coverage"] == 1.0


def test_split_answer_claims_handles_markdown_lists():
    claims = split_answer_claims("- 第一条结论 [来源: a.md, 第1页]。\n2. 第二条结论。")

    assert claims == ["第一条结论 [来源: a.md, 第1页]。", "第二条结论。"]


def test_claim_support_downgrades_answer_quality_when_needed():
    claim_support = audit_answer_claim_support("缺少来源的结论。", [])

    quality = quality_with_claim_support(grounded_quality(), claim_support)

    assert quality["status"] == "citation_needs_review"
    assert quality["claim_support"]["unsupported_claims"] == 1
    assert "缺少来源" in quality["label"]


def test_evidence_summary_uses_claim_support_gap():
    claim_support = audit_answer_claim_support("缺少来源的结论。", [])
    summary = EvidenceService().summarize(
        [{"file_name": "alpha.md", "evidence_level": "strong", "snippet": "alpha"}],
        claim_support=claim_support,
    )

    assert summary["level"] == "weak"
    assert summary["label"] == "部分结论缺少来源"
    assert summary["recommendations"][0].startswith("先打开来源")


def test_stream_finalize_returns_claim_support_quality():
    chunks = [
        {
            "text": "Alpha fact",
            "file_name": "alpha.md",
            "file_path": "/tmp/alpha.md",
            "page_num": 1,
            "qdrant_id": 1,
            "rerank_score": 0.9,
        }
    ]

    answer, citations, quality = QueryService().finalize_stream_answer_with_quality(
        "Alpha [[cite:q:1]]。Beta missing。",
        chunks,
        grounded_quality(),
    )

    assert answer == "Alpha [来源: alpha.md, 第1页]。Beta missing。"
    assert [citation["chunk_id"] for citation in citations] == ["q:1"]
    assert quality["status"] == "citation_needs_review"
    assert quality["claim_support"]["unsupported_claims"] == 1


@pytest.mark.parametrize("index", range(50))
def test_claim_support_rejects_wrong_source_content_matrix(index):
    citations = [
        Citation(
            file_name=f"source_{index}.md",
            page_num=1,
            snippet=f"Alpha roadmap approval keeps local index stable for team {index}.",
            score=0.9,
        )
    ]

    audit = audit_answer_claim_support(
        f"Beta retention policy blocks cloud sync path [来源: source_{index}.md, 第1页]。",
        citations,
    )

    assert audit["weak_source_claims"] == 1
    assert audit["supported_claims"] == 0
    assert audit["coverage"] == 0.0
