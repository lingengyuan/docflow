from __future__ import annotations

from src.api.services.evidence_service import EvidenceService
from src.api.services.query_service import QueryService
from src.query.answer_quality import grounded_quality, quality_with_claim_support
from src.query.claim_support import audit_answer_claim_support, split_answer_claims
from src.query.generator import Citation


def test_claim_support_marks_fully_cited_answer_supported():
    citations = [Citation(file_name="alpha.md", page_num=1, snippet="alpha", score=0.9)]

    audit = audit_answer_claim_support("Alpha 结论成立 [来源: alpha.md, 第1页]。", citations)

    assert audit["level"] == "supported"
    assert audit["total_claims"] == 1
    assert audit["supported_claims"] == 1
    assert audit["coverage"] == 1.0


def test_claim_support_flags_uncited_claims():
    citations = [Citation(file_name="alpha.md", page_num=1, snippet="alpha", score=0.9)]

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
