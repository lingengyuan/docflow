from __future__ import annotations

import pytest

from src.query.generator import (
    AnswerGenerator,
    Citation,
    apply_structured_citations,
    citation_from_chunk,
    sanitize_inline_citations,
    validate_citations,
)


def test_citation_from_chunk_carries_stable_coordinates():
    chunk = {
        "qdrant_id": 42,
        "file_name": "notes.md",
        "file_path": "/tmp/notes.md",
        "page_num": 1,
        "section": "Plan",
        "matched_text": "budget approved",
        "text": "context before budget approved context after",
        "rerank_score": 0.88,
    }

    citation = citation_from_chunk(chunk)

    assert citation.chunk_id == "q:42"
    assert citation.document_id == "/tmp/notes.md"
    assert citation.qdrant_id == 42
    assert citation.char_start == 15
    assert citation.char_end == 30


def test_validate_citations_drops_unretrieved_chunk_ids():
    chunks = [{"qdrant_id": 1, "file_name": "a.md", "page_num": 1, "text": "a"}]
    valid = Citation(file_name="a.md", page_num=1, snippet="a", score=1.0, chunk_id="q:1")
    invalid = Citation(file_name="b.md", page_num=1, snippet="b", score=1.0, chunk_id="q:2")

    assert validate_citations([valid, invalid], chunks) == [valid]


def test_context_includes_structured_chunk_citation_hint():
    context = AnswerGenerator._build_context(
        [
            {
                "qdrant_id": 17,
                "file_name": "source.md",
                "page_num": 3,
                "text": "trusted fact",
            }
        ]
    )

    assert "chunk_id: q:17" in context
    assert "引用格式: [[cite:q:17]]" in context


def test_structured_citations_keep_only_used_verified_chunks():
    citations = [
        Citation(file_name="a.md", page_num=1, snippet="alpha", score=0.9, chunk_id="q:1"),
        Citation(file_name="b.md", page_num=2, snippet="beta", score=0.8, chunk_id="q:2"),
    ]

    cleaned, used = apply_structured_citations("只引用第一段 [[cite:q:1]]。", citations)

    assert cleaned == "只引用第一段 [来源: a.md, 第1页]。"
    assert used == [citations[0]]


def test_structured_citations_mark_invalid_chunk_ids_unverified():
    citations = [
        Citation(file_name="a.md", page_num=1, snippet="alpha", score=0.9, chunk_id="q:1")
    ]

    cleaned, used = apply_structured_citations("编造引用 [[cite:q:404]]。", citations)

    assert cleaned == "编造引用 [未验证来源]。"
    assert used == []


def test_structured_citations_keep_valid_ids_in_first_use_order():
    citations = [
        Citation(file_name="a.md", page_num=1, snippet="alpha", score=0.9, chunk_id="q:1"),
        Citation(file_name="b.md", page_num=2, snippet="beta", score=0.8, chunk_id="q:2"),
        Citation(file_name="c.md", page_num=3, snippet="gamma", score=0.7, chunk_id="q:3"),
    ]

    cleaned, used = apply_structured_citations(
        "第二段 [[cite:q:2]]，编造 [[cite:q:404]]，第一段 [[cite:q:1]]。",
        citations,
    )

    assert cleaned == "第二段 [来源: b.md, 第2页]，编造 [未验证来源]，第一段 [来源: a.md, 第1页]。"
    assert used == [citations[1], citations[0]]


def test_structured_citations_dedupe_repeated_verified_ids():
    citation = Citation(file_name="a.md", page_num=1, snippet="alpha", score=0.9, chunk_id="q:1")

    cleaned, used = apply_structured_citations(
        "第一处 [[cite:q:1]]，第二处 [[cite:q:1]]。",
        [citation],
    )

    assert cleaned == "第一处 [来源: a.md, 第1页]，第二处 [来源: a.md, 第1页]。"
    assert used == [citation]


def test_structured_citations_preserve_legacy_inline_answers():
    citations = [
        Citation(file_name="a.md", page_num=1, snippet="alpha", score=0.9, chunk_id="q:1"),
        Citation(file_name="b.md", page_num=2, snippet="beta", score=0.8, chunk_id="q:2"),
    ]
    answer = "旧格式仍可用 [来源: a.md, 第1页]。"

    cleaned, used = apply_structured_citations(answer, citations)

    assert cleaned == answer
    assert used == citations


def test_unverified_inline_model_citations_are_marked():
    citations = [
        Citation(
            file_name="trusted.md",
            page_num=2,
            snippet="trusted",
            score=0.9,
            chunk_id="q:7",
        )
    ]
    answer = "可信结论 [来源: trusted.md, 第2页]。编造结论 [来源: fake.pdf, 第8页]。"

    cleaned = sanitize_inline_citations(answer, citations)

    assert "[来源: trusted.md, 第2页]" in cleaned
    assert "[来源: fake.pdf, 第8页]" not in cleaned
    assert "[未验证来源]" in cleaned


def test_generator_records_claim_support_quality():
    class FakeGenerator(AnswerGenerator):
        def _call_ollama_with_system(self, system_prompt: str, user_msg: str) -> str:
            return "可信结论 [[cite:q:1]]。缺少来源的结论。"

    answer = FakeGenerator().generate(
        "question",
        [
            {
                "qdrant_id": 1,
                "file_name": "trusted.md",
                "page_num": 2,
                "text": "可信结论",
                "rerank_score": 0.9,
            }
        ],
    )

    assert answer.text == "可信结论 [来源: trusted.md, 第2页]。缺少来源的结论。"
    assert answer.quality["claim_support"]["level"] == "partial"
    assert answer.quality["claim_support"]["unsupported_claims"] == 1


@pytest.mark.parametrize(
    ("file_name", "page_num"),
    [
        ("alpha.md", 1),
        ("alpha.md", 2),
        ("beta.pdf", 1),
        ("beta.pdf", 8),
        ("notes.txt", 3),
        ("plan.docx", 4),
        ("code.py", 5),
        ("image-notes.md", 6),
        ("meeting.md", 7),
        ("research.pdf", 9),
        ("summary.md", 10),
        ("draft.txt", 11),
        ("local-privacy.md", 12),
        ("workflow.md", 13),
        ("source.md", 14),
        ("guide.md", 15),
        ("table.md", 16),
        ("scan.pdf", 17),
        ("mixed.docx", 18),
        ("final.md", 19),
    ],
)
def test_inline_citation_validation_covers_many_verified_sources(file_name, page_num):
    citations = [
        Citation(
            file_name=file_name,
            page_num=page_num,
            snippet="verified",
            score=0.8,
            chunk_id=f"q:{page_num}",
        )
    ]
    answer = f"结论 [来源: {file_name}, 第{page_num}页]。"

    assert sanitize_inline_citations(answer, citations) == answer
