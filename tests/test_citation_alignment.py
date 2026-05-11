from __future__ import annotations

import pytest

from src.query.generator import (
    Citation,
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
