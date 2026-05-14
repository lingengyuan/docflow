from __future__ import annotations

from src.api.services.query_service import QueryService


def test_finalize_stream_answer_filters_to_structured_cited_chunk():
    chunks = [
        {
            "text": "Alpha fact",
            "file_name": "alpha.md",
            "file_path": "/tmp/alpha.md",
            "page_num": 1,
            "qdrant_id": 1,
            "rerank_score": 0.9,
        },
        {
            "text": "Beta fact",
            "file_name": "beta.md",
            "file_path": "/tmp/beta.md",
            "page_num": 2,
            "qdrant_id": 2,
            "rerank_score": 0.8,
        },
    ]

    answer, citations = QueryService().finalize_stream_answer("Alpha [[cite:q:1]]。", chunks)

    assert answer == "Alpha [来源: alpha.md, 第1页]。"
    assert [citation["chunk_id"] for citation in citations] == ["q:1"]


def test_finalize_stream_answer_marks_fabricated_structured_chunk():
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

    answer, citations = QueryService().finalize_stream_answer("Claim [[cite:q:999]]。", chunks)

    assert answer == "Claim [未验证来源]。"
    assert citations == []
