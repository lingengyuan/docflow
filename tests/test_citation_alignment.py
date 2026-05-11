from __future__ import annotations

from src.query.generator import Citation, citation_from_chunk, validate_citations


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
