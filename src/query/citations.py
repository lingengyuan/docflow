"""Citation structures and validation helpers."""

from __future__ import annotations

import re
from dataclasses import dataclass


@dataclass
class Citation:
    file_name: str
    page_num: int
    snippet: str
    score: float
    file_path: str = ""
    section: str = ""
    chunk_id: str = ""
    document_id: str = ""
    qdrant_id: int | None = None
    char_start: int = 0
    char_end: int = 0


def citation_from_chunk(chunk: dict) -> Citation:
    qdrant_id = chunk.get("qdrant_id")
    chunk_id = str(chunk.get("chunk_id") or (f"q:{qdrant_id}" if qdrant_id is not None else ""))
    document_id = str(
        chunk.get("document_id") or chunk.get("file_path") or chunk.get("file_name") or ""
    )
    matched_text = (
        chunk.get("matched_text")
        or chunk.get("child_text")
        or chunk.get("raw_text")
        or chunk.get("text")
        or ""
    )
    parent_text = chunk.get("text") or chunk.get("parent_text") or matched_text
    char_start = parent_text.find(matched_text) if matched_text else 0
    if char_start < 0:
        char_start = 0
    char_end = char_start + len(matched_text)
    return Citation(
        file_name=chunk["file_name"],
        file_path=chunk.get("file_path", ""),
        page_num=chunk["page_num"],
        snippet=matched_text[:200],
        score=chunk.get("rerank_score", chunk.get("rrf_score", chunk.get("score", 0.0))),
        section=chunk.get("section", ""),
        chunk_id=chunk_id,
        document_id=document_id,
        qdrant_id=int(qdrant_id) if qdrant_id is not None else None,
        char_start=char_start,
        char_end=char_end,
    )


def validate_citations(citations: list[Citation], chunks: list[dict]) -> list[Citation]:
    valid_chunk_ids = {
        str(
            chunk.get("chunk_id")
            or (f"q:{chunk.get('qdrant_id')}" if chunk.get("qdrant_id") is not None else "")
        )
        for chunk in chunks
    }
    valid_chunk_ids.discard("")
    return [
        citation
        for citation in citations
        if not citation.chunk_id or citation.chunk_id in valid_chunk_ids
    ]


INLINE_CITATION_RE = re.compile(r"\[来源:\s*([^,\]，]+)(?:[,，]\s*第?(\d+)页)?\]")
STRUCTURED_CITATION_RE = re.compile(r"\[\[cite:([^\]]+)\]\]")


def apply_structured_citations(text: str, citations: list[Citation]) -> tuple[str, list[Citation]]:
    citation_by_id = {citation.chunk_id: citation for citation in citations if citation.chunk_id}
    used_ids: list[str] = []

    def replace(match: re.Match[str]) -> str:
        chunk_id = match.group(1).strip()
        citation = citation_by_id.get(chunk_id)
        if citation is None:
            return "[未验证来源]"
        if chunk_id not in used_ids:
            used_ids.append(chunk_id)
        return f"[来源: {citation.file_name}, 第{citation.page_num}页]"

    cleaned = STRUCTURED_CITATION_RE.sub(replace, text)
    if not used_ids and cleaned == text:
        return text, citations
    used_citations = [
        citation_by_id[chunk_id] for chunk_id in used_ids if chunk_id in citation_by_id
    ]
    return cleaned, used_citations


def sanitize_inline_citations(text: str, citations: list[Citation]) -> str:
    verified = {
        (citation.file_name, str(citation.page_num))
        for citation in citations
        if citation.file_name and citation.page_num
    }
    verified_files = {citation.file_name for citation in citations if citation.file_name}

    def replace(match: re.Match[str]) -> str:
        file_name = match.group(1).strip()
        page_num = (match.group(2) or "").strip()
        if (file_name, page_num) in verified or (not page_num and file_name in verified_files):
            return match.group(0)
        return "[未验证来源]"

    return INLINE_CITATION_RE.sub(replace, text)
