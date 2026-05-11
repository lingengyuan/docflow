"""Query and conversation helpers for API handlers."""

from __future__ import annotations

import json
import re


class QueryService:
    def build_retrieval_query(self, question: str, conversation_context: list[dict]) -> str:
        if not self.looks_like_followup(question):
            return question
        previous_user_questions = [
            message["content"]
            for message in conversation_context
            if message.get("role") == "user" and message.get("content")
        ]
        if not previous_user_questions:
            return question
        return f"{previous_user_questions[-1]}\n{question}"

    def looks_like_followup(self, question: str) -> bool:
        q = question.strip().lower()
        chinese_markers = (
            "展开",
            "继续",
            "上面",
            "刚才",
            "前面",
            "这个",
            "那个",
            "这点",
            "第二点",
            "第三点",
            "第一点",
        )
        if any(marker in q for marker in chinese_markers):
            return True
        english_markers = ("it", "that", "this", "above", "previous")
        return any(re.search(rf"\b{re.escape(marker)}\b", q) for marker in english_markers)

    def response_citations(self, citations) -> list[dict]:
        seen_chunks: dict[str, dict] = {}
        for citation in citations:
            key = citation.chunk_id or citation.file_path or citation.file_name
            if key not in seen_chunks or citation.score > seen_chunks[key]["score"]:
                seen_chunks[key] = {
                    "file_name": citation.file_name,
                    "file_path": citation.file_path,
                    "page_num": citation.page_num,
                    "section": citation.section,
                    "snippet": citation.snippet,
                    "score": round(citation.score, 4),
                    "chunk_id": citation.chunk_id,
                    "document_id": citation.document_id,
                    "qdrant_id": citation.qdrant_id,
                    "char_start": citation.char_start,
                    "char_end": citation.char_end,
                }
        return list(seen_chunks.values())

    def stream_citations(self, chunks: list[dict]) -> list[dict]:
        seen_chunks: dict[str, dict] = {}
        for chunk in chunks:
            qdrant_id = chunk.get("qdrant_id")
            chunk_id = str(
                chunk.get("chunk_id") or (f"q:{qdrant_id}" if qdrant_id is not None else "")
            )
            key = chunk_id or chunk.get("file_path") or chunk["file_name"]
            score = chunk.get("rerank_score", chunk.get("rrf_score", 0.0))
            matched_text = (
                chunk.get("matched_text")
                or chunk.get("child_text")
                or chunk.get("raw_text")
                or chunk.get("text", "")
            )
            parent_text = chunk.get("text") or chunk.get("parent_text") or matched_text
            char_start = parent_text.find(matched_text) if matched_text else 0
            if char_start < 0:
                char_start = 0
            if key not in seen_chunks or score > seen_chunks[key]["score"]:
                seen_chunks[key] = {
                    "file_name": chunk["file_name"],
                    "file_path": chunk.get("file_path", ""),
                    "page_num": chunk["page_num"],
                    "section": chunk.get("section", ""),
                    "snippet": matched_text[:200],
                    "score": round(score, 4),
                    "chunk_id": chunk_id,
                    "document_id": str(
                        chunk.get("document_id") or chunk.get("file_path") or chunk["file_name"]
                    ),
                    "qdrant_id": int(qdrant_id) if qdrant_id is not None else None,
                    "char_start": char_start,
                    "char_end": char_start + len(matched_text),
                }
        return list(seen_chunks.values())

    def decode_history_items(self, items: list[dict]) -> list[dict]:
        for item in items:
            try:
                item["citations"] = json.loads(item["citations"])
            except Exception:
                item["citations"] = []
            try:
                item["file_filter"] = json.loads(item["file_filter"])
            except Exception:
                item["file_filter"] = []
        return items
