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
            "展开", "继续", "上面", "刚才", "前面", "这个", "那个", "这点", "第二点",
            "第三点", "第一点",
        )
        if any(marker in q for marker in chinese_markers):
            return True
        english_markers = ("it", "that", "this", "above", "previous")
        return any(re.search(rf"\b{re.escape(marker)}\b", q) for marker in english_markers)

    def response_citations(self, citations) -> list[dict]:
        seen_files: dict[str, dict] = {}
        for citation in citations:
            key = citation.file_path or citation.file_name
            if key not in seen_files or citation.score > seen_files[key]["score"]:
                seen_files[key] = {
                    "file_name": citation.file_name,
                    "file_path": citation.file_path,
                    "page_num": citation.page_num,
                    "section": citation.section,
                    "snippet": citation.snippet,
                    "score": round(citation.score, 4),
                }
        return list(seen_files.values())

    def stream_citations(self, chunks: list[dict]) -> list[dict]:
        seen_files: dict[str, dict] = {}
        for chunk in chunks:
            key = chunk.get("file_path") or chunk["file_name"]
            score = chunk.get("rerank_score", chunk.get("rrf_score", 0.0))
            if key not in seen_files or score > seen_files[key]["score"]:
                seen_files[key] = {
                    "file_name": chunk["file_name"],
                    "file_path": chunk.get("file_path", ""),
                    "page_num": chunk["page_num"],
                    "section": chunk.get("section", ""),
                    "snippet": chunk["text"][:200],
                    "score": round(score, 4),
                }
        return list(seen_files.values())

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
