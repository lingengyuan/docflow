"""Typed query quality settings."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from src.config_defaults import INSUFFICIENT_EVIDENCE_MESSAGE, TABLE_KEYWORDS


@dataclass(frozen=True)
class QuerySettings:
    seed: int = 42
    temperature: float = 0.0
    top_p: float = 1.0
    max_tokens: int = 2048
    min_rerank_score: float = 0.12
    min_vector_score: float = 0.40
    related_notes_limit: int = 4
    default_answer_chunks: int = 5
    min_answer_chunks: int = 3
    max_research_steps: int = 4
    table_keywords: frozenset[str] = field(default_factory=lambda: frozenset(TABLE_KEYWORDS))
    insufficient_evidence_message: str = INSUFFICIENT_EVIDENCE_MESSAGE
    fallback_mode: str = "visible_snippet_fallback"

    @classmethod
    def from_mapping(cls, cfg: dict[str, Any]) -> QuerySettings:
        query_cfg = cfg.get("query", {})
        table_keywords = query_cfg.get("table_keywords")
        return cls(
            seed=int(query_cfg.get("seed", 42)),
            temperature=float(query_cfg.get("temperature", 0.0)),
            top_p=float(query_cfg.get("top_p", 1.0)),
            max_tokens=int(query_cfg.get("max_tokens", 2048)),
            min_rerank_score=float(query_cfg.get("min_rerank_score", 0.12)),
            min_vector_score=float(query_cfg.get("min_vector_score", 0.40)),
            related_notes_limit=int(query_cfg.get("related_notes_limit", 4)),
            default_answer_chunks=int(query_cfg.get("default_answer_chunks", 5)),
            min_answer_chunks=int(query_cfg.get("min_answer_chunks", 3)),
            max_research_steps=int(query_cfg.get("max_research_steps", 4)),
            table_keywords=frozenset(str(value) for value in (table_keywords or TABLE_KEYWORDS)),
            insufficient_evidence_message=str(
                query_cfg.get("insufficient_evidence_message", INSUFFICIENT_EVIDENCE_MESSAGE)
            ),
            fallback_mode=str(query_cfg.get("fallback_mode", "visible_snippet_fallback")),
        )
