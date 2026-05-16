"""Typed query settings and default retrieval thresholds."""

from __future__ import annotations

from dataclasses import dataclass, field

TABLE_KEYWORDS = {
    "表格",
    "数据",
    "统计",
    "总计",
    "合计",
    "金额",
    "数量",
    "比例",
    "table",
    "data",
    "total",
    "sum",
    "amount",
    "count",
    "ratio",
    "percent",
}

INSUFFICIENT_EVIDENCE_MESSAGE = (
    "在现有文档中未找到足够可靠的信息。请扩大提问范围、换个问法，或确认相关文件已经完成入库。"
)
MIN_RERANK_SCORE = 0.12
MIN_VECTOR_SCORE = 0.40
RELATED_NOTES_LIMIT = 4
DEFAULT_ANSWER_CHUNKS = 5
MIN_ANSWER_CHUNKS = 3
MAX_RESEARCH_STEPS = 4


@dataclass(frozen=True)
class QuerySettings:
    min_rerank_score: float = MIN_RERANK_SCORE
    min_vector_score: float = MIN_VECTOR_SCORE
    related_notes_limit: int = RELATED_NOTES_LIMIT
    default_answer_chunks: int = DEFAULT_ANSWER_CHUNKS
    min_answer_chunks: int = MIN_ANSWER_CHUNKS
    max_research_steps: int = MAX_RESEARCH_STEPS
    table_keywords: frozenset[str] = field(default_factory=lambda: frozenset(TABLE_KEYWORDS))

    @classmethod
    def from_config(cls, cfg: dict) -> QuerySettings:
        query_cfg = cfg.get("query", {})
        table_keywords = query_cfg.get("table_keywords")
        return cls(
            min_rerank_score=float(query_cfg.get("min_rerank_score", MIN_RERANK_SCORE)),
            min_vector_score=float(query_cfg.get("min_vector_score", MIN_VECTOR_SCORE)),
            related_notes_limit=int(query_cfg.get("related_notes_limit", RELATED_NOTES_LIMIT)),
            default_answer_chunks=int(
                query_cfg.get("default_answer_chunks", DEFAULT_ANSWER_CHUNKS)
            ),
            min_answer_chunks=int(query_cfg.get("min_answer_chunks", MIN_ANSWER_CHUNKS)),
            max_research_steps=int(query_cfg.get("max_research_steps", MAX_RESEARCH_STEPS)),
            table_keywords=frozenset(table_keywords or TABLE_KEYWORDS),
        )
