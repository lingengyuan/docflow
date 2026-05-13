"""Rule-based query routing for hybrid retrieval."""

from __future__ import annotations

import logging
import re

logger = logging.getLogger(__name__)


class QueryRouter:
    """
    根据查询特征动态调整 BM25（FTS5）与向量检索的权重。

    关键词信号（引号短语、日期、扩展名）→ 偏向 FTS5
    长语义查询（无关键词信号）→ 偏向向量
    默认 → 均衡
    """

    _KEYWORD_PATTERNS = [
        re.compile(r'"[^"]+"'),  # "精确短语"
        re.compile(r"\b\d{4}[-/]\d{1,2}"),  # 日期 2024-01
        re.compile(r"\.\w{2,4}\b"),  # 文件扩展名 .pdf
        re.compile(r"[A-Z]{2,}\d+"),  # 编号 INV2024
        re.compile(
            r"[\u4e00-\u9fff].*\b(?:[A-Z]{2,}|[A-Za-z]+[-_][A-Za-z0-9_-]+|"
            r"[A-Za-z]*\d[A-Za-z0-9_-]*|fallback|reranked|backup|export-chunks|restore-plan)\b"
            r"|\b(?:[A-Z]{2,}|[A-Za-z]+[-_][A-Za-z0-9_-]+|"
            r"[A-Za-z]*\d[A-Za-z0-9_-]*|fallback|reranked|backup|export-chunks|restore-plan)\b"
            r".*[\u4e00-\u9fff]"
        ),
    ]
    _CROSS_DOC_PATTERNS = [
        re.compile(r"对比|比较|差异|区别|汇总|总结|跨文档|多个文件"),
        re.compile(r"\b(compare|comparison|differences?|summari[sz]e|across files?)\b", re.I),
    ]

    @classmethod
    def classify(cls, query: str) -> dict:
        signals = sum(1 for p in cls._KEYWORD_PATTERNS if p.search(query))
        is_cross_doc = any(p.search(query) for p in cls._CROSS_DOC_PATTERNS)
        if signals >= 2:
            route = {
                "query_type": "exact",
                "bm25_weight": 2.0,
                "vec_weight": 0.5,
                "top_k_retrieval": 12,
                "top_k_rerank": 3,
            }
        elif is_cross_doc:
            route = {
                "query_type": "cross_document",
                "bm25_weight": 1.0,
                "vec_weight": 1.5,
                "top_k_retrieval": 30,
                "top_k_rerank": 10,
            }
        elif signals == 1:
            route = {
                "query_type": "keyword",
                "bm25_weight": 1.5,
                "vec_weight": 1.0,
                "top_k_retrieval": 24,
                "top_k_rerank": 6,
            }
        elif len(query) > 20 and signals == 0:
            route = {
                "query_type": "semantic",
                "bm25_weight": 0.5,
                "vec_weight": 2.0,
                "top_k_retrieval": 24,
                "top_k_rerank": 8,
            }
        else:
            route = {
                "query_type": "balanced",
                "bm25_weight": 1.0,
                "vec_weight": 1.0,
                "top_k_retrieval": 20,
                "top_k_rerank": 5,
            }
        logger.debug(f"[router] query={query[:40]!r} signals={signals} route={route}")
        return route

