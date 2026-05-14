from __future__ import annotations

import logging
import time as _time

from src.query.constants import QUERY_INSTRUCTION
from src.query.router import QueryRouter

logger = logging.getLogger(__name__)


def debug_retrieve(
    retriever,
    query: str,
    file_filter: list[str] | None = None,
    retrieval_mode: str = "hybrid",
    prefer_tables: bool = False,
    include_rerank: bool = True,
    max_text_chars: int = 300,
) -> dict:
    timings: dict[str, float] = {}
    degradations: list[dict] = []
    t0 = _time.perf_counter()

    route = QueryRouter.classify(query)
    search_limit = route["top_k_retrieval"]
    rerank_limit = route["top_k_rerank"]
    mode = retriever._normalize_retrieval_mode(retrieval_mode)

    query_vec_list: list[float] | None = None
    if mode != "full_text":
        instructed_query = f"Instruct: {QUERY_INSTRUCTION}\nQuery: {query}"
        try:
            query_vec = retriever.embed_model.encode(
                [instructed_query],
                normalize_embeddings=True,
                convert_to_numpy=True,
            )[0]
            query_vec_list = query_vec.tolist() if hasattr(query_vec, "tolist") else list(query_vec)
        except Exception as exc:
            logger.warning("[retriever] debug vector encode failed", exc_info=True)
            degradations.append(retriever._degradation("vector", exc))
    timings["embed_ms"] = round((_time.perf_counter() - t0) * 1000, 2)

    t1 = _time.perf_counter()
    vec_results: list[dict] = []
    if query_vec_list is not None:
        try:
            vec_results = retriever._vector_search(query_vec_list, file_filter, limit=search_limit)
        except Exception as exc:
            logger.warning("[retriever] debug vector search failed", exc_info=True)
            degradations.append(retriever._degradation("vector", exc))
    timings["vector_ms"] = round((_time.perf_counter() - t1) * 1000, 2)

    t1 = _time.perf_counter()
    fts_results: list[dict] = []
    try:
        fts_results = retriever._fts_search(
            query,
            file_filter,
            limit=search_limit,
            degradations=degradations,
        )
    except Exception as exc:
        logger.warning("[retriever] debug FTS search failed", exc_info=True)
        degradations.append(retriever._degradation("fts", exc))
    timings["fts_ms"] = round((_time.perf_counter() - t1) * 1000, 2)

    fused = retriever._rrf_fuse(
        vec_results,
        fts_results,
        prefer_tables=prefer_tables,
        vec_weight=route["vec_weight"],
        bm25_weight=route["bm25_weight"],
    )
    top_candidates = fused[:search_limit]
    deduped = retriever._deduplicate(top_candidates)

    reranked: list[dict] = []
    if include_rerank and deduped:
        t1 = _time.perf_counter()
        reranked = retriever._rerank_or_fallback(
            query,
            [dict(item) for item in deduped],
            rerank_limit,
            degradations,
        )
        parent_expanded = retriever._expand_parent_context([dict(item) for item in reranked])
        timings["rerank_ms"] = round((_time.perf_counter() - t1) * 1000, 2)
    else:
        timings["rerank_ms"] = 0.0
        parent_expanded = retriever._expand_parent_context([dict(item) for item in deduped])
    retriever._attach_degradations(parent_expanded, degradations)

    timings["total_ms"] = round((_time.perf_counter() - t0) * 1000, 2)
    return {
        "query": query,
        "file_filter": file_filter or [],
        "retrieval_mode": mode,
        "prefer_tables": prefer_tables,
        "router": route,
        "top_k_retrieval": search_limit,
        "top_k_rerank": rerank_limit,
        "status": "degraded" if degradations else "ok",
        "degradations": degradations,
        "timings": timings,
        "stages": {
            "vector": [debug_item(item, max_text_chars) for item in vec_results],
            "fts": [debug_item(item, max_text_chars) for item in fts_results],
            "fused": [debug_item(item, max_text_chars) for item in top_candidates],
            "deduped": [debug_item(item, max_text_chars) for item in deduped],
            "reranked": [debug_item(item, max_text_chars) for item in reranked],
            "parent_expanded": [debug_item(item, max_text_chars) for item in parent_expanded],
        },
    }


def debug_item(item: dict, max_text_chars: int = 300) -> dict:
    text = item.get("text", "") or ""
    return {
        "qdrant_id": item.get("qdrant_id"),
        "file_name": item.get("file_name", ""),
        "file_path": item.get("file_path", ""),
        "page_num": item.get("page_num", 0),
        "section": item.get("section", ""),
        "chunk_type": item.get("chunk_type", ""),
        "score": item.get("score"),
        "vec_score": item.get("vec_score"),
        "rrf_score": item.get("rrf_score"),
        "rerank_score": item.get("rerank_score"),
        "char_count": item.get("char_count", len(text)),
        "parent_id": item.get("parent_id", 0),
        "parent_text_length": item.get(
            "parent_text_length",
            len(item.get("parent_text", "") or ""),
        ),
        "retrieval_status": item.get("retrieval_status", "ok"),
        "rerank_fallback": item.get("rerank_fallback", False),
        "matched_text_preview": (item.get("matched_text") or item.get("child_text") or "")[
            :max_text_chars
        ],
        "text_preview": text[:max_text_chars],
        "text_length": len(text),
    }
