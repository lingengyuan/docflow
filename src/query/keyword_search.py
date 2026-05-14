from __future__ import annotations

import logging

from src.query.constants import COLLECTION_NAME

logger = logging.getLogger(__name__)


def tokenize(text: str) -> list[str]:
    import jieba

    return [t for t in jieba.cut(text.lower()) if t.strip()]


def fts_search(
    retriever,
    query: str,
    file_filter: list[str] | None,
    limit: int | None = None,
    degradations: list[dict] | None = None,
) -> list[dict]:
    limit = limit or retriever.top_k_retrieval
    tokens = tokenize(query)
    rows: list[dict] = []

    if tokens:
        escaped = [t.replace('"', "") for t in tokens]
        fts_query = " OR ".join(f'"{t}"' for t in escaped if t)
        if fts_query:
            try:
                rows = retriever._store.search_fts(fts_query, file_filter, limit=limit)
            except Exception as exc:
                logger.warning("[retriever] FTS5 exact search failed: %s", exc, exc_info=True)

    if not rows:
        logger.debug("[retriever] FTS5 exact empty -> trigram fallback")
        try:
            rows = retriever._store.search_fts_trigram(query, file_filter, limit=limit)
        except Exception as exc:
            logger.warning("[retriever] FTS5 trigram search failed: %s", exc, exc_info=True)

    if not rows:
        return []

    top_ids = [r["qdrant_id"] for r in rows]
    score_map = {r["qdrant_id"]: r["score"] for r in rows}
    row_payloads = {r["qdrant_id"]: fts_row_item(r) for r in rows}

    try:
        fetched = retriever._qdrant.retrieve(
            collection_name=COLLECTION_NAME,
            ids=top_ids,
            with_payload=True,
        )
        id_to_payload = {p.id: p.payload for p in fetched}
    except Exception as exc:
        logger.warning(
            "[retriever] Qdrant payload fetch failed; using SQLite FTS payloads", exc_info=True
        )
        if degradations is not None:
            degradations.append(retriever._degradation("fts_payload", exc))
        id_to_payload = {}

    results = []
    for qid in top_ids:
        payload = {**row_payloads.get(qid, {}), **(id_to_payload.get(qid) or {})}
        if not payload:
            continue
        results.append({"qdrant_id": qid, "score": score_map[qid], **payload})
    return results


def fts_row_item(row: dict) -> dict:
    raw_text = row.get("raw_text", "") or ""
    return {
        "file_name": row.get("file_name", ""),
        "file_path": row.get("file_path", ""),
        "page_num": row.get("page_num", 0),
        "section": row.get("section", ""),
        "chunk_type": row.get("chunk_type", ""),
        "text": raw_text,
        "raw_text": raw_text,
        "child_text": raw_text,
        "parent_id": row.get("parent_id", 0),
        "parent_text": row.get("parent_text", ""),
        "contextual_prefix": row.get("contextual_prefix", ""),
        "char_count": row.get("char_count", len(raw_text)),
    }
