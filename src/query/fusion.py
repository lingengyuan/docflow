from __future__ import annotations

from typing import Any


def rrf_fuse(
    vec_results: list[dict],
    fts_results: list[dict],
    k: int = 60,
    prefer_tables: bool = False,
    vec_score_threshold: float = 0.3,
    vec_weight: float = 1.0,
    bm25_weight: float = 1.0,
) -> list[dict]:
    scores: dict[int, float] = {}
    id_to_item: dict[int, dict] = {}
    vec_scores: dict[int, float] = {}

    for rank, item in enumerate(vec_results):
        qid = item["qdrant_id"]
        scores[qid] = scores.get(qid, 0.0) + vec_weight / (k + rank + 1)
        id_to_item[qid] = item
        vec_scores[qid] = item["score"]

    for rank, item in enumerate(fts_results):
        qid = item["qdrant_id"]
        scores[qid] = scores.get(qid, 0.0) + bm25_weight / (k + rank + 1)
        if qid not in id_to_item:
            id_to_item[qid] = item

    if prefer_tables:
        for qid, item in id_to_item.items():
            if item.get("chunk_type") in ("table", "table_summary"):
                scores[qid] *= 1.5

    sorted_ids = sorted(scores, key=lambda x: scores[x], reverse=True)
    results = []
    for qid in sorted_ids:
        vs = vec_scores.get(qid, 0.0)
        if qid in vec_scores and vs < vec_score_threshold:
            continue
        results.append({**id_to_item[qid], "rrf_score": scores[qid], "vec_score": vs})
    return results


def deduplicate(candidates: list[dict]) -> list[dict]:
    seen: dict[tuple, dict] = {}
    for item in candidates:
        parent_id = item.get("parent_id", 0)
        if parent_id:
            key: tuple[Any, ...] = (item.get("file_path", ""), parent_id)
        else:
            key = (
                item.get("file_path", ""),
                item.get("page_num", 0),
                item.get("text", "")[:128],
            )
        if key not in seen or item.get("rrf_score", 0) > seen[key].get("rrf_score", 0):
            seen[key] = item
    return list(seen.values())


def expand_parent_context(retriever, candidates: list[dict]) -> list[dict]:
    if not candidates:
        return []

    qdrant_ids = [
        int(item["qdrant_id"]) for item in candidates if item.get("qdrant_id") is not None
    ]
    try:
        stored_contexts = retriever._store.get_chunk_context_by_qdrant_ids(qdrant_ids)
    except Exception as exc:
        retriever._logger.debug("[retriever] parent context lookup failed: %s", exc, exc_info=True)
        stored_contexts = {}

    expanded: list[dict] = []
    seen: set[tuple] = set()
    for item in candidates:
        qid = int(item["qdrant_id"]) if item.get("qdrant_id") is not None else 0
        stored = stored_contexts.get(qid, {})
        child_text = (
            stored.get("raw_text")
            or item.get("raw_text")
            or item.get("child_text")
            or item.get("text", "")
        )
        parent_text = stored.get("parent_text") or item.get("parent_text") or child_text
        parent_id = stored.get("parent_id") or item.get("parent_id", 0)
        key = (
            item.get("file_path", ""),
            parent_id or item.get("page_num", 0),
            (parent_text or child_text)[:128],
        )
        if key in seen:
            continue
        seen.add(key)

        expanded_item = dict(item)
        expanded_item["matched_text"] = child_text
        expanded_item["child_text"] = child_text
        expanded_item["text"] = parent_text
        expanded_item["parent_id"] = parent_id
        expanded_item["parent_text_length"] = len(parent_text)
        expanded_item["contextual_prefix"] = stored.get("contextual_prefix") or item.get(
            "contextual_prefix", ""
        )
        expanded.append(expanded_item)

    return expanded
