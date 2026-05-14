from __future__ import annotations

from src.query.constants import COLLECTION_NAME


def vector_search(
    retriever,
    query_vec: list[float],
    file_filter: list[str] | None,
    limit: int | None = None,
) -> list[dict]:
    results = retriever._vector_store.search(
        collection_name=COLLECTION_NAME,
        query=query_vec,
        file_filter=file_filter,
        limit=limit or retriever.top_k_retrieval,
    )
    return [
        {
            "qdrant_id": hit.id,
            "score": hit.score,
            **hit.payload,
        }
        for hit in results
    ]


def fetch_file_chunks(retriever, qdrant_ids: list[int], max_chunks: int = 15) -> list[dict]:
    if not qdrant_ids:
        return []
    sample_ids = qdrant_ids[: max_chunks * 3]
    records = retriever._qdrant.retrieve(
        collection_name=COLLECTION_NAME,
        ids=sample_ids,
        with_payload=True,
    )
    chunks = [{"qdrant_id": r.id, **dict(r.payload or {})} for r in records]
    chunks.sort(key=lambda c: (c.get("page_num", 0), c.get("qdrant_id", 0)))
    return chunks[:max_chunks]


def fetch_chunks_by_ids(
    retriever, qdrant_ids: list[int], max_text_chars: int = 500
) -> dict[int, dict]:
    if not qdrant_ids:
        return {}
    max_text_chars = max(0, max_text_chars)

    result: dict[int, dict] = {}
    for i in range(0, len(qdrant_ids), 100):
        batch = qdrant_ids[i : i + 100]
        records = retriever._qdrant.retrieve(
            collection_name=COLLECTION_NAME,
            ids=batch,
            with_payload=True,
        )
        for record in records:
            payload = dict(record.payload or {})
            text = payload.get("text", "")
            payload["text_preview"] = text[:max_text_chars]
            payload["text_length"] = len(text)
            payload.pop("text", None)
            result[int(record.id)] = payload
    return result
