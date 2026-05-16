"""Pure helpers used by QueryEngine orchestration."""

from __future__ import annotations

from src.query.answer_quality import (
    insufficient_evidence_quality,
    local_model_unavailable_quality,
    quality_with_claim_support,
    retrieval_quality_from_chunks,
)
from src.query.generator import Answer, citation_from_chunk
from src.query.settings import QuerySettings


def split_answer_and_related(
    chunks: list[dict],
    settings: QuerySettings,
) -> tuple[list[dict], list[dict]]:
    if not chunks:
        return [], []
    related_limit = settings.related_notes_limit
    if len(chunks) > related_limit + settings.min_answer_chunks:
        answer_limit = len(chunks) - related_limit
    else:
        answer_limit = min(len(chunks), settings.default_answer_chunks)
    answer_chunks = chunks[:answer_limit]
    related_chunks = chunks[answer_limit:]
    return answer_chunks, related_notes(answer_chunks, related_chunks, limit=related_limit)


def research_queries(
    question: str,
    settings: QuerySettings,
    max_steps: int = 3,
) -> list[str]:
    clean = " ".join(str(question or "").split())
    if not clean:
        return []
    limit = max(1, min(int(max_steps or 3), settings.max_research_steps))
    candidates = [
        clean,
        f"{clean}\n关键事实 证据 背景",
        f"{clean}\n对比 差异 风险 结论",
        f"{clean}\n时间线 原因 影响",
    ]
    result: list[str] = []
    seen: set[str] = set()
    for item in candidates:
        if item in seen:
            continue
        seen.add(item)
        result.append(item)
        if len(result) >= limit:
            break
    return result


def chunk_key(chunk: dict) -> str:
    if chunk.get("qdrant_id") is not None:
        return f"q:{chunk.get('qdrant_id')}"
    return "|".join(
        str(chunk.get(key, ""))
        for key in ("file_path", "file_name", "page_num", "section", "text", "raw_text")
    )


def related_notes(
    answer_chunks: list[dict],
    related_chunks: list[dict],
    extra_exclude_keys: set[str] | None = None,
    limit: int | None = None,
) -> list[dict]:
    limit = QuerySettings().related_notes_limit if limit is None else int(limit)
    cited_keys = {
        chunk.get("file_path") or chunk.get("file_name")
        for chunk in answer_chunks
        if chunk.get("file_path") or chunk.get("file_name")
    }
    cited_keys.update(extra_exclude_keys or set())
    notes: list[dict] = []
    seen: set[str] = set()
    for chunk in related_chunks:
        key = chunk.get("file_path") or chunk.get("file_name")
        if not key or key in cited_keys or key in seen:
            continue
        seen.add(key)
        score = chunk.get("rerank_score", chunk.get("rrf_score", chunk.get("score", 0.0)))
        text = chunk.get("text") or chunk.get("raw_text") or chunk.get("parent_text") or ""
        notes.append(
            {
                "file_name": chunk.get("file_name", ""),
                "file_path": chunk.get("file_path", ""),
                "page_num": chunk.get("page_num", 0),
                "section": chunk.get("section", ""),
                "snippet": text[:220],
                "score": round(float(score or 0.0), 4),
                "chunk_type": chunk.get("chunk_type", "text"),
            }
        )
        if len(notes) >= limit:
            break
    return notes


def is_table_query(question: str, settings: QuerySettings | None = None) -> bool:
    settings = settings or QuerySettings()
    q_lower = question.lower()
    return any(kw in q_lower for kw in settings.table_keywords)


def has_sufficient_evidence(
    chunks: list[dict],
    settings: QuerySettings | None = None,
) -> bool:
    settings = settings or QuerySettings()
    if not chunks:
        return False
    top = chunks[0]
    if top.get("rerank_score") is not None and not top.get("rerank_fallback"):
        return float(top.get("rerank_score") or 0) >= settings.min_rerank_score
    vec_score = top.get("vec_score")
    if vec_score is not None and float(vec_score or 0) > 0:
        return float(vec_score) >= settings.min_vector_score
    return True


def fallback_answer(
    chunks: list[dict],
    exc: Exception,
    settings: QuerySettings | None = None,
) -> Answer:
    settings = settings or QuerySettings()
    if not chunks:
        return Answer(
            text=settings.insufficient_evidence_message,
            citations=[],
            quality=insufficient_evidence_quality(),
        )

    text = (
        "已找到相关文档片段，但本地回答模型暂时不可用。"
        "请先查看下方引用片段；稍后可重试完整回答。"
    )
    citations = [citation_from_chunk(chunk) for chunk in chunks]
    return Answer(text=text, citations=citations, quality=local_model_unavailable_quality(exc))


def answer_quality(chunks: list[dict], generator_quality: dict | None) -> dict:
    quality = retrieval_quality_from_chunks(chunks)
    claim_support = (generator_quality or {}).get("claim_support")
    if isinstance(claim_support, dict):
        return quality_with_claim_support(quality, claim_support)
    return quality
