from __future__ import annotations

import sys

from fastapi import HTTPException

from src.api.handlers.query_handlers import _normalize_retrieval_mode
from src.api.schemas import ObsidianRelatedRequest
from src.query.engine import QueryEngine


def _api():
    return sys.modules["src.api.app_impl"]


async def obsidian_related_notes(req: ObsidianRelatedRequest):
    if _api().query_engine is None:
        raise HTTPException(503, "Query engine not ready")
    query_text = " ".join(
        part.strip()
        for part in [req.selection or "", req.note_title or "", req.note_content or ""]
        if part and part.strip()
    ).strip()
    if not query_text:
        raise HTTPException(400, "Note content or selection is required")
    query_text = query_text[:4000]
    exclude = {
        value
        for value in {
            req.note_path,
            req.note_title,
            f"{req.note_title}.md" if req.note_title else "",
        }
        if value
    }
    limit = max(1, min(int(req.limit or 6), 12))
    chunks = _api().query_engine.retriever.retrieve(
        query=query_text,
        file_filter=None,
        retrieval_mode=_normalize_retrieval_mode(req.retrieval_mode),
        prefer_tables=False,
        related_k=limit,
    )
    related_notes = QueryEngine._related_notes(
        [],
        chunks,
        extra_exclude_keys=exclude,
        limit=limit,
    )
    return {
        "related_notes": related_notes,
        "count": len(related_notes),
    }
