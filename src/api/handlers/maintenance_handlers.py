from __future__ import annotations

import sys

from fastapi import HTTPException

from src.api.handlers.query_handlers import _resolve_query_options
from src.api.model_tasks import ModelTaskTimeout
from src.api.schemas import DebugRetrieveRequest


def _api():
    return sys.modules["src.api.app_impl"]


async def debug_retrieve(req: DebugRetrieveRequest):
    """本地调试：返回向量、全文、融合、去重、精排的完整检索链路。"""
    if _api().query_engine is None:
        raise HTTPException(503, "Query engine not ready")
    options = _resolve_query_options(req)
    prefer_tables = _api().query_engine._is_table_query(req.question)
    try:
        return await _api().model_tasks.run(
            "debug_retrieve",
            lambda: _api().query_engine.retriever.debug_retrieve(
                req.question,
                file_filter=options.file_filter or None,
                retrieval_mode=options.retrieval_mode,
                prefer_tables=prefer_tables,
                include_rerank=req.include_rerank,
                max_text_chars=max(0, min(req.max_text_chars, 2000)),
            ),
            timeout_s=_api().MODEL_TASK_TIMEOUT_S,
        )
    except ModelTaskTimeout as exc:
        _api().logger.warning(
            "[api/debug/retrieve] timeout id=%s question=%r", exc.task_id, req.question[:80]
        )
        raise HTTPException(504, _api().MODEL_TIMEOUT_MESSAGE) from exc
