from __future__ import annotations

import json

from fastapi import HTTPException

from src.api.model_tasks import ModelTaskTimeout
from src.api.runtime import get_api_runtime
from src.api.schemas import (
    AnswerFeedbackRequest,
    ConversationCreateRequest,
    QueryOptions,
    QueryRequest,
    QueryResponse,
    ResearchRequest,
    ResearchResponse,
)
from src.domain_types import FileStatus


def _api():
    return get_api_runtime()


def _claim_support(quality: dict) -> dict | None:
    return quality.get("claim_support") if isinstance(quality, dict) else None


def _resolve_query_options(req) -> QueryOptions:
    scope_mode = str(getattr(req, "scope_mode", None) or "all").strip().lower().replace("-", "_")
    retrieval_mode = _normalize_retrieval_mode(getattr(req, "retrieval_mode", None))
    if scope_mode == "full_text":
        retrieval_mode = "full_text"

    legacy_filter = _clean_file_filter(getattr(req, "file_filter", None))
    if legacy_filter:
        return QueryOptions(
            file_filter=legacy_filter,
            retrieval_mode=retrieval_mode,
            scope={
                "mode": "file_filter",
                "label": "指定文件",
                "files": legacy_filter,
                "retrieval_mode": retrieval_mode,
            },
        )

    if scope_mode in {"all", "full_text"}:
        label = "全文模式" if scope_mode == "full_text" else "全部知识库"
        return QueryOptions(
            file_filter=[],
            retrieval_mode=retrieval_mode,
            scope={"mode": scope_mode, "label": label, "retrieval_mode": retrieval_mode},
        )

    if scope_mode == "collection":
        if _api().store is None:
            raise HTTPException(503, "File store not ready")
        collection = str(getattr(req, "collection", "") or "").strip()
        if not collection:
            raise HTTPException(400, "Collection is required for collection scope")
        files = _api().store.list_files(status=FileStatus.DONE, collection=collection)
        file_names = _unique_file_names(files)
        if not file_names:
            raise HTTPException(404, f"No indexed files found in collection: {collection}")
        return QueryOptions(
            file_filter=file_names,
            retrieval_mode=retrieval_mode,
            scope={
                "mode": "collection",
                "label": collection,
                "collection": collection,
                "file_count": len(file_names),
                "retrieval_mode": retrieval_mode,
            },
        )

    if scope_mode == "file":
        file_name = str(getattr(req, "file_name", "") or "").strip()
        file_id = getattr(req, "file_id", None)
        if file_id is not None:
            if _api().store is None:
                raise HTTPException(503, "File store not ready")
            record = _api().store.get_file_by_id(int(file_id))
            if record is None or record.get("status") != "done":
                raise HTTPException(404, "Indexed file not found")
            file_name = str(record.get("file_name") or "").strip()
        if not file_name:
            raise HTTPException(400, "File is required for file scope")
        return QueryOptions(
            file_filter=[file_name],
            retrieval_mode=retrieval_mode,
            scope={
                "mode": "file",
                "label": file_name,
                "file_id": file_id,
                "file_name": file_name,
                "retrieval_mode": retrieval_mode,
            },
        )

    raise HTTPException(400, f"Unsupported query scope: {scope_mode}")


def _clean_file_filter(values: list[str] | None) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for value in values or []:
        file_name = str(value or "").strip()
        if not file_name or file_name in seen:
            continue
        seen.add(file_name)
        result.append(file_name)
    return result


def _unique_file_names(files: list[dict]) -> list[str]:
    return _clean_file_filter([str(item.get("file_name") or "") for item in files])


def _normalize_retrieval_mode(mode: str | None) -> str:
    normalized = str(mode or "hybrid").strip().lower().replace("-", "_")
    if normalized in {"fts", "fulltext", "full_text"}:
        return "full_text"
    return "hybrid"


async def query(req: QueryRequest):
    if _api().query_engine is None:
        raise HTTPException(503, "Query engine not ready")
    options = _resolve_query_options(req)
    conversation_id = _resolve_conversation_id(req.conversation_id)
    conversation_context = _conversation_context(conversation_id)
    retrieval_query = _build_retrieval_query(req.question, conversation_context)
    file_filter_json = json.dumps(options.file_filter, ensure_ascii=False)
    if _api().store is not None and conversation_id is not None:
        _api().store.add_message(
            conversation_id=conversation_id,
            role="user",
            content=req.question,
            file_filter_json=file_filter_json,
        )
    try:
        result = await _api().model_tasks.run(
            "query",
            lambda: _api().query_engine.query(
                req.question,
                file_filter=options.file_filter or None,
                retrieval_mode=options.retrieval_mode,
                conversation_context=conversation_context,
                retrieval_query=retrieval_query,
            ),
            timeout_s=_api().MODEL_TASK_TIMEOUT_S,
        )
    except ModelTaskTimeout as exc:
        _api().logger.warning(
            "[api/query] timeout id=%s question=%r",
            exc.task_id,
            req.question[:80],
        )
        raise HTTPException(504, _api().MODEL_TIMEOUT_MESSAGE) from exc
    citations_data = _api().query_service.response_citations(result.citations)
    quality = getattr(result, "quality", {}) or {}
    history_id = None
    if _api().store is not None:
        history_id = _api().store.add_history(
            question=req.question,
            answer=result.text,
            citations_json=json.dumps(citations_data, ensure_ascii=False),
            file_filter_json=file_filter_json,
            conversation_id=conversation_id or 0,
        )
        if conversation_id is not None:
            _api().store.add_message(
                conversation_id=conversation_id,
                role="assistant",
                content=result.text,
                citations_json=json.dumps(citations_data, ensure_ascii=False),
                file_filter_json=file_filter_json,
            )
    return QueryResponse(
        answer=result.text,
        citations=citations_data,
        evidence=_api().query_service.evidence_summary(citations_data, _claim_support(quality)),
        quality=quality,
        related_notes=getattr(result, "related_notes", []),
        history_id=history_id,
        conversation_id=conversation_id,
        scope=options.scope,
        reproducible=getattr(result, "reproducible", True),
    )


async def research(req: ResearchRequest):
    if _api().query_engine is None:
        raise HTTPException(503, "Query engine not ready")
    options = _resolve_query_options(req)
    conversation_id = _resolve_conversation_id(req.conversation_id)
    conversation_context = _conversation_context(conversation_id)
    file_filter_json = json.dumps(options.file_filter, ensure_ascii=False)
    if _api().store is not None and conversation_id is not None:
        _api().store.add_message(
            conversation_id=conversation_id,
            role="user",
            content=req.question,
            file_filter_json=file_filter_json,
        )
    try:
        result = await _api().model_tasks.run(
            "research",
            lambda: _api().query_engine.deep_research(
                req.question,
                file_filter=options.file_filter or None,
                retrieval_mode=options.retrieval_mode,
                max_steps=req.max_steps,
                conversation_context=conversation_context,
            ),
            timeout_s=max(_api().MODEL_TASK_TIMEOUT_S, 180),
        )
    except ModelTaskTimeout as exc:
        _api().logger.warning(
            "[api/research] timeout id=%s question=%r",
            exc.task_id,
            req.question[:80],
        )
        raise HTTPException(504, _api().MODEL_TIMEOUT_MESSAGE) from exc
    citations_data = _api().query_service.response_citations(result.citations)
    quality = getattr(result, "quality", {}) or {}
    history_id = None
    if _api().store is not None:
        history_id = _api().store.add_history(
            question=req.question,
            answer=result.text,
            citations_json=json.dumps(citations_data, ensure_ascii=False),
            file_filter_json=file_filter_json,
            conversation_id=conversation_id or 0,
        )
        if conversation_id is not None:
            _api().store.add_message(
                conversation_id=conversation_id,
                role="assistant",
                content=result.text,
                citations_json=json.dumps(citations_data, ensure_ascii=False),
                file_filter_json=file_filter_json,
            )
    return ResearchResponse(
        answer=result.text,
        citations=citations_data,
        evidence=_api().query_service.evidence_summary(citations_data, _claim_support(quality)),
        quality=quality,
        related_notes=getattr(result, "related_notes", []),
        research_steps=getattr(result, "research_steps", []),
        history_id=history_id,
        conversation_id=conversation_id,
        scope=options.scope,
        reproducible=getattr(result, "reproducible", True),
    )


async def answer_feedback(req: AnswerFeedbackRequest):
    if _api().store is None:
        raise HTTPException(503, "Store not ready")
    try:
        feedback = _api().store.set_answer_feedback(req.history_id, req.rating, req.note or "")
    except KeyError as exc:
        raise HTTPException(404, "Answer history item not found") from exc
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc
    return {"ok": True, "feedback": feedback, "summary": _api().store.get_feedback_summary()}


def _resolve_conversation_id(conversation_id: int | None) -> int | None:
    if _api().store is None:
        return None
    if conversation_id is None:
        return _api().store.create_conversation()
    if _api().store.get_conversation(conversation_id) is None:
        raise HTTPException(404, "Conversation not found")
    return conversation_id


def _conversation_context(conversation_id: int | None, limit: int = 6) -> list[dict]:
    if _api().store is None or conversation_id is None:
        return []
    return _api().store.list_messages(conversation_id, limit=limit)


def _build_retrieval_query(question: str, conversation_context: list[dict]) -> str:
    return _api().query_service.build_retrieval_query(question, conversation_context)


def _looks_like_followup(question: str) -> bool:
    return _api().query_service.looks_like_followup(question)


async def list_conversations(limit: int = 50):
    if _api().store is None:
        raise HTTPException(503, "Store not ready")
    return _api().store.list_conversations(limit=limit)


async def create_conversation(req: ConversationCreateRequest):
    if _api().store is None:
        raise HTTPException(503, "Store not ready")
    conversation_id = _api().store.create_conversation(req.title)
    return _api().store.get_conversation(conversation_id)


async def list_conversation_messages(conversation_id: int, limit: int = 100):
    if _api().store is None:
        raise HTTPException(503, "Store not ready")
    if _api().store.get_conversation(conversation_id) is None:
        raise HTTPException(404, "Conversation not found")
    items = _api().store.list_messages(conversation_id, limit=limit)
    return _api().query_service.decode_history_items(items)


async def delete_conversation(conversation_id: int):
    if _api().store is None:
        raise HTTPException(503, "Store not ready")
    if not _api().store.delete_conversation(conversation_id):
        raise HTTPException(404, "Conversation not found")
    return {"ok": True}
