from __future__ import annotations

import asyncio
import json
import queue
import threading
from time import perf_counter

from fastapi import HTTPException, Request
from fastapi.responses import StreamingResponse

from src.api.handlers.query_handlers import (
    _build_retrieval_query,
    _claim_support,
    _conversation_context,
    _resolve_conversation_id,
    _resolve_query_options,
)
from src.api.runtime import get_api_runtime
from src.api.schemas import QueryRequest


def _api():
    return get_api_runtime()


async def query_stream(req: QueryRequest, request: Request):
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
    loop = asyncio.get_event_loop()
    q: queue.Queue = queue.Queue()
    cancel = threading.Event()

    def _run():
        try:
            if conversation_id is not None:
                q.put(("conversation", {"conversation_id": conversation_id}))
            stream_result = _api().query_engine.query_stream(
                req.question,
                file_filter=options.file_filter or None,
                retrieval_mode=options.retrieval_mode,
                cancel_event=cancel,
                conversation_context=conversation_context,
                retrieval_query=retrieval_query,
                include_related=True,
            )
            if len(stream_result) == 4:
                chunks, token_gen, related_notes, quality = stream_result
            elif len(stream_result) == 3:
                chunks, token_gen, related_notes = stream_result
                quality = {}
            else:
                chunks, token_gen = stream_result
                related_notes = []
                quality = {}
            if cancel.is_set():
                return
            citations_data = _api().query_service.stream_citations(chunks)
            if cancel.is_set():
                return
            q.put(("citations", citations_data))
            q.put(("evidence", _api().query_service.evidence_summary(citations_data)))
            q.put(("quality", quality))
            q.put(("related_notes", related_notes))
            full_answer = []
            last_quality = json.dumps(quality, ensure_ascii=False, sort_keys=True)
            for token in token_gen:
                if cancel.is_set():
                    return
                current_quality = json.dumps(quality, ensure_ascii=False, sort_keys=True)
                if current_quality != last_quality:
                    q.put(("quality", quality))
                    last_quality = current_quality
                full_answer.append(token)
                q.put(("token", token))
            answer_text = "".join(full_answer).strip()
            finalize = _api().query_service.finalize_stream_answer_with_quality
            answer_text, citations_data, quality = finalize(answer_text, chunks, quality)
            evidence_data = _api().query_service.evidence_summary(
                citations_data,
                _claim_support(quality),
            )
            q.put(("quality", quality))
            answer_payload = {
                "answer": answer_text,
                "citations": citations_data,
                "evidence": evidence_data,
                "quality": quality,
            }
            q.put(("answer", answer_payload))
            history_id = None
            if not cancel.is_set() and _api().store is not None:
                history_id = _api().store.add_history(
                    question=req.question,
                    answer=answer_text,
                    citations_json=json.dumps(citations_data, ensure_ascii=False),
                    file_filter_json=file_filter_json,
                    conversation_id=conversation_id or 0,
                )
                if conversation_id is not None:
                    _api().store.add_message(
                        conversation_id=conversation_id,
                        role="assistant",
                        content=answer_text,
                        citations_json=json.dumps(citations_data, ensure_ascii=False),
                        file_filter_json=file_filter_json,
                    )
            if not cancel.is_set():
                q.put(("done", {"history_id": history_id, "conversation_id": conversation_id}))
        except Exception as e:
            if not cancel.is_set():
                q.put(("error", str(e)))

    task = _api().model_tasks.submit("query_stream", _run)

    async def event_stream():
        first_content_at: float | None = None
        last_content_at = perf_counter()
        while True:
            if await request.is_disconnected():
                cancel.set()
                _api().model_tasks.cancel_and_retire(task, reason="client disconnected")
                break
            try:
                event, data = await loop.run_in_executor(
                    None,
                    q.get,
                    True,
                    _api().STREAM_QUEUE_POLL_S,
                )
            except queue.Empty:
                now = perf_counter()
                if (
                    first_content_at is None
                    and now - task.started_at > _api().STREAM_FIRST_CONTENT_TIMEOUT_S
                ):
                    cancel.set()
                    timeout = _api().STREAM_FIRST_CONTENT_TIMEOUT_S
                    reason = f"stream first content timeout after {timeout:.1f}s"
                    _api().model_tasks.cancel_and_retire(task, reason=reason)
                    yield (
                        "event: error\n"
                        f"data: {json.dumps(_api().MODEL_TIMEOUT_MESSAGE, ensure_ascii=False)}\n\n"
                    )
                    break
                if (
                    first_content_at is not None
                    and now - last_content_at > _api().STREAM_IDLE_TIMEOUT_S
                ):
                    cancel.set()
                    _api().model_tasks.cancel_and_retire(
                        task,
                        reason=f"stream idle timeout after {_api().STREAM_IDLE_TIMEOUT_S:.1f}s",
                    )
                    yield (
                        "event: error\n"
                        f"data: {json.dumps(_api().MODEL_TIMEOUT_MESSAGE, ensure_ascii=False)}\n\n"
                    )
                    break
                if task.future.done() and q.empty():
                    break
                continue
            if event in (
                "citations",
                "evidence",
                "quality",
                "related_notes",
                "token",
                "answer",
                "done",
                "error",
            ):
                if first_content_at is None:
                    first_content_at = perf_counter()
                last_content_at = perf_counter()
            yield f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"
            if event in ("done", "error"):
                break

    return StreamingResponse(event_stream(), media_type="text/event-stream")
