from __future__ import annotations

from pathlib import Path

from fastapi import HTTPException, UploadFile

from src.api.model_tasks import ModelTaskTimeout
from src.api.runtime import get_api_runtime
from src.api.schemas import (
    AnswerNoteRequest,
    KnowledgeOutputRequest,
    NoteCreateRequest,
    WebImportRequest,
)


def _api():
    return get_api_runtime()


async def import_url(req: WebImportRequest):
    if _api().store is None or _api().ingest_queue is None:
        raise HTTPException(503, "Not ready")
    try:
        item = _api().fetch_webpage_markdown(req.url, title=req.title)
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc
    except Exception as exc:
        raise HTTPException(502, f"Failed to fetch URL: {exc}") from exc
    return _write_import_and_enqueue(
        "web",
        item,
        collection=req.collection or "Web Imports",
        user_tags=req.user_tags or ["web"],
    )


async def create_note(req: NoteCreateRequest):
    if _api().store is None or _api().ingest_queue is None:
        raise HTTPException(503, "Not ready")
    try:
        item = _api().build_quick_note_markdown(
            req.title,
            req.content,
            tags=req.user_tags or ["note"],
        )
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc
    result = _write_import_and_enqueue(
        "note",
        item,
        collection=req.collection or "Notes",
        user_tags=req.user_tags or ["note"],
    )
    note_file_id = int((result.get("file") or {}).get("id") or 0)
    result["source_links"] = _link_file_sources(
        note_file_id,
        req.source_file_ids,
        req.source_relation or "source_note",
    )
    return result


async def save_answer_note(req: AnswerNoteRequest):
    if _api().store is None or _api().ingest_queue is None:
        raise HTTPException(503, "Not ready")
    try:
        item = _api().build_answer_note_markdown(
            req.title,
            req.answer,
            question=req.question,
            citations=req.citations or [],
            tags=req.user_tags or ["answer"],
        )
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc
    result = _write_import_and_enqueue(
        "answer",
        item,
        collection=req.collection or "Saved Answers",
        user_tags=req.user_tags or ["answer"],
    )
    note_file_id = int((result.get("file") or {}).get("id") or 0)
    source_file_ids = _source_file_ids_from_citations(req.citations or [])
    result["source_links"] = _link_file_sources(note_file_id, source_file_ids, "answer_note")
    return result


def _source_file_ids_from_citations(citations: list[dict]) -> list[int]:
    if _api().store is None or not citations:
        return []
    files = _api().store.list_files()
    by_path = {str(file.get("file_path") or ""): int(file["id"]) for file in files}
    by_name: dict[str, int] = {}
    for file in files:
        file_name = str(file.get("file_name") or "")
        if file_name and file_name not in by_name:
            by_name[file_name] = int(file["id"])
    source_ids = []
    for citation in citations:
        file_path = str(citation.get("file_path") or "")
        file_name = str(citation.get("file_name") or "")
        file_id = by_path.get(file_path) or by_name.get(file_name)
        if file_id:
            source_ids.append(file_id)
    return list(dict.fromkeys(source_ids))


async def create_knowledge_output(req: KnowledgeOutputRequest):
    if _api().store is None or _api().ingest_queue is None or _api().query_engine is None:
        raise HTTPException(503, "Not ready")
    try:
        output = _api().get_knowledge_output_type(req.output_type)
        source_text, source_files = _build_knowledge_output_source(req)
        title = req.title or output.label
        generated = await _api().model_tasks.run(
            "knowledge_output",
            lambda: _api().query_engine.generate_knowledge_output(output.id, title, source_text),
            timeout_s=_api().MODEL_TASK_TIMEOUT_S,
        )
        user_tags = _api().knowledge_output_tags(output.id, req.user_tags)
        item = _api().build_knowledge_output_markdown(
            title,
            output.id,
            generated,
            source_files=source_files,
            tags=req.user_tags,
        )
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc
    except ModelTaskTimeout as exc:
        _api().logger.warning(
            "[api/knowledge-output] timeout id=%s type=%s",
            exc.task_id,
            req.output_type,
        )
        raise HTTPException(504, _api().MODEL_TIMEOUT_MESSAGE) from exc

    result = _write_import_and_enqueue(
        "knowledge",
        item,
        collection=req.collection or "Knowledge Outputs",
        user_tags=user_tags,
    )
    knowledge_file_id = int((result.get("file") or {}).get("id") or 0)
    source_links = _link_file_sources(knowledge_file_id, req.file_ids, "knowledge_output")
    return {
        **result,
        "output_type": output.id,
        "source_files": source_files,
        "source_links": source_links,
        "preview": generated[:500],
    }


def _link_file_sources(note_file_id: int, source_file_ids: list[int], relation: str) -> list[int]:
    if _api().store is None or not note_file_id or not source_file_ids:
        return []
    return _api().store.replace_note_source_links(note_file_id, source_file_ids, relation)


def _build_knowledge_output_source(req: KnowledgeOutputRequest) -> tuple[str, list[str]]:
    _api()._sync_app_state()
    return _api().import_service.build_knowledge_output_source(_api().app_state, req)


def _format_knowledge_file_context(file_name: str, chunks: list[dict]) -> str:
    return _api().import_service.format_knowledge_file_context(file_name, chunks)


def _write_import_and_enqueue(
    prefix: str,
    item,
    collection: str,
    user_tags: list[str],
) -> dict:
    _api()._sync_app_state()
    return _api().import_service.write_import_and_enqueue(
        _api().app_state,
        prefix=prefix,
        item=item,
        collection=collection,
        user_tags=user_tags,
    )


async def upload_file(file: UploadFile):
    """上传文件到第一个监控目录（支持所有已注册格式）。"""
    if not _api().watch_dirs or _api().ingest_queue is None:
        raise HTTPException(503, "Not ready")
    dest = _api().import_service.safe_upload_destination(
        _api().watch_dirs[0].path,
        file.filename or "",
    )
    supported_exts = _api().pipeline.registry.supported_extensions
    suffix = dest.suffix.lower()
    if suffix not in supported_exts:
        raise HTTPException(400, f"Unsupported file type: {suffix}. Supported: {supported_exts}")
    try:
        with dest.open("wb") as f:
            while chunk := await file.read(1024 * 1024):
                f.write(chunk)
    finally:
        await file.close()

    return _api().ingest_queue.submit(dest)


async def create_demo_library():
    if _api().ingest_queue is None:
        raise HTTPException(503, "Not ready")
    from src.maintenance.demo import create_demo_files

    result = create_demo_files(_api().CONFIG_PATH)
    queued = []
    for item in result["files"]:
        queued.append(_api().ingest_queue.submit(Path(item["path"])))
    return {**result, "queued": len(queued), "queue_results": queued}
