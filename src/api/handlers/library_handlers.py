from __future__ import annotations

from pathlib import Path

import yaml
from fastapi import HTTPException
from fastapi.responses import FileResponse, Response

from src.api.model_tasks import ModelTaskTimeout
from src.api.runtime import get_api_runtime
from src.api.schemas import (
    BatchFavoriteRequest,
    BatchMetadataRequest,
    BatchRebuildRequest,
    FileMetadataRequest,
    KnowledgeRelationshipRequest,
    SummarizeRequest,
)
from src.domain_types import FileStatus
from src.ingest.watcher import _is_excluded


def _api():
    return get_api_runtime()


async def trigger_ingest():
    """手动触发全量扫描所有监控目录（异步，立即返回）。"""
    if _api().ingest_queue is None or not _api().watch_dirs:
        raise HTTPException(503, "Pipeline not ready")
    supported_exts = _api().pipeline.registry.supported_extensions
    all_files: list[Path] = []
    for wd in _api().watch_dirs:
        for ext in wd.extensions if wd.extensions else supported_exts:
            pattern = f"**/*{ext}" if wd.recursive else f"*{ext}"
            all_files.extend(f for f in wd.path.glob(pattern) if not _is_excluded(f))
    result = _api().ingest_queue.submit_many(all_files)
    return {**result, "files": [p.name for p in all_files]}


async def queue_status():
    if _api().ingest_queue is None:
        return {
            "queue_size": 0,
            "processing": None,
            "processing_files": [],
            "pending_files": [],
            "progress": None,
            "last_completed": None,
            "paused": False,
            "pause_reason": None,
            "paused_since": None,
            "foreground": _api().model_tasks.status(),
        }
    return {**_api().ingest_queue.status(), "foreground": _api().model_tasks.status()}


async def list_files(
    status: str | None = None,
    collection: str | None = None,
    tag: str | None = None,
    favorite: bool | None = None,
    kind: str | None = None,
    recent: bool | None = None,
):
    if _api().store is None:
        raise HTTPException(503, "Store not ready")
    return _api().store.list_files(
        status=status,
        collection=collection,
        tag=tag,
        favorite=favorite,
        kind=kind,
        recent=recent,
    )


async def library_meta():
    if _api().store is None:
        raise HTTPException(503, "Store not ready")
    return _api().store.list_library_facets()


async def storage_usage():
    if _api().store is None:
        raise HTTPException(503, "Store not ready")
    with open(_api().CONFIG_PATH, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    return _api()._collect_storage_usage(cfg, _api().store)


async def knowledge_overview(file_id: int | None = None):
    if _api().store is None:
        raise HTTPException(503, "Store not ready")
    return _api().knowledge_service.overview(_api().store, active_file_id=file_id)


async def knowledge_review(limit: int = 6):
    if _api().store is None:
        raise HTTPException(503, "Store not ready")
    return _api().knowledge_service.review(_api().store, limit=limit)


async def confirm_knowledge_relationship(req: KnowledgeRelationshipRequest):
    if _api().store is None:
        raise HTTPException(503, "Store not ready")
    source_id = int(req.source_file_id)
    target_id = int(req.target_file_id)
    if source_id == target_id:
        raise HTTPException(400, "请选择两份不同资料")
    source = _api().store.get_file_by_id(source_id)
    target = _api().store.get_file_by_id(target_id)
    if source is None or target is None:
        raise HTTPException(404, "资料不存在")
    relation = str(req.relation or "manual_relationship").strip()[:40]
    relation = relation or "manual_relationship"
    current_targets = [
        int((link.get("file") or {}).get("id") or 0)
        for link in _api().store.list_outbound_links(source_id)
        if str(link.get("relation") or "") == relation
    ]
    saved = _api().store.replace_note_source_links(
        source_id,
        [*current_targets, target_id],
        relation,
    )
    return {
        "source": source,
        "target": target,
        "relation": relation,
        "source_links": saved,
    }


async def update_file_metadata(file_id: int, req: FileMetadataRequest):
    if _api().store is None:
        raise HTTPException(503, "Store not ready")
    record = _api().store.update_file_metadata(
        file_id,
        collection=req.collection,
        user_tags=req.user_tags,
    )
    if record is None:
        raise HTTPException(404, "File not found")
    return record


async def batch_favorite(req: BatchFavoriteRequest):
    if _api().store is None:
        raise HTTPException(503, "Store not ready")
    if not req.file_ids:
        raise HTTPException(400, "No file IDs provided")
    changed = _api().store.set_favorites(req.file_ids, favorited=req.favorited)
    return {"file_ids": changed, "favorited": req.favorited, "count": len(changed)}


async def batch_update_file_metadata(req: BatchMetadataRequest):
    if _api().store is None:
        raise HTTPException(503, "Store not ready")
    if not req.file_ids:
        raise HTTPException(400, "No file IDs provided")
    records = _api().store.update_files_metadata(
        req.file_ids,
        collection=req.collection,
        user_tags=req.user_tags,
    )
    return {"files": records, "count": len(records)}


async def batch_rebuild_files(req: BatchRebuildRequest):
    if _api().store is None or _api().ingest_queue is None:
        raise HTTPException(503, "Not ready")
    if not req.file_ids:
        raise HTTPException(400, "No file IDs provided")

    paths: list[Path] = []
    missing_ids: list[int] = []
    for file_id in dict.fromkeys(req.file_ids):
        record = _api().store.get_file_by_id(file_id)
        if record is None:
            missing_ids.append(file_id)
            continue
        path = Path(record["file_path"])
        if path.exists():
            paths.append(path)
        else:
            missing_ids.append(file_id)

    if not paths:
        raise HTTPException(404, "No existing files found")
    result = _api().ingest_queue.submit_many(paths)
    return {
        **result,
        "requested": len(req.file_ids),
        "files": [path.name for path in paths],
        "missing_ids": missing_ids,
    }


async def preview_file(file_id: int):
    file_path, media_type = _resolve_preview_file(file_id)
    return FileResponse(str(file_path), media_type=media_type)


async def preview_file_head(file_id: int):
    file_path, media_type = _resolve_preview_file(file_id)
    return Response(
        status_code=200,
        headers={
            "content-length": str(file_path.stat().st_size),
            "content-type": media_type,
        },
    )


def _resolve_preview_file(file_id: int) -> tuple[Path, str]:
    if _api().store is None:
        raise HTTPException(503, "Store not ready")
    record = _api().store.get_file_by_id(file_id)
    if record is None:
        raise HTTPException(404, "File not found")
    file_path = Path(record["file_path"])
    if not file_path.exists():
        raise HTTPException(404, "File not found on disk")
    # 根据扩展名选择 MIME type
    suffix = file_path.suffix.lower()
    media_types = {
        ".pdf": "application/pdf",
        ".md": "text/markdown; charset=utf-8",
        ".txt": "text/plain; charset=utf-8",
        ".docx": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
    }
    media_type = media_types.get(suffix, "application/octet-stream")
    return file_path, media_type


async def list_file_chunks(file_id: int, max_text_chars: int = 500):
    """本地调试：查看文件实际切出的 chunk 和文本预览。"""
    if _api().store is None or _api().query_engine is None:
        raise HTTPException(503, "Not ready")
    record = _api().store.get_file_by_id(file_id)
    if record is None:
        raise HTTPException(404, "File not found")

    chunk_rows = _api().store.list_file_chunks(file_id)
    qdrant_ids = [row["qdrant_id"] for row in chunk_rows]
    payloads = _api().query_engine.retriever.fetch_chunks_by_ids(
        qdrant_ids,
        max_text_chars=max(0, min(max_text_chars, 2000)),
    )
    chunks = []
    for row in chunk_rows:
        payload = payloads.get(row["qdrant_id"], {})
        chunks.append(
            {
                **row,
                "text_preview": payload.get("text_preview", ""),
                "text_length": payload.get("text_length", 0),
            }
        )
    return {"file": record, "chunks": chunks, "count": len(chunks)}


async def list_history(limit: int = 50):
    if _api().store is None:
        raise HTTPException(503, "Store not ready")
    items = _api().store.list_history(limit=limit)
    return _api().query_service.decode_history_items(items)


async def search_history(q: str, limit: int = 20):
    if _api().store is None:
        raise HTTPException(503, "Store not ready")
    items = _api().store.search_history(q, limit=limit)
    return _api().query_service.decode_history_items(items)


async def clear_history():
    if _api().store is None:
        raise HTTPException(503, "Store not ready")
    _api().store.clear_history()
    return {"ok": True}


async def list_favorites():
    if _api().store is None:
        raise HTTPException(503, "Store not ready")
    return _api().store.list_favorites()


async def toggle_favorite(file_id: int):
    if _api().store is None:
        raise HTTPException(503, "Store not ready")
    added = _api().store.toggle_favorite(file_id)
    return {"file_id": file_id, "favorited": added}


async def summarize_files(req: SummarizeRequest):
    if _api().store is None or _api().query_engine is None:
        raise HTTPException(503, "Not ready")
    if not req.file_ids:
        raise HTTPException(400, "No file IDs provided")

    summaries: list[str] = []
    for fid in req.file_ids:
        record = _api().store.get_file_by_id(fid)
        if not record or record["status"] != FileStatus.DONE:
            continue
        qdrant_ids = _api().store.get_file_qdrant_ids(fid)
        try:
            md = await _api().model_tasks.run(
                "summarize",
                lambda record=record, qdrant_ids=qdrant_ids: _api().query_engine.summarize_file(
                    record["file_name"],
                    qdrant_ids,
                ),
                timeout_s=_api().MODEL_TASK_TIMEOUT_S,
            )
        except ModelTaskTimeout as exc:
            _api().logger.warning("[api/summarize] timeout id=%s file_id=%s", exc.task_id, fid)
            raise HTTPException(504, _api().MODEL_TIMEOUT_MESSAGE) from exc
        summaries.append(md)

    if not summaries:
        raise HTTPException(404, "No valid files found")

    combined = "\n\n---\n\n".join(summaries)
    return Response(
        content=combined,
        media_type="text/markdown; charset=utf-8",
        headers={"Content-Disposition": 'attachment; filename="docflow-summary.md"'},
    )
