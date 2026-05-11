"""Import, note, and knowledge-output helpers for API handlers."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from fastapi import HTTPException

from src.domain_types import FileStatus
from src.ingest.imports import write_markdown_import
from src.ingest.store import DocStore
from src.knowledge_outputs import KNOWLEDGE_OUTPUT_SOURCE_CHAR_LIMIT


class ImportService:
    def build_knowledge_output_source(self, state: Any, req) -> tuple[str, list[str]]:
        if state.store is None or state.query_engine is None:
            raise HTTPException(503, "Not ready")

        source_parts: list[str] = []
        source_files: list[str] = []
        manual_text = (req.source_text or "").strip()
        if manual_text:
            source_parts.append("## 手动输入\n\n" + manual_text)

        for file_id in dict.fromkeys(req.file_ids):
            record = state.store.get_file_by_id(file_id)
            if not record or record["status"] != FileStatus.DONE:
                continue
            qdrant_ids = state.store.get_file_qdrant_ids(file_id)
            chunks = state.query_engine.retriever.fetch_file_chunks(qdrant_ids, max_chunks=12)
            file_context = self.format_knowledge_file_context(record["file_name"], chunks)
            if not file_context:
                continue
            source_files.append(record["file_name"])
            source_parts.append(file_context)

        source = "\n\n---\n\n".join(part for part in source_parts if part.strip()).strip()
        if not source:
            raise ValueError("Knowledge output source is empty")
        if len(source) > KNOWLEDGE_OUTPUT_SOURCE_CHAR_LIMIT:
            source = (
                source[:KNOWLEDGE_OUTPUT_SOURCE_CHAR_LIMIT].rstrip() + "\n\n[内容已按长度上限截断]"
            )
        return source, source_files

    def format_knowledge_file_context(self, file_name: str, chunks: list[dict]) -> str:
        rows: list[str] = []
        for chunk in chunks:
            text = (chunk.get("text") or chunk.get("raw_text") or "").strip()
            if not text:
                continue
            page = chunk.get("page_num") or 0
            section = f" / {chunk.get('section')}" if chunk.get("section") else ""
            rows.append(f"### 第{page}页{section}\n\n{text}")
        if not rows:
            return ""
        return f"## 文件：{file_name}\n\n" + "\n\n".join(rows)

    def write_import_and_enqueue(
        self,
        state: Any,
        *,
        prefix: str,
        item,
        collection: str,
        user_tags: list[str],
    ) -> dict:
        if state.store is None or state.ingest_queue is None or not state.watch_dirs:
            raise HTTPException(503, "Not ready")
        root = state.watch_dirs[0].path
        path = write_markdown_import(root, prefix, item)
        file_id = state.store.upsert_file(
            path,
            path.name,
            DocStore.compute_hash(path),
            status=FileStatus.PENDING,
            total_pages=1,
            mtime_ns=path.stat().st_mtime_ns,
        )
        record = state.store.update_file_metadata(
            file_id, collection=collection, user_tags=user_tags
        )
        queue_result = state.ingest_queue.submit(path)
        return {
            "status": "queued",
            "path": str(path),
            "file": record,
            "queue": queue_result,
        }

    def safe_upload_destination(self, root: Path, filename: str) -> Path:
        safe_name = Path(filename or "").name
        if not safe_name:
            raise HTTPException(400, "Missing filename")
        return root / safe_name
