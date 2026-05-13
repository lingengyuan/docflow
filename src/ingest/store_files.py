"""File and chunk metadata operations for DocStore."""

# mypy: disable-error-code="attr-defined"

from __future__ import annotations

import json
from pathlib import Path

from src.domain_types import ChunkRecord, FileRecord, FileStatus
from src.ingest.store_shared import DEFAULT_COLLECTION


class StoreFileMixin:
    def needs_ingest(self, file_path: str | Path) -> tuple[bool, str | None]:
        """
        Returns (need_ingest, file_hash_or_none).

        True if:
        - File is not in DB, or
        - File hash has changed (file updated), or
        - Previous ingest errored / interrupted

        优化：先比较 mtime，未变则跳过 hash 计算（大 vault 启动加速）。
        hash 一旦计算就返回，避免 pipeline.ingest() 重复计算。
        """
        path = Path(file_path)
        with self._conn() as conn:
            row = conn.execute(
                "SELECT file_hash, status, mtime_ns FROM files WHERE file_path = ?",
                (str(path),),
            ).fetchone()
        if row is None:
            return True, None
        if row["status"] in {
            FileStatus.PENDING.value,
            FileStatus.ERROR.value,
            FileStatus.PROCESSING.value,
        }:
            return True, None
        # mtime 快跳：未变则大概率不需要重新 ingest
        current_mtime = path.stat().st_mtime_ns
        if row["mtime_ns"] and current_mtime <= row["mtime_ns"]:
            return False, None
        # mtime 变了才算 hash（防止 touch 但内容没变）
        file_hash = self.compute_hash(path)
        return row["file_hash"] != file_hash, file_hash
    def upsert_file(
        self,
        file_path: str | Path,
        file_name: str,
        file_hash: str,
        status: FileStatus | str = FileStatus.PENDING,
        total_pages: int = 0,
        is_scanned: bool = False,
        tags: str = "[]",
        mtime_ns: int = 0,
    ) -> int:
        path = str(file_path)
        status_value = self._normalize_status(status)
        with self._conn() as conn:
            conn.execute(
                """
                INSERT INTO files (
                    file_path, file_name, file_hash, status,
                    total_pages, is_scanned, tags, mtime_ns
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(file_path) DO UPDATE SET
                    file_name   = excluded.file_name,
                    file_hash   = excluded.file_hash,
                    status      = excluded.status,
                    total_pages = excluded.total_pages,
                    is_scanned  = excluded.is_scanned,
                    tags        = excluded.tags,
                    mtime_ns    = excluded.mtime_ns,
                    error_msg   = '',
                    updated_at  = datetime('now')
            """,
                (
                    path,
                    file_name,
                    file_hash,
                    status_value,
                    total_pages,
                    int(is_scanned),
                    tags,
                    mtime_ns,
                ),
            )
            file_id = conn.execute("SELECT id FROM files WHERE file_path = ?", (path,)).fetchone()[
                "id"
            ]
        return file_id
    def reset_processing_files(self) -> int:
        """启动时调用：将残留的 'processing' 状态（上次 server 崩溃遗留）重置为 'error'。
        由于 needs_ingest() 对 'error' 返回 True，这些文件会在启动扫描时自动重新入队。
        """
        with self._conn() as conn:
            result = conn.execute(
                "UPDATE files SET status=?, error_msg='Interrupted (server restart)', "
                "updated_at=datetime('now') WHERE status=?",
                (FileStatus.ERROR.value, FileStatus.PROCESSING.value),
            )
            return result.rowcount
    def cleanup_deleted_files(self) -> list[dict]:
        """清理磁盘上已不存在的文件记录。返回被删除的文件列表（含 qdrant_ids 供向量清理）。"""
        removed: list[dict] = []
        with self._conn() as conn:
            rows = conn.execute("SELECT id, file_path, file_name FROM files").fetchall()
            for row in rows:
                if not Path(row["file_path"]).exists():
                    file_id = row["id"]
                    # Single query for both id and qdrant_id
                    chunk_rows = conn.execute(
                        "SELECT id, qdrant_id FROM chunks WHERE file_id = ?", (file_id,)
                    ).fetchall()
                    qids = [r["qdrant_id"] for r in chunk_rows]
                    chunk_ids = [r["id"] for r in chunk_rows]
                    if chunk_ids:
                        ph = ",".join("?" * len(chunk_ids))
                        conn.execute(f"DELETE FROM chunks_fts WHERE rowid IN ({ph})", chunk_ids)
                        conn.execute(
                            f"DELETE FROM chunks_fts_trigram WHERE rowid IN ({ph})", chunk_ids
                        )
                    # 删除 chunks、favorites、file 记录
                    conn.execute("DELETE FROM chunks WHERE file_id = ?", (file_id,))
                    conn.execute("DELETE FROM favorites WHERE file_id = ?", (file_id,))
                    conn.execute("DELETE FROM files WHERE id = ?", (file_id,))
                    removed.append(
                        {
                            "file_name": row["file_name"],
                            "qdrant_ids": qids,
                        }
                    )
        return removed
    def set_status(self, file_path: str | Path, status: FileStatus | str, error_msg: str = ""):
        status_value = self._normalize_status(status)
        with self._conn() as conn:
            conn.execute(
                """
                UPDATE files
                SET status = ?, error_msg = ?, updated_at = datetime('now')
                WHERE file_path = ?
            """,
                (status_value, error_msg, str(file_path)),
            )
    def set_chunk_count(self, file_path: str | Path, count: int):
        with self._conn() as conn:
            conn.execute(
                """
                UPDATE files
                SET chunk_count = ?, updated_at = datetime('now')
                WHERE file_path = ?
            """,
                (count, str(file_path)),
            )
    def add_chunks(self, file_id: int, chunk_records: list[ChunkRecord]):
        """
        chunk_records: list of {qdrant_id, chunk_type, page_num, section, char_count,
          parent_id?, raw_text?, embedding_text?, parent_text?, contextual_prefix?, tokenized_text?}
        同步写入两个 FTS5 索引：
          chunks_fts         — jieba 预分词，精确匹配
          chunks_fts_trigram — trigram，子串匹配（OCR 容错降级）
        """
        with self._conn() as conn:
            # Delete old FTS5 entries before clearing chunks
            old_ids = [
                r["id"]
                for r in conn.execute(
                    "SELECT id FROM chunks WHERE file_id = ?", (file_id,)
                ).fetchall()
            ]
            if old_ids:
                placeholders = ",".join("?" * len(old_ids))
                conn.execute(f"DELETE FROM chunks_fts WHERE rowid IN ({placeholders})", old_ids)
                conn.execute(
                    f"DELETE FROM chunks_fts_trigram WHERE rowid IN ({placeholders})", old_ids
                )
            conn.execute("DELETE FROM chunks WHERE file_id = ?", (file_id,))

            if not chunk_records:
                return

            inserted_ids: list[int] = []
            for record in chunk_records:
                cursor = conn.execute(
                    """INSERT INTO chunks (
                           file_id, qdrant_id, chunk_type, page_num, section, char_count,
                           parent_id, raw_text, embedding_text, parent_text, contextual_prefix
                       )
                       VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                    (
                        file_id,
                        record["qdrant_id"],
                        record["chunk_type"],
                        record["page_num"],
                        record["section"],
                        record["char_count"],
                        record.get("parent_id", 0),
                        record.get("raw_text", ""),
                        record.get("embedding_text", record.get("raw_text", "")),
                        record.get("parent_text", ""),
                        record.get("contextual_prefix", ""),
                    ),
                )
                inserted_ids.append(int(cursor.lastrowid))

            # Batch insert FTS5 entries
            fts_rows = []
            fts_trigram_rows = []
            for chunk_id, r in zip(inserted_ids, chunk_records, strict=True):
                tokenized = r.get("tokenized_text", "")
                if tokenized:
                    fts_rows.append((chunk_id, tokenized))
                raw = r.get("raw_text", "")
                if raw:
                    fts_trigram_rows.append((chunk_id, raw))

            if fts_rows:
                conn.executemany(
                    "INSERT INTO chunks_fts(rowid, tokenized_text) VALUES (?, ?)",
                    fts_rows,
                )
            if fts_trigram_rows:
                conn.executemany(
                    "INSERT INTO chunks_fts_trigram(rowid, raw_text) VALUES (?, ?)",
                    fts_trigram_rows,
                )
    def list_files(
        self,
        status: FileStatus | str | None = None,
        collection: str | None = None,
        tag: str | None = None,
        favorite: bool | None = None,
        kind: str | None = None,
        recent: bool | None = None,
    ) -> list[FileRecord]:
        query = """
            SELECT f.*, fav.id IS NOT NULL AS favorited
            FROM files f
            LEFT JOIN favorites fav ON fav.file_id = f.id
        """
        params: list = []
        clauses = []
        if status:
            clauses.append("f.status = ?")
            params.append(self._normalize_status(status))
        normalized_collection = self._normalize_collection(collection) if collection else ""
        if normalized_collection:
            clauses.append("f.collection = ?")
            params.append(normalized_collection)
        if favorite is True:
            clauses.append("fav.id IS NOT NULL")
        elif favorite is False:
            clauses.append("fav.id IS NULL")
        kind_patterns = self._file_kind_patterns(kind)
        normalized_tag = self._normalize_tag(tag) if tag else ""
        if kind_patterns:
            clauses.append(
                "(" + " OR ".join("LOWER(f.file_name) LIKE ?" for _ in kind_patterns) + ")"
            )
            params.extend(kind_patterns)
        if clauses:
            query += " WHERE " + " AND ".join(clauses)
        query += " ORDER BY f.updated_at DESC"
        if recent is True and not normalized_tag:
            query += " LIMIT 30"
        with self._conn() as conn:
            rows = conn.execute(query, params).fetchall()
        result = [self._file_row_to_dict(r) for r in rows]
        if normalized_tag:
            result = [row for row in result if normalized_tag in row["user_tags"]]
            if recent is True:
                result = result[:30]
        return result
    def get_file_by_path(self, file_path: str | Path) -> FileRecord | None:
        with self._conn() as conn:
            row = conn.execute(
                "SELECT * FROM files WHERE file_path = ?", (str(file_path),)
            ).fetchone()
        return self._file_row_to_dict(row) if row else None
    def get_file_by_id(self, file_id: int) -> FileRecord | None:
        with self._conn() as conn:
            row = conn.execute("SELECT * FROM files WHERE id = ?", (file_id,)).fetchone()
        return self._file_row_to_dict(row) if row else None
    def update_file_metadata(
        self,
        file_id: int,
        collection: str | None = None,
        user_tags: list[str] | None = None,
    ) -> FileRecord | None:
        updates = []
        params: list = []
        if collection is not None:
            updates.append("collection = ?")
            params.append(self._normalize_collection(collection))
        if user_tags is not None:
            updates.append("user_tags = ?")
            params.append(json.dumps(self._normalize_tags(user_tags), ensure_ascii=False))
        if not updates:
            return self.get_file_by_id(file_id)

        params.append(file_id)
        with self._conn() as conn:
            row = conn.execute("SELECT id FROM files WHERE id = ?", (file_id,)).fetchone()
            if row is None:
                return None
            conn.execute(
                f"UPDATE files SET {', '.join(updates)}, updated_at = datetime('now') WHERE id = ?",
                params,
            )
        return self.get_file_by_id(file_id)
    def update_files_metadata(
        self,
        file_ids: list[int],
        collection: str | None = None,
        user_tags: list[str] | None = None,
    ) -> list[FileRecord]:
        updated: list[FileRecord] = []
        for file_id in self._unique_ids(file_ids):
            record = self.update_file_metadata(file_id, collection=collection, user_tags=user_tags)
            if record is not None:
                updated.append(record)
        return updated
    def list_library_facets(self) -> dict:
        with self._conn() as conn:
            collection_rows = conn.execute(
                """
                SELECT collection, COUNT(*) AS count
                FROM files
                GROUP BY collection
                ORDER BY lower(collection)
                """
            ).fetchall()
            rows = conn.execute(
                "SELECT file_name, tags, user_tags, updated_at FROM files"
            ).fetchall()
            favorite_count = conn.execute("SELECT COUNT(*) AS count FROM favorites").fetchone()[
                "count"
            ]

        user_tag_counts: dict[str, int] = {}
        document_tag_counts: dict[str, int] = {}
        for row in rows:
            for tag_name in self._parse_json_list(row["user_tags"]):
                user_tag_counts[tag_name] = user_tag_counts.get(tag_name, 0) + 1
            for tag_name in self._parse_json_list(row["tags"]):
                document_tag_counts[tag_name] = document_tag_counts.get(tag_name, 0) + 1

        type_counts = {
            "pdf": sum(1 for row in rows if self._file_kind(row["file_name"]) == "pdf"),
            "markdown": sum(1 for row in rows if self._file_kind(row["file_name"]) == "markdown"),
            "image": sum(1 for row in rows if self._file_kind(row["file_name"]) == "image"),
            "code": sum(1 for row in rows if self._file_kind(row["file_name"]) == "code"),
        }

        return {
            "collections": [
                {"name": row["collection"] or DEFAULT_COLLECTION, "count": row["count"]}
                for row in collection_rows
            ],
            "user_tags": self._facet_rows(user_tag_counts),
            "document_tags": self._facet_rows(document_tag_counts),
            "favorites": favorite_count,
            "total_files": len(rows),
            "types": type_counts,
            "recent": min(len(rows), 30),
        }
    @staticmethod
    def _file_kind(file_name: str) -> str:
        suffix = Path(file_name or "").suffix.lower()
        if suffix == ".pdf":
            return "pdf"
        if suffix in {".md", ".markdown"}:
            return "markdown"
        if suffix in {".jpg", ".jpeg", ".png", ".webp", ".heic", ".heif"}:
            return "image"
        if suffix in {".py", ".rs", ".ts", ".tsx", ".js", ".jsx", ".css", ".sh"}:
            return "code"
        if suffix == ".docx":
            return "docx"
        if suffix == ".txt":
            return "text"
        return "other"
    @staticmethod
    def _file_kind_patterns(kind: str | None) -> list[str]:
        normalized = (kind or "").strip().lower()
        groups = {
            "pdf": [".pdf"],
            "markdown": [".md", ".markdown"],
            "image": [".jpg", ".jpeg", ".png", ".webp", ".heic", ".heif"],
            "code": [".py", ".rs", ".ts", ".tsx", ".js", ".jsx", ".css", ".sh"],
            "docx": [".docx"],
            "text": [".txt"],
        }
        return [f"%{suffix}" for suffix in groups.get(normalized, [])]
