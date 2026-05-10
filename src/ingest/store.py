"""
DocStore — SQLite 元数据存储。

Schema:
  files  — 文件状态追踪（pending / processing / done / error）
  chunks — chunk 元数据索引（与 Qdrant 向量 ID 对应）
"""

from __future__ import annotations

import hashlib
import sqlite3
import threading
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
import json


DEFAULT_COLLECTION = "Inbox"


# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------

@dataclass
class FileRecord:
    id: int
    file_path: str
    file_name: str
    file_hash: str
    status: str          # pending | processing | done | error
    total_pages: int
    is_scanned: bool
    chunk_count: int
    error_msg: str
    created_at: str
    updated_at: str


# ---------------------------------------------------------------------------
# DocStore
# ---------------------------------------------------------------------------

class DocStore:
    def __init__(self, db_path: str | Path):
        self.db_path = str(db_path)
        self._local = threading.local()
        self._init_db()
        self._migrate()

    # ------------------------------------------------------------------
    # Schema
    # ------------------------------------------------------------------

    def _init_db(self):
        with self._conn() as conn:
            conn.executescript("""
                CREATE TABLE IF NOT EXISTS files (
                    id           INTEGER PRIMARY KEY AUTOINCREMENT,
                    file_path    TEXT    NOT NULL UNIQUE,
                    file_name    TEXT    NOT NULL,
                    file_hash    TEXT    NOT NULL,
                    status       TEXT    NOT NULL DEFAULT 'pending',
                    total_pages  INTEGER NOT NULL DEFAULT 0,
                    is_scanned   INTEGER NOT NULL DEFAULT 0,
                    chunk_count  INTEGER NOT NULL DEFAULT 0,
                    error_msg    TEXT    NOT NULL DEFAULT '',
                    tags         TEXT    NOT NULL DEFAULT '[]',
                    collection   TEXT    NOT NULL DEFAULT 'Inbox',
                    user_tags    TEXT    NOT NULL DEFAULT '[]',
                    mtime_ns     INTEGER NOT NULL DEFAULT 0,
                    created_at   TEXT    NOT NULL DEFAULT (datetime('now')),
                    updated_at   TEXT    NOT NULL DEFAULT (datetime('now'))
                );

                CREATE TABLE IF NOT EXISTS chunks (
                    id           INTEGER PRIMARY KEY AUTOINCREMENT,
                    file_id      INTEGER NOT NULL REFERENCES files(id),
                    qdrant_id    INTEGER NOT NULL,
                    chunk_type   TEXT    NOT NULL,
                    page_num     INTEGER NOT NULL,
                    section      TEXT    NOT NULL DEFAULT '',
                    char_count   INTEGER NOT NULL DEFAULT 0,
                    parent_id    INTEGER NOT NULL DEFAULT 0,
                    raw_text     TEXT    NOT NULL DEFAULT '',
                    embedding_text TEXT  NOT NULL DEFAULT '',
                    parent_text  TEXT    NOT NULL DEFAULT '',
                    contextual_prefix TEXT NOT NULL DEFAULT '',
                    created_at   TEXT    NOT NULL DEFAULT (datetime('now'))
                );

                CREATE TABLE IF NOT EXISTS history (
                    id          INTEGER PRIMARY KEY AUTOINCREMENT,
                    question    TEXT    NOT NULL,
                    answer      TEXT    NOT NULL,
                    citations   TEXT    NOT NULL DEFAULT '[]',
                    file_filter TEXT    NOT NULL DEFAULT '[]',
                    conversation_id INTEGER NOT NULL DEFAULT 0,
                    created_at  TEXT    NOT NULL DEFAULT (datetime('now'))
                );

                CREATE TABLE IF NOT EXISTS conversations (
                    id         INTEGER PRIMARY KEY AUTOINCREMENT,
                    title      TEXT    NOT NULL DEFAULT '',
                    created_at TEXT    NOT NULL DEFAULT (datetime('now')),
                    updated_at TEXT    NOT NULL DEFAULT (datetime('now'))
                );

                CREATE TABLE IF NOT EXISTS messages (
                    id              INTEGER PRIMARY KEY AUTOINCREMENT,
                    conversation_id INTEGER NOT NULL REFERENCES conversations(id),
                    role            TEXT    NOT NULL,
                    content         TEXT    NOT NULL,
                    citations       TEXT    NOT NULL DEFAULT '[]',
                    file_filter     TEXT    NOT NULL DEFAULT '[]',
                    created_at      TEXT    NOT NULL DEFAULT (datetime('now'))
                );

                CREATE TABLE IF NOT EXISTS favorites (
                    id         INTEGER PRIMARY KEY AUTOINCREMENT,
                    file_id    INTEGER NOT NULL UNIQUE REFERENCES files(id),
                    created_at TEXT    NOT NULL DEFAULT (datetime('now'))
                );

                CREATE INDEX IF NOT EXISTS idx_chunks_file_id ON chunks(file_id);
                CREATE INDEX IF NOT EXISTS idx_files_hash     ON files(file_hash);
                CREATE INDEX IF NOT EXISTS idx_files_status   ON files(status);
                CREATE INDEX IF NOT EXISTS idx_messages_conversation_id ON messages(conversation_id, id);

                CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts USING fts5(
                    tokenized_text,
                    tokenize='unicode61'
                );

                CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts_trigram USING fts5(
                    raw_text,
                    tokenize='trigram'
                );

                CREATE VIRTUAL TABLE IF NOT EXISTS history_fts USING fts5(
                    question,
                    tokenize='trigram'
                );

                CREATE TABLE IF NOT EXISTS embedding_cache (
                    model_name TEXT NOT NULL,
                    text_hash  TEXT NOT NULL,
                    vector     BLOB NOT NULL,
                    created_at TEXT NOT NULL DEFAULT (datetime('now')),
                    PRIMARY KEY (model_name, text_hash)
                );
            """)

    def _migrate(self):
        """增量迁移：为已有 DB 添加新列（幂等）。"""
        migrations = [
            "ALTER TABLE files ADD COLUMN tags TEXT NOT NULL DEFAULT '[]'",
            "ALTER TABLE files ADD COLUMN collection TEXT NOT NULL DEFAULT 'Inbox'",
            "ALTER TABLE files ADD COLUMN user_tags TEXT NOT NULL DEFAULT '[]'",
            "ALTER TABLE files ADD COLUMN mtime_ns INTEGER NOT NULL DEFAULT 0",
            "ALTER TABLE chunks ADD COLUMN parent_id INTEGER NOT NULL DEFAULT 0",
            "ALTER TABLE chunks ADD COLUMN raw_text TEXT NOT NULL DEFAULT ''",
            "ALTER TABLE chunks ADD COLUMN embedding_text TEXT NOT NULL DEFAULT ''",
            "ALTER TABLE chunks ADD COLUMN parent_text TEXT NOT NULL DEFAULT ''",
            "ALTER TABLE chunks ADD COLUMN contextual_prefix TEXT NOT NULL DEFAULT ''",
            "ALTER TABLE history ADD COLUMN conversation_id INTEGER NOT NULL DEFAULT 0",
        ]
        with self._conn() as conn:
            for sql in migrations:
                try:
                    conn.execute(sql)
                except sqlite3.OperationalError:
                    pass  # 列已存在
            conn.execute("CREATE INDEX IF NOT EXISTS idx_chunks_parent_id ON chunks(file_id, parent_id)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_files_collection ON files(collection)")

    @contextmanager
    def _conn(self):
        conn = getattr(self._local, "conn", None)
        if conn is None:
            conn = sqlite3.connect(self.db_path)
            conn.row_factory = sqlite3.Row
            conn.execute("PRAGMA journal_mode=WAL")
            self._local.conn = conn
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise

    # ------------------------------------------------------------------
    # File operations
    # ------------------------------------------------------------------

    @staticmethod
    def compute_hash(file_path: str | Path) -> str:
        h = hashlib.sha256()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(65536), b""):
                h.update(chunk)
        return h.hexdigest()

    @staticmethod
    def compute_text_hash(text: str) -> str:
        return hashlib.sha256(text.encode("utf-8")).hexdigest()

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
        if row["status"] in ("pending", "error", "processing"):
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
        status: str = "pending",
        total_pages: int = 0,
        is_scanned: bool = False,
        tags: str = "[]",
        mtime_ns: int = 0,
    ) -> int:
        path = str(file_path)
        with self._conn() as conn:
            conn.execute("""
                INSERT INTO files (file_path, file_name, file_hash, status, total_pages, is_scanned, tags, mtime_ns)
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
            """, (path, file_name, file_hash, status, total_pages, int(is_scanned), tags, mtime_ns))
            file_id = conn.execute(
                "SELECT id FROM files WHERE file_path = ?", (path,)
            ).fetchone()["id"]
        return file_id

    def reset_processing_files(self) -> int:
        """启动时调用：将残留的 'processing' 状态（上次 server 崩溃遗留）重置为 'error'。
        由于 needs_ingest() 对 'error' 返回 True，这些文件会在启动扫描时自动重新入队。
        """
        with self._conn() as conn:
            result = conn.execute(
                "UPDATE files SET status='error', error_msg='Interrupted (server restart)', "
                "updated_at=datetime('now') WHERE status='processing'"
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
                        conn.execute(f"DELETE FROM chunks_fts_trigram WHERE rowid IN ({ph})", chunk_ids)
                    # 删除 chunks、favorites、file 记录
                    conn.execute("DELETE FROM chunks WHERE file_id = ?", (file_id,))
                    conn.execute("DELETE FROM favorites WHERE file_id = ?", (file_id,))
                    conn.execute("DELETE FROM files WHERE id = ?", (file_id,))
                    removed.append({
                        "file_name": row["file_name"],
                        "qdrant_ids": qids,
                    })
        return removed

    def set_status(self, file_path: str | Path, status: str, error_msg: str = ""):
        with self._conn() as conn:
            conn.execute("""
                UPDATE files
                SET status = ?, error_msg = ?, updated_at = datetime('now')
                WHERE file_path = ?
            """, (status, error_msg, str(file_path)))

    def set_chunk_count(self, file_path: str | Path, count: int):
        with self._conn() as conn:
            conn.execute("""
                UPDATE files
                SET chunk_count = ?, updated_at = datetime('now')
                WHERE file_path = ?
            """, (count, str(file_path)))

    def add_chunks(self, file_id: int, chunk_records: list[dict]):
        """
        chunk_records: list of {qdrant_id, chunk_type, page_num, section, char_count,
          parent_id?, raw_text?, embedding_text?, parent_text?, contextual_prefix?, tokenized_text?}
        同步写入两个 FTS5 索引：
          chunks_fts         — jieba 预分词，精确匹配
          chunks_fts_trigram — trigram，子串匹配（OCR 容错降级）
        """
        with self._conn() as conn:
            # Delete old FTS5 entries before clearing chunks
            old_ids = [r["id"] for r in conn.execute(
                "SELECT id FROM chunks WHERE file_id = ?", (file_id,)
            ).fetchall()]
            if old_ids:
                placeholders = ",".join("?" * len(old_ids))
                conn.execute(f"DELETE FROM chunks_fts WHERE rowid IN ({placeholders})", old_ids)
                conn.execute(f"DELETE FROM chunks_fts_trigram WHERE rowid IN ({placeholders})", old_ids)
            conn.execute("DELETE FROM chunks WHERE file_id = ?", (file_id,))

            if not chunk_records:
                return

            # Batch insert chunks
            conn.executemany(
                """INSERT INTO chunks (
                       file_id, qdrant_id, chunk_type, page_num, section, char_count,
                       parent_id, raw_text, embedding_text, parent_text, contextual_prefix
                   )
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                [(
                    file_id,
                    r["qdrant_id"],
                    r["chunk_type"],
                    r["page_num"],
                    r["section"],
                    r["char_count"],
                    r.get("parent_id", 0),
                    r.get("raw_text", ""),
                    r.get("embedding_text", r.get("raw_text", "")),
                    r.get("parent_text", ""),
                    r.get("contextual_prefix", ""),
                )
                 for r in chunk_records],
            )
            # Compute rowid range (AUTOINCREMENT guarantees contiguous IDs within a single executemany)
            last_id = conn.execute("SELECT last_insert_rowid()").fetchone()[0]
            first_id = last_id - len(chunk_records) + 1

            # Batch insert FTS5 entries
            fts_rows = []
            fts_trigram_rows = []
            for i, r in enumerate(chunk_records):
                chunk_id = first_id + i
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

    # ------------------------------------------------------------------
    # Query
    # ------------------------------------------------------------------

    def list_files(
        self,
        status: str | None = None,
        collection: str | None = None,
        tag: str | None = None,
        favorite: bool | None = None,
        kind: str | None = None,
        recent: bool | None = None,
    ) -> list[dict]:
        query = """
            SELECT f.*, fav.id IS NOT NULL AS favorited
            FROM files f
            LEFT JOIN favorites fav ON fav.file_id = f.id
        """
        params: list = []
        clauses = []
        if status:
            clauses.append("f.status = ?")
            params.append(status)
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
            clauses.append("(" + " OR ".join("LOWER(f.file_name) LIKE ?" for _ in kind_patterns) + ")")
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

    def get_file_by_path(self, file_path: str | Path) -> dict | None:
        with self._conn() as conn:
            row = conn.execute(
                "SELECT * FROM files WHERE file_path = ?", (str(file_path),)
            ).fetchone()
        return self._file_row_to_dict(row) if row else None

    def get_file_by_id(self, file_id: int) -> dict | None:
        with self._conn() as conn:
            row = conn.execute(
                "SELECT * FROM files WHERE id = ?", (file_id,)
            ).fetchone()
        return self._file_row_to_dict(row) if row else None

    def update_file_metadata(
        self,
        file_id: int,
        collection: str | None = None,
        user_tags: list[str] | None = None,
    ) -> dict | None:
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
    ) -> list[dict]:
        updated: list[dict] = []
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
            rows = conn.execute("SELECT file_name, tags, user_tags, updated_at FROM files").fetchall()
            favorite_count = conn.execute("SELECT COUNT(*) AS count FROM favorites").fetchone()["count"]

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

    def get_file_qdrant_ids(self, file_id: int) -> list[int]:
        """返回某文件所有 chunk 的 Qdrant point ID（用于重新索引时清理旧向量）。"""
        with self._conn() as conn:
            rows = conn.execute(
                "SELECT qdrant_id FROM chunks WHERE file_id = ? ORDER BY id", (file_id,)
            ).fetchall()
        return [r["qdrant_id"] for r in rows]

    def max_qdrant_id(self) -> int:
        """Return the highest Qdrant point ID recorded in SQLite, or -1 when empty."""
        with self._conn() as conn:
            row = conn.execute("SELECT MAX(qdrant_id) AS max_id FROM chunks").fetchone()
        return int(row["max_id"]) if row and row["max_id"] is not None else -1

    def list_chunk_index(self) -> list[dict]:
        """Return all chunk rows needed for consistency checks and Qdrant-only rebuilds."""
        with self._conn() as conn:
            rows = conn.execute(
                """
                SELECT c.id, c.file_id, c.qdrant_id, c.chunk_type, c.page_num, c.section,
                       c.char_count, c.parent_id, c.raw_text, c.embedding_text, c.parent_text,
                       c.contextual_prefix, f.file_name, f.file_path
                FROM chunks c
                JOIN files f ON f.id = c.file_id
                ORDER BY c.id
                """
            ).fetchall()
        return [dict(r) for r in rows]

    def list_file_chunk_counts(self) -> list[dict]:
        """Return declared and actual chunk counts for every file record."""
        with self._conn() as conn:
            rows = conn.execute(
                """
                SELECT f.id, f.file_name, f.file_path, f.status, f.chunk_count,
                       COUNT(c.id) AS actual_chunk_count
                FROM files f
                LEFT JOIN chunks c ON c.file_id = f.id
                GROUP BY f.id
                ORDER BY f.id
                """
            ).fetchall()
        return [dict(r) for r in rows]

    def clear_index(self) -> None:
        """Clear indexed files and chunks while preserving history and embedding cache."""
        with self._conn() as conn:
            conn.execute("DELETE FROM chunks_fts")
            conn.execute("DELETE FROM chunks_fts_trigram")
            conn.execute("DELETE FROM chunks")
            conn.execute("DELETE FROM favorites")
            conn.execute("DELETE FROM files")

    def list_file_chunks(self, file_id: int) -> list[dict]:
        """返回某文件所有 chunk 元数据，按 SQLite chunk id 排序。"""
        with self._conn() as conn:
            rows = conn.execute(
                """
                SELECT id, file_id, qdrant_id, chunk_type, page_num, section, char_count,
                       parent_id, raw_text, embedding_text, parent_text, contextual_prefix, created_at
                FROM chunks
                WHERE file_id = ?
                ORDER BY id
                """,
                (file_id,),
            ).fetchall()
        return [dict(r) for r in rows]

    def get_chunk_context_by_qdrant_ids(self, qdrant_ids: list[int]) -> dict[int, dict]:
        """Return stored raw/parent context keyed by Qdrant point id."""
        unique_ids = list(dict.fromkeys(qdrant_ids))
        if not unique_ids:
            return {}

        result: dict[int, dict] = {}
        with self._conn() as conn:
            for i in range(0, len(unique_ids), 500):
                batch = unique_ids[i:i + 500]
                placeholders = ",".join("?" * len(batch))
                rows = conn.execute(
                    f"""
                    SELECT qdrant_id, parent_id, raw_text, embedding_text, parent_text, contextual_prefix
                    FROM chunks
                    WHERE qdrant_id IN ({placeholders})
                    """,
                    batch,
                ).fetchall()
                for row in rows:
                    result[int(row["qdrant_id"])] = dict(row)
        return result

    # ------------------------------------------------------------------
    # Embedding cache
    # ------------------------------------------------------------------

    def get_cached_embeddings(self, model_name: str, text_hashes: list[str]) -> dict[str, "np.ndarray"]:
        import numpy as np

        unique_hashes = list(dict.fromkeys(text_hashes))
        if not unique_hashes:
            return {}

        result: dict[str, np.ndarray] = {}
        with self._conn() as conn:
            for i in range(0, len(unique_hashes), 500):
                batch = unique_hashes[i:i + 500]
                placeholders = ",".join("?" * len(batch))
                rows = conn.execute(
                    f"""
                    SELECT text_hash, vector
                    FROM embedding_cache
                    WHERE model_name = ? AND text_hash IN ({placeholders})
                    """,
                    [model_name, *batch],
                ).fetchall()
                for row in rows:
                    result[row["text_hash"]] = np.frombuffer(
                        row["vector"], dtype=np.float32
                    ).copy()
        return result

    def put_cached_embeddings(self, model_name: str, vectors_by_hash: dict[str, "np.ndarray"]):
        import numpy as np

        if not vectors_by_hash:
            return

        rows = [
            (
                model_name,
                text_hash,
                np.asarray(vector, dtype=np.float32).tobytes(),
            )
            for text_hash, vector in vectors_by_hash.items()
        ]
        with self._conn() as conn:
            conn.executemany(
                """
                INSERT OR REPLACE INTO embedding_cache (model_name, text_hash, vector)
                VALUES (?, ?, ?)
                """,
                rows,
            )

    # ------------------------------------------------------------------
    # History
    # ------------------------------------------------------------------

    def add_history(
        self,
        question: str,
        answer: str,
        citations_json: str,
        file_filter_json: str = "[]",
        conversation_id: int = 0,
    ) -> int:
        with self._conn() as conn:
            conn.execute(
                """
                INSERT INTO history (question, answer, citations, file_filter, conversation_id)
                VALUES (?, ?, ?, ?, ?)
                """,
                (question, answer, citations_json, file_filter_json, conversation_id),
            )
            row_id = conn.execute("SELECT last_insert_rowid()").fetchone()[0]
            conn.execute(
                "INSERT INTO history_fts(rowid, question) VALUES (?, ?)",
                (row_id, question),
            )
        return row_id

    def search_history(self, query: str, limit: int = 20) -> list[dict]:
        """全文搜索历史问题，返回匹配的历史记录（含 answer、citations）。"""
        try:
            sql = """
                SELECT h.id, h.question, h.answer, h.citations, h.file_filter,
                       h.conversation_id, h.created_at
                FROM (
                    SELECT rowid, -rank AS score
                    FROM history_fts
                    WHERE history_fts MATCH ?
                    ORDER BY rank
                    LIMIT ?
                ) fts
                JOIN history h ON h.id = fts.rowid
                ORDER BY fts.score DESC
            """
            with self._conn() as conn:
                rows = conn.execute(sql, [query, limit]).fetchall()
            return [dict(r) for r in rows]
        except Exception:
            return []

    def list_history(self, limit: int = 50) -> list[dict]:
        with self._conn() as conn:
            rows = conn.execute(
                "SELECT * FROM history ORDER BY created_at DESC LIMIT ?", (limit,)
            ).fetchall()
        return [dict(r) for r in rows]

    def clear_history(self):
        with self._conn() as conn:
            conn.execute("DELETE FROM history_fts")
            conn.execute("DELETE FROM history")

    # ------------------------------------------------------------------
    # Conversations
    # ------------------------------------------------------------------

    def create_conversation(self, title: str = "") -> int:
        with self._conn() as conn:
            conn.execute(
                "INSERT INTO conversations (title) VALUES (?)",
                (title.strip(),),
            )
            return conn.execute("SELECT last_insert_rowid()").fetchone()[0]

    def get_conversation(self, conversation_id: int) -> dict | None:
        with self._conn() as conn:
            row = conn.execute(
                "SELECT * FROM conversations WHERE id = ?",
                (conversation_id,),
            ).fetchone()
        return dict(row) if row else None

    def list_conversations(self, limit: int = 50) -> list[dict]:
        with self._conn() as conn:
            rows = conn.execute(
                """
                SELECT c.*,
                       (
                           SELECT COUNT(*)
                           FROM messages m
                           WHERE m.conversation_id = c.id
                       ) AS message_count,
                       (
                           SELECT m.content
                           FROM messages m
                           WHERE m.conversation_id = c.id
                           ORDER BY m.id DESC
                           LIMIT 1
                       ) AS last_message
                FROM conversations c
                ORDER BY c.updated_at DESC, c.id DESC
                LIMIT ?
                """,
                (limit,),
            ).fetchall()
        return [dict(row) for row in rows]

    def delete_conversation(self, conversation_id: int) -> bool:
        with self._conn() as conn:
            existing = conn.execute(
                "SELECT id FROM conversations WHERE id = ?",
                (conversation_id,),
            ).fetchone()
            if existing is None:
                return False
            conn.execute("DELETE FROM messages WHERE conversation_id = ?", (conversation_id,))
            conn.execute("DELETE FROM conversations WHERE id = ?", (conversation_id,))
            return True

    def add_message(
        self,
        conversation_id: int,
        role: str,
        content: str,
        citations_json: str = "[]",
        file_filter_json: str = "[]",
    ) -> int:
        if role not in {"user", "assistant"}:
            raise ValueError(f"Unsupported message role: {role}")
        with self._conn() as conn:
            conversation = conn.execute(
                "SELECT title FROM conversations WHERE id = ?",
                (conversation_id,),
            ).fetchone()
            if conversation is None:
                raise ValueError(f"Conversation not found: {conversation_id}")
            conn.execute(
                """
                INSERT INTO messages (conversation_id, role, content, citations, file_filter)
                VALUES (?, ?, ?, ?, ?)
                """,
                (conversation_id, role, content, citations_json, file_filter_json),
            )
            message_id = conn.execute("SELECT last_insert_rowid()").fetchone()[0]
            title_update = ""
            if role == "user" and not conversation["title"].strip():
                title_update = ", title = ?"
                params = [self._conversation_title(content), conversation_id]
            else:
                params = [conversation_id]
            conn.execute(
                f"UPDATE conversations SET updated_at = datetime('now'){title_update} WHERE id = ?",
                params,
            )
            return message_id

    def list_messages(self, conversation_id: int, limit: int | None = None) -> list[dict]:
        if limit is None:
            query = "SELECT * FROM messages WHERE conversation_id = ? ORDER BY id"
            params = (conversation_id,)
        else:
            query = """
                SELECT *
                FROM (
                    SELECT *
                    FROM messages
                    WHERE conversation_id = ?
                    ORDER BY id DESC
                    LIMIT ?
                )
                ORDER BY id
            """
            params = (conversation_id, limit)
        with self._conn() as conn:
            rows = conn.execute(query, params).fetchall()
        return [dict(row) for row in rows]

    @staticmethod
    def _conversation_title(text: str, max_chars: int = 40) -> str:
        title = " ".join(text.strip().split())
        if len(title) <= max_chars:
            return title
        return title[:max_chars - 1] + "..."

    # ------------------------------------------------------------------
    # Favorites
    # ------------------------------------------------------------------

    def toggle_favorite(self, file_id: int) -> bool:
        """添加/取消收藏。返回 True=已收藏，False=已取消。"""
        with self._conn() as conn:
            existing = conn.execute(
                "SELECT id FROM favorites WHERE file_id = ?", (file_id,)
            ).fetchone()
            if existing:
                conn.execute("DELETE FROM favorites WHERE file_id = ?", (file_id,))
                return False
            else:
                conn.execute("INSERT INTO favorites (file_id) VALUES (?)", (file_id,))
                return True

    def set_favorites(self, file_ids: list[int], favorited: bool) -> list[int]:
        ids = self._unique_ids(file_ids)
        if not ids:
            return []
        with self._conn() as conn:
            placeholders = ",".join("?" * len(ids))
            existing_rows = conn.execute(
                f"SELECT id FROM files WHERE id IN ({placeholders})",
                ids,
            ).fetchall()
            existing_ids = [row["id"] for row in existing_rows]
            if not existing_ids:
                return []
            if favorited:
                conn.executemany(
                    "INSERT OR IGNORE INTO favorites (file_id) VALUES (?)",
                    [(file_id,) for file_id in existing_ids],
                )
            else:
                placeholders = ",".join("?" * len(existing_ids))
                conn.execute(f"DELETE FROM favorites WHERE file_id IN ({placeholders})", existing_ids)
        return existing_ids

    def is_favorite(self, file_id: int) -> bool:
        with self._conn() as conn:
            row = conn.execute(
                "SELECT id FROM favorites WHERE file_id = ?", (file_id,)
            ).fetchone()
        return row is not None

    def list_favorites(self) -> list[dict]:
        with self._conn() as conn:
            rows = conn.execute("""
                SELECT f.*, 1 AS favorited FROM files f
                INNER JOIN favorites fav ON f.id = fav.file_id
                ORDER BY fav.created_at DESC
            """).fetchall()
        return [self._file_row_to_dict(r) for r in rows]

    # ------------------------------------------------------------------
    # File metadata helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _unique_ids(file_ids: list[int]) -> list[int]:
        return [int(file_id) for file_id in dict.fromkeys(file_ids) if int(file_id) > 0]

    @staticmethod
    def _parse_json_list(value) -> list[str]:
        if isinstance(value, list):
            raw = value
        else:
            try:
                raw = json.loads(value or "[]")
            except Exception:
                raw = []
        if not isinstance(raw, list):
            return []
        return [str(item).strip() for item in raw if str(item).strip()]

    @classmethod
    def _normalize_tags(cls, values: list[str]) -> list[str]:
        seen = set()
        tags: list[str] = []
        for value in values:
            tag = cls._normalize_tag(value)
            if tag and tag not in seen:
                seen.add(tag)
                tags.append(tag)
        return tags

    @staticmethod
    def _normalize_tag(value: str | None) -> str:
        return str(value or "").strip().lstrip("#")[:40]

    @staticmethod
    def _normalize_collection(value: str | None) -> str:
        collection = str(value or "").strip()
        return collection[:80] if collection else DEFAULT_COLLECTION

    @classmethod
    def _file_row_to_dict(cls, row: sqlite3.Row) -> dict:
        data = dict(row)
        data["tags"] = cls._parse_json_list(data.get("tags"))
        data["collection"] = cls._normalize_collection(data.get("collection"))
        data["user_tags"] = cls._parse_json_list(data.get("user_tags"))
        data["favorited"] = bool(data.get("favorited", False))
        return data

    @staticmethod
    def _facet_rows(counts: dict[str, int]) -> list[dict]:
        return [
            {"name": name, "count": counts[name]}
            for name in sorted(counts, key=lambda item: item.lower())
        ]

    # ------------------------------------------------------------------
    # FTS5 全文检索
    # ------------------------------------------------------------------

    def search_fts_trigram(
        self,
        query: str,
        file_filter: list[str] | None,
        limit: int,
    ) -> list[dict]:
        """
        Trigram 子串全文检索（降级层）。
        query: 原始查询字符串，FTS5 trigram tokenizer 会自动拆成 3-gram。
        返回格式与 search_fts() 相同。
        """
        subq_limit = limit * 3 if file_filter else limit
        sql = """
            SELECT c.qdrant_id, c.page_num, c.section, c.chunk_type, c.char_count,
                   c.parent_id, c.raw_text, c.parent_text, c.contextual_prefix,
                   fi.file_name, fi.file_path, fts.score
            FROM (
                SELECT rowid, -rank AS score
                FROM chunks_fts_trigram
                WHERE chunks_fts_trigram MATCH ?
                ORDER BY rank
                LIMIT ?
            ) fts
            JOIN chunks c ON c.id = fts.rowid
            JOIN files fi ON fi.id = c.file_id
        """
        params: list = [query, subq_limit]
        if file_filter:
            placeholders = ",".join("?" * len(file_filter))
            sql += f" WHERE fi.file_name IN ({placeholders})"
            params.extend(file_filter)
        sql += " ORDER BY fts.score DESC LIMIT ?"
        params.append(limit)

        with self._conn() as conn:
            rows = conn.execute(sql, params).fetchall()
        return [dict(r) for r in rows]

    def backfill_fts(self, qdrant_client, collection_name: str = "docflow") -> int:
        """
        将已有 chunks 中缺失 FTS5 记录的条目从 Qdrant 拉取文本并回填。
        用于 DB 迁移（旧版 DB 无 FTS5 记录）。
        返回回填的条目数。
        """
        import jieba

        with self._conn() as conn:
            # 找出没有 FTS5 记录的 chunks（两张表都缺）
            rows = conn.execute("""
                SELECT c.id, c.qdrant_id
                FROM chunks c
                LEFT JOIN chunks_fts ON chunks_fts.rowid = c.id
                WHERE chunks_fts.rowid IS NULL
            """).fetchall()

        if not rows:
            return 0

        chunk_ids = [r["id"] for r in rows]
        qdrant_ids = [r["qdrant_id"] for r in rows]
        id_to_chunk_id = {r["qdrant_id"]: r["id"] for r in rows}

        # Qdrant 批量拉取（每次最多 100 个）
        BATCH = 100
        filled = 0
        for i in range(0, len(qdrant_ids), BATCH):
            batch_qids = qdrant_ids[i : i + BATCH]
            records = qdrant_client.retrieve(
                collection_name=collection_name,
                ids=batch_qids,
                with_payload=True,
            )
            with self._conn() as conn:
                fts_batch = []
                trigram_batch = []
                for rec in records:
                    text = rec.payload.get("text", "") if rec.payload else ""
                    if not text:
                        continue
                    tokenized = " ".join(t for t in jieba.cut(text.lower()) if t.strip())
                    chunk_id = id_to_chunk_id.get(rec.id)
                    if chunk_id and tokenized:
                        fts_batch.append((chunk_id, tokenized))
                        trigram_batch.append((chunk_id, text))
                        filled += 1
                if fts_batch:
                    conn.executemany(
                        "INSERT OR IGNORE INTO chunks_fts(rowid, tokenized_text) VALUES (?, ?)",
                        fts_batch,
                    )
                if trigram_batch:
                    conn.executemany(
                        "INSERT OR IGNORE INTO chunks_fts_trigram(rowid, raw_text) VALUES (?, ?)",
                        trigram_batch,
                    )
        return filled

    def search_fts(
        self,
        fts_query: str,
        file_filter: list[str] | None,
        limit: int,
    ) -> list[dict]:
        """
        BM25 全文检索（SQLite FTS5）。
        fts_query: FTS5 MATCH 表达式，如 '"机器" OR "学习"'
        返回: [{qdrant_id, page_num, section, chunk_type, file_name, file_path, score}]
        score 为正值（-rank），值越大越相关。
        """
        # 子查询先在 FTS5 内排序，再 JOIN 元数据表（FTS5 rank 在子查询中更稳定）
        subq_limit = limit * 3 if file_filter else limit
        sql = """
            SELECT c.qdrant_id, c.page_num, c.section, c.chunk_type, c.char_count,
                   c.parent_id, c.raw_text, c.parent_text, c.contextual_prefix,
                   fi.file_name, fi.file_path, fts.score
            FROM (
                SELECT rowid, -rank AS score
                FROM chunks_fts
                WHERE chunks_fts MATCH ?
                ORDER BY rank
                LIMIT ?
            ) fts
            JOIN chunks c ON c.id = fts.rowid
            JOIN files fi ON fi.id = c.file_id
        """
        params: list = [fts_query, subq_limit]
        if file_filter:
            placeholders = ",".join("?" * len(file_filter))
            sql += f" WHERE fi.file_name IN ({placeholders})"
            params.extend(file_filter)
        sql += " ORDER BY fts.score DESC LIMIT ?"
        params.append(limit)

        with self._conn() as conn:
            rows = conn.execute(sql, params).fetchall()
        return [dict(r) for r in rows]
