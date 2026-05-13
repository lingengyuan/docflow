"""Chunk, embedding cache, and FTS operations for DocStore."""

# mypy: disable-error-code="attr-defined"

from __future__ import annotations

from typing import TYPE_CHECKING

from src.ingest.store_shared import _fts5_phrase

if TYPE_CHECKING:
    import numpy as np


class StoreVectorMixin:
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
            conn.execute("DELETE FROM note_source_links")
            conn.execute("DELETE FROM files")
    def list_file_chunks(self, file_id: int) -> list[dict]:
        """返回某文件所有 chunk 元数据，按 SQLite chunk id 排序。"""
        with self._conn() as conn:
            rows = conn.execute(
                """
                SELECT id, file_id, qdrant_id, chunk_type, page_num, section,
                       char_count, parent_id, raw_text, embedding_text,
                       parent_text, contextual_prefix, created_at
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
                batch = unique_ids[i : i + 500]
                placeholders = ",".join("?" * len(batch))
                rows = conn.execute(
                    f"""
                    SELECT qdrant_id, parent_id, raw_text, embedding_text,
                           parent_text, contextual_prefix
                    FROM chunks
                    WHERE qdrant_id IN ({placeholders})
                    """,
                    batch,
                ).fetchall()
                for row in rows:
                    result[int(row["qdrant_id"])] = dict(row)
        return result
    def get_cached_embeddings(
        self, model_name: str, text_hashes: list[str]
    ) -> dict[str, np.ndarray]:
        import numpy as np

        unique_hashes = list(dict.fromkeys(text_hashes))
        if not unique_hashes:
            return {}

        result: dict[str, np.ndarray] = {}
        with self._conn() as conn:
            for i in range(0, len(unique_hashes), 500):
                batch = unique_hashes[i : i + 500]
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
                    result[row["text_hash"]] = np.frombuffer(row["vector"], dtype=np.float32).copy()
        return result
    def put_cached_embeddings(self, model_name: str, vectors_by_hash: dict[str, np.ndarray]):
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
        params: list = [_fts5_phrase(query), subq_limit]
        if file_filter:
            placeholders = ",".join("?" * len(file_filter))
            path_placeholders = ",".join("?" * len(file_filter))
            sql += (
                f" WHERE fi.file_name IN ({placeholders}) "
                f"OR fi.file_path IN ({path_placeholders})"
            )
            params.extend(file_filter)
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

        [r["id"] for r in rows]
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
            path_placeholders = ",".join("?" * len(file_filter))
            sql += (
                f" WHERE fi.file_name IN ({placeholders}) "
                f"OR fi.file_path IN ({path_placeholders})"
            )
            params.extend(file_filter)
            params.extend(file_filter)
        sql += " ORDER BY fts.score DESC LIMIT ?"
        params.append(limit)

        with self._conn() as conn:
            rows = conn.execute(sql, params).fetchall()
        return [dict(r) for r in rows]
