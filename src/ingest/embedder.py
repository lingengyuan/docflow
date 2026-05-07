"""
Embedder — Qwen3-Embedding-0.6B 批量 Embedding，写入 Qdrant。

使用 sentence-transformers 加载 Qwen/Qwen3-Embedding-0.6B。
文档编码不带 instruction 前缀；查询编码由 retriever 侧添加前缀。
BM25 全文索引已迁移至 SQLite FTS5（由 DocStore 管理）。
"""

from __future__ import annotations

import contextlib
import fcntl
import logging
import os
from pathlib import Path

import numpy as np
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams

from src.embedding_backend import EmbeddingBackendConfig, load_embedding_model
from src.ingest.chunker import Chunk

logger = logging.getLogger(__name__)

COLLECTION_NAME = "docflow"


class Embedder:
    def __init__(
        self,
        qdrant_host: str = "localhost",
        qdrant_port: int = 6333,
        batch_size: int = 8,
        id_counter_path: str | Path = "qdrant_id_counter.txt",
        adaptive_batch_char_budget: int | None = None,
        adaptive_batch_max: int | None = None,
        embedding_config: EmbeddingBackendConfig | None = None,
    ):
        self.batch_size = batch_size
        self.adaptive_batch_char_budget = adaptive_batch_char_budget or (batch_size * 1024)
        self.adaptive_batch_max = adaptive_batch_max or max(batch_size, batch_size * 2)
        self._embedding_config = embedding_config or EmbeddingBackendConfig(
            model_name="Qwen/Qwen3-Embedding-0.6B"
        )

        self._model = None          # lazy-loaded SentenceTransformer
        self._vector_dim: int | None = None

        self._qdrant = QdrantClient(host=qdrant_host, port=qdrant_port)

        # Monotonic ID counter (safe after deletions)
        self._id_counter_path = Path(id_counter_path)
        self._qdrant_next_id = self._load_id_counter()

    # ------------------------------------------------------------------
    # Model (lazy load) + collection management
    # ------------------------------------------------------------------

    @property
    def model(self):
        if self._model is None:
            self._model = load_embedding_model(self._embedding_config)
            self._vector_dim = self._model.get_sentence_embedding_dimension()
            logger.info(f"[embedder] Embedding dim: {self._vector_dim}")
            self._ensure_collection(self._vector_dim)
        return self._model

    def _ensure_collection(self, vector_dim: int):
        if self._qdrant.collection_exists(COLLECTION_NAME):
            info = self._qdrant.get_collection(COLLECTION_NAME)
            existing_dim = info.config.params.vectors.size
            if existing_dim == vector_dim:
                return
            logger.warning(
                f"[embedder] Vector dim changed {existing_dim} → {vector_dim}. "
                "Recreating Qdrant collection — all files need re-ingestion."
            )
            self._qdrant.delete_collection(COLLECTION_NAME)
            self._reset_id_counter()

        self._qdrant.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=VectorParams(size=vector_dim, distance=Distance.COSINE),
        )

    # ------------------------------------------------------------------
    # Embed & store
    # ------------------------------------------------------------------

    def embed_chunks(self, chunks: list[Chunk]) -> list[int]:
        """
        批量 embed chunks，写入 Qdrant。
        返回 qdrant point ID 列表（与 chunks 一一对应）。
        """
        if not chunks:
            return []

        dense_vecs = self.encode_texts([c.embedding_text or c.raw_text or c.text for c in chunks])
        return self.upsert_embeddings(chunks, dense_vecs)

    @property
    def embedding_model_name(self) -> str:
        return self._embedding_config.model_name

    @property
    def embedding_cache_key(self) -> str:
        return self._embedding_config.cache_key()

    def _adaptive_batch_size(self, texts: list[str]) -> int:
        if not texts:
            return 1

        avg_chars = max(1, sum(len(t) for t in texts) // len(texts))
        adaptive = max(1, self.adaptive_batch_char_budget // avg_chars)
        adaptive = min(adaptive, self.adaptive_batch_max)
        return max(1, min(len(texts), adaptive))

    def encode_texts(self, texts: list[str], progress_callback=None) -> np.ndarray:
        if not texts:
            return np.empty((0, 0), dtype=np.float32)

        import torch

        batch_size = self._adaptive_batch_size(texts)
        vectors: list[np.ndarray] = []
        encoded = 0
        is_mps = self._embedding_config.device == "mps" and torch.backends.mps.is_available()
        use_empty_cache = is_mps and self._embedding_config.mps_empty_cache
        use_inference_mode = self._embedding_config.mps_inference_mode
        ctx = torch.inference_mode() if use_inference_mode else contextlib.nullcontext()

        with ctx:
            for i in range(0, len(texts), batch_size):
                batch = texts[i:i + batch_size]
                batch_vectors = self.model.encode(
                    batch,
                    batch_size=batch_size,
                    normalize_embeddings=True,
                    convert_to_numpy=True,
                    show_progress_bar=False,
                )
                batch_vectors = np.asarray(batch_vectors, dtype=np.float32)
                if batch_vectors.ndim == 1:
                    batch_vectors = batch_vectors.reshape(1, -1)
                vectors.append(batch_vectors)
                encoded += len(batch)

                # MPS 缓解②：每批次后释放 Metal buffer pool 中已 free 的缓存
                if use_empty_cache:
                    torch.mps.empty_cache()

                if progress_callback:
                    progress_callback(
                        {
                            "encoded_texts": encoded,
                            "total_texts": len(texts),
                            "batch_size": batch_size,
                        }
                    )

        return np.concatenate(vectors, axis=0)

    def upsert_embeddings(
        self,
        chunks: list[Chunk],
        dense_vecs: np.ndarray,
        min_next_id: int | None = None,
    ) -> list[int]:
        if len(chunks) != len(dense_vecs):
            raise ValueError("chunks and dense_vecs length mismatch")
        if not chunks:
            return []

        dense_vecs = np.asarray(dense_vecs, dtype=np.float32)
        if dense_vecs.ndim != 2:
            raise ValueError("dense_vecs must be a 2D array")
        if self._vector_dim is None:
            self._vector_dim = dense_vecs.shape[1]
            self._ensure_collection(self._vector_dim)

        ids = self._reserve_ids(len(chunks), min_next_id=min_next_id or 0)

        points = [
            PointStruct(
                id=ids[j],
                vector=dense_vecs[j].tolist(),
                payload={
                    "file_name": chunks[j].file_name,
                    "file_path": chunks[j].file_path,
                    "page_num": chunks[j].page_num,
                    "section": chunks[j].section,
                    "chunk_type": chunks[j].chunk_type,
                    "text": chunks[j].raw_text,
                    "raw_text": chunks[j].raw_text,
                    "embedding_text": chunks[j].embedding_text,
                    "child_text": chunks[j].raw_text,
                    "parent_id": chunks[j].parent_id,
                    "parent_text": chunks[j].parent_text,
                    "contextual_prefix": chunks[j].contextual_prefix,
                    "char_count": chunks[j].char_count,
                },
            )
            for j in range(len(chunks))
        ]

        self._qdrant.upsert(collection_name=COLLECTION_NAME, points=points)

        return ids

    # ------------------------------------------------------------------
    # Monotonic ID counter
    # ------------------------------------------------------------------

    def _load_id_counter(self) -> int:
        if self._id_counter_path.exists():
            try:
                return int(self._id_counter_path.read_text(encoding="utf-8").strip())
            except (ValueError, IOError):
                pass
        try:
            return self.max_point_id() + 1
        except Exception:
            return 0

    def _next_id(self) -> int:
        return self._qdrant_next_id

    def _reserve_ids(self, count: int, min_next_id: int = 0) -> list[int]:
        """Reserve a monotonic ID range under an interprocess file lock."""
        if count <= 0:
            return []
        self._id_counter_path.parent.mkdir(parents=True, exist_ok=True)
        with self._id_counter_path.open("a+", encoding="utf-8") as f:
            fcntl.flock(f.fileno(), fcntl.LOCK_EX)
            try:
                current = self._read_counter_handle(f)
                start = max(current, self._qdrant_next_id, int(min_next_id or 0))
                next_id = start + count
                self._write_counter_handle(f, next_id)
                self._qdrant_next_id = next_id
            finally:
                fcntl.flock(f.fileno(), fcntl.LOCK_UN)
        return list(range(start, next_id))

    def sync_id_counter(self, min_next_id: int = 0) -> dict:
        """Advance the local ID counter to at least min_next_id under lock."""
        self._id_counter_path.parent.mkdir(parents=True, exist_ok=True)
        with self._id_counter_path.open("a+", encoding="utf-8") as f:
            fcntl.flock(f.fileno(), fcntl.LOCK_EX)
            try:
                current = self._read_counter_handle(f)
                target = max(current, self._qdrant_next_id, int(min_next_id or 0))
                if target != current:
                    self._write_counter_handle(f, target)
                self._qdrant_next_id = target
                return {
                    "path": str(self._id_counter_path),
                    "previous": current,
                    "value": target,
                    "min_next_id": int(min_next_id or 0),
                    "advanced": target != current,
                }
            finally:
                fcntl.flock(f.fileno(), fcntl.LOCK_UN)

    def _reset_id_counter(self):
        self._qdrant_next_id = 0
        self._write_counter_locked(0)

    def max_point_id(self) -> int:
        """Return the highest point ID in Qdrant, or -1 when the collection is empty."""
        try:
            if not self._qdrant.collection_exists(COLLECTION_NAME):
                return -1
            max_id = -1
            offset = None
            while True:
                records, offset = self._qdrant.scroll(
                    collection_name=COLLECTION_NAME,
                    limit=256,
                    offset=offset,
                    with_payload=False,
                    with_vectors=False,
                )
                for record in records:
                    max_id = max(max_id, int(record.id))
                if offset is None:
                    break
            return max_id
        except Exception:
            logger.warning("[embedder] failed to inspect Qdrant point IDs", exc_info=True)
            raise

    def _write_counter_locked(self, value: int) -> None:
        self._id_counter_path.parent.mkdir(parents=True, exist_ok=True)
        with self._id_counter_path.open("a+", encoding="utf-8") as f:
            fcntl.flock(f.fileno(), fcntl.LOCK_EX)
            try:
                self._write_counter_handle(f, value)
            finally:
                fcntl.flock(f.fileno(), fcntl.LOCK_UN)

    @staticmethod
    def _read_counter_handle(handle) -> int:
        handle.seek(0)
        raw = handle.read().strip()
        try:
            return int(raw or "0")
        except ValueError:
            return 0

    @staticmethod
    def _write_counter_handle(handle, value: int) -> None:
        handle.seek(0)
        handle.truncate()
        handle.write(str(value))
        handle.flush()
        os.fsync(handle.fileno())

    def delete_file_vectors(self, qdrant_ids: list[int]):
        """删除某个文件的所有 Qdrant 向量（重新索引时调用）。FTS5 清理由 store.add_chunks() 负责。"""
        if not qdrant_ids:
            return
        from qdrant_client.models import PointIdsList
        self._qdrant.delete(
            collection_name=COLLECTION_NAME,
            points_selector=PointIdsList(points=qdrant_ids),
        )
