"""
Embedder — Qwen3-Embedding-0.6B 批量 Embedding，写入 Qdrant。

使用 sentence-transformers 加载 Qwen/Qwen3-Embedding-0.6B。
文档编码不带 instruction 前缀；查询编码由 retriever 侧添加前缀。
BM25 全文索引已迁移至 SQLite FTS5（由 DocStore 管理）。
"""

from __future__ import annotations

import contextlib
import logging
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

        dense_vecs = self.encode_texts([c.text for c in chunks])
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

    def upsert_embeddings(self, chunks: list[Chunk], dense_vecs: np.ndarray) -> list[int]:
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

        start_id = self._next_id()
        ids = list(range(start_id, start_id + len(chunks)))

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
                    "text": chunks[j].text,
                    "char_count": chunks[j].char_count,
                },
            )
            for j in range(len(chunks))
        ]

        # Single upsert + single ID counter write
        self._qdrant.upsert(collection_name=COLLECTION_NAME, points=points)
        self._advance_id(len(chunks))

        return ids

    # ------------------------------------------------------------------
    # Monotonic ID counter
    # ------------------------------------------------------------------

    def _load_id_counter(self) -> int:
        if self._id_counter_path.exists():
            try:
                return int(self._id_counter_path.read_text().strip())
            except (ValueError, IOError):
                pass
        try:
            info = self._qdrant.get_collection(COLLECTION_NAME)
            return info.points_count
        except Exception:
            return 0

    def _next_id(self) -> int:
        return self._qdrant_next_id

    def _advance_id(self, count: int):
        self._qdrant_next_id += count
        self._id_counter_path.write_text(str(self._qdrant_next_id))

    def _reset_id_counter(self):
        self._qdrant_next_id = 0
        self._id_counter_path.write_text("0")

    def delete_file_vectors(self, qdrant_ids: list[int]):
        """删除某个文件的所有 Qdrant 向量（重新索引时调用）。FTS5 清理由 store.add_chunks() 负责。"""
        if not qdrant_ids:
            return
        from qdrant_client.models import PointIdsList
        self._qdrant.delete(
            collection_name=COLLECTION_NAME,
            points_selector=PointIdsList(points=qdrant_ids),
        )
