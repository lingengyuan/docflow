"""
Embedder — Qwen3-Embedding-0.6B 批量 Embedding，写入 Qdrant。

使用 sentence-transformers 加载 Qwen/Qwen3-Embedding-0.6B。
文档编码不带 instruction 前缀；查询编码由 retriever 侧添加前缀。
BM25 全文索引已迁移至 SQLite FTS5（由 DocStore 管理）。
"""

from __future__ import annotations

import logging
from pathlib import Path

from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams

from src.ingest.chunker import Chunk

logger = logging.getLogger(__name__)

COLLECTION_NAME = "docflow"


class Embedder:
    def __init__(
        self,
        qdrant_host: str = "localhost",
        qdrant_port: int = 6333,
        embedding_model: str = "Qwen/Qwen3-Embedding-0.6B",
        batch_size: int = 8,
        device: str = "cpu",
        id_counter_path: str | Path = "qdrant_id_counter.txt",
    ):
        self.batch_size = batch_size
        self._embedding_model_name = embedding_model
        self._device = device

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
            from sentence_transformers import SentenceTransformer
            logger.info(f"[embedder] Loading embedding model: {self._embedding_model_name}")
            self._model = SentenceTransformer(
                self._embedding_model_name,
                device=self._device,
                trust_remote_code=True,
            )
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

        texts = [c.text for c in chunks]
        all_ids: list[int] = []

        for i in range(0, len(texts), self.batch_size):
            batch_texts = texts[i : i + self.batch_size]
            batch_chunks = chunks[i : i + self.batch_size]

            dense_vecs = self.model.encode(
                batch_texts,
                batch_size=self.batch_size,
                normalize_embeddings=True,
                convert_to_numpy=True,
            )
            start_id = self._next_id()
            ids = list(range(start_id, start_id + len(batch_chunks)))

            points = [
                PointStruct(
                    id=ids[j],
                    vector=dense_vecs[j].tolist(),
                    payload={
                        "file_name": batch_chunks[j].file_name,
                        "file_path": batch_chunks[j].file_path,
                        "page_num": batch_chunks[j].page_num,
                        "section": batch_chunks[j].section,
                        "chunk_type": batch_chunks[j].chunk_type,
                        "text": batch_chunks[j].text,
                        "char_count": batch_chunks[j].char_count,
                    },
                )
                for j in range(len(batch_chunks))
            ]
            self._qdrant.upsert(collection_name=COLLECTION_NAME, points=points)
            self._advance_id(len(batch_chunks))

            all_ids.extend(ids)

        return all_ids

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
