"""Hybrid retrieval facade and orchestration."""

from __future__ import annotations

import logging
import time as _time
from pathlib import Path

from src.embedding_backend import EmbeddingBackendConfig, load_embedding_model
from src.ingest.store import DocStore
from src.query import debug as debug_module
from src.query import fusion, keyword_search, vector_search
from src.query.constants import COLLECTION_NAME, QUERY_INSTRUCTION
from src.query.reranker import MLXReranker
from src.query.router import QueryRouter
from src.vector_store import QdrantVectorStore

logger = logging.getLogger(__name__)


class HybridRetriever:
    _logger = logger

    def __init__(
        self,
        qdrant_host: str = "localhost",
        qdrant_port: int = 6333,
        reranker_model: str = "Qwen/Qwen3-Reranker-0.6B",
        reranker_instruction: str = "",
        db_path: str | Path = "docflow.db",
        top_k_retrieval: int = 20,
        top_k_rerank: int = 5,
        store: DocStore | None = None,
        embedding_config: EmbeddingBackendConfig | None = None,
        allow_model_download: bool = False,
        collection_name: str = COLLECTION_NAME,
    ):
        self.top_k_retrieval = top_k_retrieval
        self.top_k_rerank = top_k_rerank
        self.collection_name = collection_name
        self._vector_store = QdrantVectorStore(host=qdrant_host, port=qdrant_port)
        self._qdrant = self._vector_store.client
        self._store = store or DocStore(db_path)
        self._embed_model = None
        self._reranker: MLXReranker | None = None
        self._embedding_config = embedding_config or EmbeddingBackendConfig(
            model_name="Qwen/Qwen3-Embedding-0.6B"
        )
        self._reranker_model_name = reranker_model
        self._reranker_instruction = reranker_instruction
        self._allow_model_download = allow_model_download

    @property
    def embed_model(self):
        if self._embed_model is None:
            self._embed_model = load_embedding_model(self._embedding_config)
        return self._embed_model

    @property
    def reranker(self) -> MLXReranker:
        if self._reranker is None:
            self._reranker = MLXReranker(
                model_name=self._reranker_model_name,
                instruction=self._reranker_instruction,
                allow_model_download=self._allow_model_download,
            )
        return self._reranker

    def retrieve(
        self,
        query: str,
        file_filter: list[str] | None = None,
        retrieval_mode: str = "hybrid",
        prefer_tables: bool = False,
        cancel_event=None,
        related_k: int = 0,
    ) -> list[dict]:
        t0 = _time.time()
        route = QueryRouter.classify(query)
        search_limit = route["top_k_retrieval"]
        output_limit = route["top_k_rerank"] + max(0, int(related_k or 0))
        degradations: list[dict] = []
        mode = self._normalize_retrieval_mode(retrieval_mode)

        vec_results: list[dict] = []
        if mode != "full_text":
            try:
                query_vec_list = self._encode_query(query)
                logger.info(f"[perf] embed: {_time.time() - t0:.2f}s")
                if self._is_cancelled(cancel_event):
                    return []
                t1 = _time.time()
                vec_results = self._vector_search(query_vec_list, file_filter, limit=search_limit)
                logger.info(
                    f"[perf] vector_search: {_time.time() - t1:.2f}s ({len(vec_results)} results)"
                )
            except Exception as exc:
                logger.warning(
                    "[retriever] vector path unavailable; falling back to FTS",
                    exc_info=True,
                )
                degradations.append(self._degradation("vector", exc))

        if self._is_cancelled(cancel_event):
            return []

        t1 = _time.time()
        try:
            fts_results = self._fts_search(
                query,
                file_filter,
                limit=search_limit,
                degradations=degradations,
            )
        except Exception as exc:
            logger.warning("[retriever] FTS path unavailable", exc_info=True)
            degradations.append(self._degradation("fts", exc))
            fts_results = []
        logger.info(f"[perf] fts_search: {_time.time() - t1:.2f}s ({len(fts_results)} results)")
        if self._is_cancelled(cancel_event):
            return []

        fused = self._rrf_fuse(
            vec_results,
            fts_results,
            prefer_tables=prefer_tables,
            vec_weight=route["vec_weight"],
            bm25_weight=route["bm25_weight"],
        )
        top_candidates = self._deduplicate(fused[:search_limit])
        if not top_candidates or self._is_cancelled(cancel_event):
            return []

        t1 = _time.time()
        result = self._rerank_or_fallback(
            query,
            top_candidates,
            output_limit,
            degradations,
            cancel_event=cancel_event,
        )
        result = self._expand_parent_context(result)
        self._attach_degradations(result, degradations)
        logger.info(f"[perf] rerank: {_time.time() - t1:.2f}s ({len(top_candidates)} pairs)")
        logger.info(f"[perf] total_retrieve: {_time.time() - t0:.2f}s ({len(result)} results)")
        return result

    def debug_retrieve(
        self,
        query: str,
        file_filter: list[str] | None = None,
        retrieval_mode: str = "hybrid",
        prefer_tables: bool = False,
        include_rerank: bool = True,
        max_text_chars: int = 300,
    ) -> dict:
        return debug_module.debug_retrieve(
            self,
            query,
            file_filter=file_filter,
            retrieval_mode=retrieval_mode,
            prefer_tables=prefer_tables,
            include_rerank=include_rerank,
            max_text_chars=max_text_chars,
        )

    def close(self) -> None:
        self._vector_store.close()

    @staticmethod
    def _tokenize(text: str) -> list[str]:
        return keyword_search.tokenize(text)

    def _vector_search(
        self, query_vec: list[float], file_filter: list[str] | None, limit: int | None = None
    ) -> list[dict]:
        return vector_search.vector_search(self, query_vec, file_filter, limit=limit)

    def _fts_search(
        self,
        query: str,
        file_filter: list[str] | None,
        limit: int | None = None,
        degradations: list[dict] | None = None,
    ) -> list[dict]:
        return keyword_search.fts_search(
            self, query, file_filter, limit=limit, degradations=degradations
        )

    @staticmethod
    def _fts_row_item(row: dict) -> dict:
        return keyword_search.fts_row_item(row)

    def fetch_file_chunks(self, qdrant_ids: list[int], max_chunks: int = 15) -> list[dict]:
        return vector_search.fetch_file_chunks(self, qdrant_ids, max_chunks=max_chunks)

    def fetch_chunks_by_ids(
        self, qdrant_ids: list[int], max_text_chars: int = 500
    ) -> dict[int, dict]:
        return vector_search.fetch_chunks_by_ids(
            self, qdrant_ids, max_text_chars=max_text_chars
        )

    _rrf_fuse = staticmethod(fusion.rrf_fuse)
    _deduplicate = staticmethod(fusion.deduplicate)

    def _expand_parent_context(self, candidates: list[dict]) -> list[dict]:
        return fusion.expand_parent_context(self, candidates)

    @staticmethod
    def _degradation(stage: str, exc: Exception) -> dict:
        return {
            "stage": stage,
            "status": "degraded",
            "error_type": exc.__class__.__name__,
            "message": str(exc),
        }

    @staticmethod
    def _attach_degradations(results: list[dict], degradations: list[dict]) -> None:
        if not degradations:
            return
        for item in results:
            item["retrieval_status"] = "degraded"
            item["degradations"] = degradations

    def _rerank_or_fallback(
        self,
        query: str,
        candidates: list[dict],
        top_k: int,
        degradations: list[dict],
        cancel_event=None,
    ) -> list[dict]:
        if self._is_cancelled(cancel_event):
            return []
        try:
            reranked = self._rerank(query, candidates, cancel_event=cancel_event, top_k=top_k)
            if reranked:
                return reranked
            if self._is_cancelled(cancel_event):
                return []
            degradations.append(
                {
                    "stage": "reranker",
                    "status": "degraded",
                    "error_type": "EmptyRerankResult",
                    "message": "Reranker returned no results; using fused candidates.",
                }
            )
        except Exception as exc:
            logger.warning("[retriever] reranker failed; using fused candidates", exc_info=True)
            degradations.append(self._degradation("reranker", exc))

        fallback = [dict(item) for item in candidates[:top_k]]
        for item in fallback:
            item["rerank_fallback"] = True
        return fallback

    @staticmethod
    def _normalize_retrieval_mode(mode: str | None) -> str:
        normalized = str(mode or "hybrid").strip().lower().replace("-", "_")
        if normalized in {"fts", "fulltext", "full_text"}:
            return "full_text"
        return "hybrid"

    @staticmethod
    def _debug_item(item: dict, max_text_chars: int = 300) -> dict:
        return debug_module.debug_item(item, max_text_chars=max_text_chars)

    def _encode_query(self, query: str) -> list[float]:
        instructed_query = f"Instruct: {QUERY_INSTRUCTION}\nQuery: {query}"
        query_vec = self.embed_model.encode(
            [instructed_query],
            normalize_embeddings=True,
            convert_to_numpy=True,
        )[0]
        return query_vec.tolist() if hasattr(query_vec, "tolist") else list(query_vec)

    @staticmethod
    def _is_cancelled(cancel_event) -> bool:
        return cancel_event is not None and cancel_event.is_set()

    def _rerank(
        self,
        query: str,
        candidates: list[dict],
        cancel_event=None,
        top_k: int | None = None,
    ) -> list[dict]:
        if self._is_cancelled(cancel_event):
            return []
        pairs = [[query, c["text"]] for c in candidates]
        rerank_scores = self.reranker.compute_score(
            pairs,
            normalize=True,
            cancel_event=cancel_event,
        )
        if not rerank_scores:
            return []

        for i, item in enumerate(candidates):
            item["rerank_score"] = float(rerank_scores[i])

        candidates.sort(key=lambda x: x["rerank_score"], reverse=True)
        top_score = candidates[0]["rerank_score"] if candidates else 0
        cutoff = top_score * 0.10
        filtered = [c for c in candidates if c["rerank_score"] >= cutoff]
        return filtered[: (top_k or self.top_k_rerank)]
