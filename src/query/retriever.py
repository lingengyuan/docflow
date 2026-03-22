"""
HybridRetriever — 向量检索 + FTS5 关键词检索 + RRF 融合 + 精排。

pipeline:
  Qwen3-Embedding-0.6B dense 向量检索 top-20
  SQLite FTS5 BM25 关键词检索 top-20（替代原 pickle BM25）
  RRF 融合 + 向量分数过滤 → candidates
  Qwen3-Reranker-0.6B 精排 → top-5  (MLX runtime, Apple Silicon)
"""

from __future__ import annotations

import logging
import os
import re
from pathlib import Path

from qdrant_client import QdrantClient

from src.ingest.store import DocStore

logger = logging.getLogger(__name__)

COLLECTION_NAME = "docflow"

QUERY_INSTRUCTION = "Retrieve relevant text passages that answer the query."


# ---------------------------------------------------------------------------
# QueryRouter — 规则路由，0ms 开销，动态调整向量/关键词检索权重
# ---------------------------------------------------------------------------

class QueryRouter:
    """
    根据查询特征动态调整 BM25（FTS5）与向量检索的权重。

    关键词信号（引号短语、日期、扩展名）→ 偏向 FTS5
    长语义查询（无关键词信号）→ 偏向向量
    默认 → 均衡
    """

    _KEYWORD_PATTERNS = [
        re.compile(r'"[^"]+"'),              # "精确短语"
        re.compile(r'\b\d{4}[-/]\d{1,2}'),  # 日期 2024-01
        re.compile(r'\.\w{2,4}\b'),          # 文件扩展名 .pdf
        re.compile(r'[A-Z]{2,}\d+'),         # 编号 INV2024
    ]

    @classmethod
    def classify(cls, query: str) -> dict:
        signals = sum(1 for p in cls._KEYWORD_PATTERNS if p.search(query))
        if signals >= 2:
            weights = {"bm25_weight": 2.0, "vec_weight": 0.5}
        elif len(query) > 20 and signals == 0:
            weights = {"bm25_weight": 0.5, "vec_weight": 2.0}
        else:
            weights = {"bm25_weight": 1.0, "vec_weight": 1.0}
        logger.debug(f"[router] query={query[:40]!r} signals={signals} weights={weights}")
        return weights


# ---------------------------------------------------------------------------
# MLXReranker
# ---------------------------------------------------------------------------

class MLXReranker:
    """
    Qwen3-Reranker-0.6B 生成式重排序模型，使用 mlx-lm 在 Apple Silicon 上推理。
    比 PyTorch MPS 快约 200x：10 pairs ~0.5s（原来 10.45s）。
    """

    SYSTEM_PROMPT = (
        "Judge whether the Document meets the requirements based on the Query and the Instruct, "
        "output your judgement in 'yes' or 'no'."
    )

    def __init__(
        self,
        model_name: str = "Qwen/Qwen3-Reranker-0.6B",
        instruction: str = "",
        max_length: int = 4096,
    ):
        from mlx_lm import load

        self.instruction = instruction or QUERY_INSTRUCTION
        self.max_length = max_length

        logger.info(f"[reranker] Loading MLX reranker: {model_name}")
        self._model, self._tokenizer = load(model_name)

        self._yes_id = self._tokenizer.encode("yes", add_special_tokens=False)[0]
        self._no_id = self._tokenizer.encode("no", add_special_tokens=False)[0]
        logger.info(f"[reranker] MLX reranker ready (yes_id={self._yes_id}, no_id={self._no_id})")

    def _build_prompt(self, query: str, passage: str) -> str:
        user_msg = (
            f"<Instruct>: {self.instruction}\n"
            f"<Query>: {query}\n"
            f"<Document>: {passage}"
        )
        messages = [
            {"role": "system", "content": self.SYSTEM_PROMPT},
            {"role": "user", "content": user_msg},
        ]
        text = self._tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        return text + "<think>\n\n</think>\n\n"

    def compute_score(self, pairs: list[list[str]], normalize: bool = True) -> list[float]:
        """pairs: [[query, passage], ...] → relevance scores in [0, 1]。"""
        import mlx.core as mx

        scores = []
        for q, p in pairs:
            prompt = self._build_prompt(q, p)
            tokens = self._tokenizer.encode(prompt)
            if len(tokens) > self.max_length:
                tokens = tokens[: self.max_length]

            inputs = mx.array([tokens])
            logits = self._model(inputs)
            mx.eval(logits)

            last = logits[0, -1, [self._yes_id, self._no_id]]
            score = float(mx.softmax(last, axis=0)[0])
            scores.append(score)

        return scores


# ---------------------------------------------------------------------------
# HybridRetriever
# ---------------------------------------------------------------------------

class HybridRetriever:
    def __init__(
        self,
        qdrant_host: str = "localhost",
        qdrant_port: int = 6333,
        embedding_model: str = "Qwen/Qwen3-Embedding-0.6B",
        reranker_model: str = "Qwen/Qwen3-Reranker-0.6B",
        reranker_instruction: str = "",
        db_path: str | Path = "docflow.db",
        top_k_retrieval: int = 20,
        top_k_rerank: int = 5,
        device: str = "cpu",
    ):
        self.top_k_retrieval = top_k_retrieval
        self.top_k_rerank = top_k_rerank

        self._qdrant = QdrantClient(host=qdrant_host, port=qdrant_port)
        self._store = DocStore(db_path)
        self._embed_model = None
        self._reranker: MLXReranker | None = None
        self._embedding_model_name = embedding_model
        self._reranker_model_name = reranker_model
        self._reranker_instruction = reranker_instruction
        self._device = device

    # ------------------------------------------------------------------
    # Lazy-load models
    # ------------------------------------------------------------------

    @property
    def embed_model(self):
        if self._embed_model is None:
            import torch
            from sentence_transformers import SentenceTransformer
            n_threads = os.cpu_count() or 4
            torch.set_num_threads(n_threads)
            logger.info(f"[retriever] CPU threads: {n_threads}")
            logger.info(f"[retriever] Loading embedding model: {self._embedding_model_name}")
            self._embed_model = SentenceTransformer(
                self._embedding_model_name,
                device=self._device,
                trust_remote_code=True,
            )
        return self._embed_model

    @property
    def reranker(self) -> MLXReranker:
        if self._reranker is None:
            self._reranker = MLXReranker(
                model_name=self._reranker_model_name,
                instruction=self._reranker_instruction,
            )
        return self._reranker

    # ------------------------------------------------------------------
    # FTS5 tokenization (jieba)
    # ------------------------------------------------------------------

    @staticmethod
    def _tokenize(text: str) -> list[str]:
        import jieba
        return [t for t in jieba.cut(text.lower()) if t.strip()]

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def retrieve(
        self,
        query: str,
        file_filter: list[str] | None = None,
        prefer_tables: bool = False,
    ) -> list[dict]:
        """
        混合检索 + 精排，返回 top-k 结果。
        每个结果：{qdrant_id, score, text, file_name, file_path, page_num, section, chunk_type}
        """
        import time as _time
        _t0 = _time.time()

        # 1. Encode query
        instructed_query = f"Instruct: {QUERY_INSTRUCTION}\nQuery: {query}"
        query_vec = self.embed_model.encode(
            [instructed_query],
            normalize_embeddings=True,
            convert_to_numpy=True,
        )[0]
        logger.info(f"[perf] embed: {_time.time()-_t0:.2f}s")

        # 2. Vector search
        _t1 = _time.time()
        vec_results = self._vector_search(query_vec.tolist(), file_filter)
        logger.info(f"[perf] vector_search: {_time.time()-_t1:.2f}s ({len(vec_results)} results)")

        # 3. FTS5 keyword search
        _t1 = _time.time()
        fts_results = self._fts_search(query, file_filter)
        logger.info(f"[perf] fts_search: {_time.time()-_t1:.2f}s ({len(fts_results)} results)")

        # 4. QueryRouter + RRF fusion
        weights = QueryRouter.classify(query)
        fused = self._rrf_fuse(vec_results, fts_results, prefer_tables=prefer_tables, **weights)
        top_candidates = fused[: self.top_k_retrieval]

        if not top_candidates:
            return []

        # 5. Deduplicate
        top_candidates = self._deduplicate(top_candidates)

        # 6. Rerank
        _t1 = _time.time()
        result = self._rerank(query, top_candidates)
        logger.info(f"[perf] rerank: {_time.time()-_t1:.2f}s ({len(top_candidates)} pairs)")
        logger.info(f"[perf] total_retrieve: {_time.time()-_t0:.2f}s ({len(result)} results)")
        return result

    # ------------------------------------------------------------------
    # Vector search
    # ------------------------------------------------------------------

    def _vector_search(self, query_vec: list[float], file_filter: list[str] | None) -> list[dict]:
        from qdrant_client.models import Filter, FieldCondition, MatchAny

        search_filter = None
        if file_filter:
            search_filter = Filter(
                must=[FieldCondition(key="file_name", match=MatchAny(any=file_filter))]
            )

        results = self._qdrant.query_points(
            collection_name=COLLECTION_NAME,
            query=query_vec,
            query_filter=search_filter,
            limit=self.top_k_retrieval,
        )
        return [
            {
                "qdrant_id": r.id,
                "score": r.score,
                **r.payload,
            }
            for r in results.points
        ]

    # ------------------------------------------------------------------
    # FTS5 keyword search (replaces pickle BM25)
    # ------------------------------------------------------------------

    def _fts_search(self, query: str, file_filter: list[str] | None) -> list[dict]:
        tokens = self._tokenize(query)
        rows: list[dict] = []

        if tokens:
            # Layer 1: jieba 精确匹配
            escaped = [t.replace('"', "") for t in tokens]
            fts_query = " OR ".join(f'"{t}"' for t in escaped if t)
            if fts_query:
                try:
                    rows = self._store.search_fts(fts_query, file_filter, limit=self.top_k_retrieval)
                except Exception:
                    logger.warning("[retriever] FTS5 exact search failed", exc_info=True)

        if not rows:
            # Layer 2: trigram 子串匹配（OCR 错字、简繁混用容错）
            logger.debug("[retriever] FTS5 exact empty → trigram fallback")
            try:
                rows = self._store.search_fts_trigram(query, file_filter, limit=self.top_k_retrieval)
            except Exception:
                logger.warning("[retriever] FTS5 trigram search failed", exc_info=True)

        if not rows:
            return []

        top_ids = [r["qdrant_id"] for r in rows]
        score_map = {r["qdrant_id"]: r["score"] for r in rows}

        # Fetch text from Qdrant (text lives in Qdrant payload for now)
        fetched = self._qdrant.retrieve(
            collection_name=COLLECTION_NAME,
            ids=top_ids,
            with_payload=True,
        )
        id_to_payload = {p.id: p.payload for p in fetched}

        results = []
        for qid in top_ids:
            payload = id_to_payload.get(qid)
            if payload is None:
                continue
            results.append({"qdrant_id": qid, "score": score_map[qid], **payload})
        return results

    def fetch_file_chunks(self, qdrant_ids: list[int], max_chunks: int = 15) -> list[dict]:
        """按 page_num 顺序获取文件的前 N 个 chunk，用于摘要生成。"""
        if not qdrant_ids:
            return []
        sample_ids = qdrant_ids[:max_chunks * 3]
        records = self._qdrant.retrieve(
            collection_name=COLLECTION_NAME,
            ids=sample_ids,
            with_payload=True,
        )
        chunks = [{"qdrant_id": r.id, **r.payload} for r in records]
        chunks.sort(key=lambda c: (c.get("page_num", 0), c.get("qdrant_id", 0)))
        return chunks[:max_chunks]

    # ------------------------------------------------------------------
    # RRF fusion
    # ------------------------------------------------------------------

    @staticmethod
    def _rrf_fuse(
        vec_results: list[dict],
        fts_results: list[dict],
        k: int = 60,
        prefer_tables: bool = False,
        vec_score_threshold: float = 0.4,
        vec_weight: float = 1.0,
        bm25_weight: float = 1.0,
    ) -> list[dict]:
        scores: dict[int, float] = {}
        id_to_item: dict[int, dict] = {}
        vec_scores: dict[int, float] = {}

        for rank, item in enumerate(vec_results):
            qid = item["qdrant_id"]
            scores[qid] = scores.get(qid, 0.0) + vec_weight / (k + rank + 1)
            id_to_item[qid] = item
            vec_scores[qid] = item["score"]

        for rank, item in enumerate(fts_results):
            qid = item["qdrant_id"]
            scores[qid] = scores.get(qid, 0.0) + bm25_weight / (k + rank + 1)
            if qid not in id_to_item:
                id_to_item[qid] = item

        if prefer_tables:
            for qid, item in id_to_item.items():
                if item.get("chunk_type") in ("table", "table_summary"):
                    scores[qid] *= 1.5

        sorted_ids = sorted(scores, key=lambda x: scores[x], reverse=True)
        results = []
        for qid in sorted_ids:
            vs = vec_scores.get(qid, 0.0)
            if vs < vec_score_threshold:
                continue
            results.append({**id_to_item[qid], "rrf_score": scores[qid], "vec_score": vs})
        return results

    # ------------------------------------------------------------------
    # Deduplication
    # ------------------------------------------------------------------

    @staticmethod
    def _deduplicate(candidates: list[dict]) -> list[dict]:
        seen: dict[tuple, dict] = {}
        for item in candidates:
            key = (item.get("file_path", ""), item.get("page_num", 0), item.get("text", "")[:128])
            if key not in seen or item.get("rrf_score", 0) > seen[key].get("rrf_score", 0):
                seen[key] = item
        return list(seen.values())

    # ------------------------------------------------------------------
    # Rerank
    # ------------------------------------------------------------------

    def _rerank(self, query: str, candidates: list[dict]) -> list[dict]:
        pairs = [[query, c["text"]] for c in candidates]
        rerank_scores = self.reranker.compute_score(pairs, normalize=True)

        for i, item in enumerate(candidates):
            item["rerank_score"] = float(rerank_scores[i])

        candidates.sort(key=lambda x: x["rerank_score"], reverse=True)

        top_score = candidates[0]["rerank_score"] if candidates else 0
        cutoff = top_score * 0.10
        filtered = [c for c in candidates if c["rerank_score"] >= cutoff]
        return filtered[: self.top_k_rerank]
