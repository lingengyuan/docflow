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
import re
from pathlib import Path

from src.domain_types import RetrievalResult
from src.embedding_backend import EmbeddingBackendConfig, load_embedding_model
from src.ingest.store import DocStore
from src.vector_store import QdrantVectorStore

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
        re.compile(
            r'[\u4e00-\u9fff].*\b(?:[A-Z]{2,}|[A-Za-z]+[-_][A-Za-z0-9_-]+|'
            r'[A-Za-z]*\d[A-Za-z0-9_-]*|fallback|reranked|backup|export-chunks|restore-plan)\b'
            r'|\b(?:[A-Z]{2,}|[A-Za-z]+[-_][A-Za-z0-9_-]+|'
            r'[A-Za-z]*\d[A-Za-z0-9_-]*|fallback|reranked|backup|export-chunks|restore-plan)\b'
            r'.*[\u4e00-\u9fff]'
        ),
    ]
    _CROSS_DOC_PATTERNS = [
        re.compile(r"对比|比较|差异|区别|汇总|总结|跨文档|多个文件"),
        re.compile(r"\b(compare|comparison|differences?|summari[sz]e|across files?)\b", re.I),
    ]

    @classmethod
    def classify(cls, query: str) -> dict:
        signals = sum(1 for p in cls._KEYWORD_PATTERNS if p.search(query))
        is_cross_doc = any(p.search(query) for p in cls._CROSS_DOC_PATTERNS)
        if signals >= 2:
            route = {
                "query_type": "exact",
                "bm25_weight": 2.0,
                "vec_weight": 0.5,
                "top_k_retrieval": 12,
                "top_k_rerank": 3,
            }
        elif is_cross_doc:
            route = {
                "query_type": "cross_document",
                "bm25_weight": 1.0,
                "vec_weight": 1.5,
                "top_k_retrieval": 30,
                "top_k_rerank": 10,
            }
        elif signals == 1:
            route = {
                "query_type": "keyword",
                "bm25_weight": 1.5,
                "vec_weight": 1.0,
                "top_k_retrieval": 24,
                "top_k_rerank": 6,
            }
        elif len(query) > 20 and signals == 0:
            route = {
                "query_type": "semantic",
                "bm25_weight": 0.5,
                "vec_weight": 2.0,
                "top_k_retrieval": 24,
                "top_k_rerank": 8,
            }
        else:
            route = {
                "query_type": "balanced",
                "bm25_weight": 1.0,
                "vec_weight": 1.0,
                "top_k_retrieval": 20,
                "top_k_rerank": 5,
            }
        logger.debug(f"[router] query={query[:40]!r} signals={signals} route={route}")
        return route


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

    def compute_score(
        self,
        pairs: list[list[str]],
        normalize: bool = True,
        cancel_event=None,
    ) -> list[float]:
        """pairs: [[query, passage], ...] → relevance scores in [0, 1]。"""
        import mlx.core as mx

        if cancel_event is not None and cancel_event.is_set():
            return []

        # Collect logits lazily — defer mx.eval() to batch-sync once
        all_last_logits = []
        for q, p in pairs:
            if cancel_event is not None and cancel_event.is_set():
                return []
            prompt = self._build_prompt(q, p)
            tokens = self._tokenizer.encode(prompt)
            if len(tokens) > self.max_length:
                tokens = tokens[: self.max_length]

            inputs = mx.array([tokens])
            logits = self._model(inputs)
            all_last_logits.append(logits[0, -1, [self._yes_id, self._no_id]])

        if not all_last_logits:
            return []

        # Single GPU sync instead of per-pair sync
        mx.eval(*all_last_logits)

        return [float(mx.softmax(last, axis=0)[0]) for last in all_last_logits]


# ---------------------------------------------------------------------------
# HybridRetriever
# ---------------------------------------------------------------------------

class HybridRetriever:
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
    ):
        self.top_k_retrieval = top_k_retrieval
        self.top_k_rerank = top_k_rerank

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

    # ------------------------------------------------------------------
    # Lazy-load models
    # ------------------------------------------------------------------

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
        retrieval_mode: str = "hybrid",
        prefer_tables: bool = False,
        cancel_event=None,
        related_k: int = 0,
    ) -> list[RetrievalResult]:
        """
        混合检索 + 精排，返回 top-k 结果。
        每个结果：{qdrant_id, score, text, file_name, file_path, page_num, section, chunk_type}
        """
        import time as _time
        _t0 = _time.time()

        route = QueryRouter.classify(query)
        search_limit = route["top_k_retrieval"]
        rerank_limit = route["top_k_rerank"]
        output_limit = rerank_limit + max(0, int(related_k or 0))
        degradations: list[dict] = []

        mode = self._normalize_retrieval_mode(retrieval_mode)

        # 1-2. Encode query + vector search. If this path is unavailable, keep FTS alive.
        vec_results: list[dict] = []
        if mode != "full_text":
            try:
                instructed_query = f"Instruct: {QUERY_INSTRUCTION}\nQuery: {query}"
                query_vec = self.embed_model.encode(
                    [instructed_query],
                    normalize_embeddings=True,
                    convert_to_numpy=True,
                )[0]
                query_vec_list = query_vec.tolist() if hasattr(query_vec, "tolist") else list(query_vec)
                logger.info(f"[perf] embed: {_time.time()-_t0:.2f}s")
                if cancel_event is not None and cancel_event.is_set():
                    return []

                _t1 = _time.time()
                vec_results = self._vector_search(query_vec_list, file_filter, limit=search_limit)
                logger.info(f"[perf] vector_search: {_time.time()-_t1:.2f}s ({len(vec_results)} results)")
            except Exception as exc:
                logger.warning("[retriever] vector path unavailable; falling back to FTS", exc_info=True)
                degradations.append(self._degradation("vector", exc))

        if cancel_event is not None and cancel_event.is_set():
            return []

        # 3. FTS5 keyword search
        fts_results: list[dict] = []
        _t1 = _time.time()
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
        logger.info(f"[perf] fts_search: {_time.time()-_t1:.2f}s ({len(fts_results)} results)")
        if cancel_event is not None and cancel_event.is_set():
            return []

        # 4. QueryRouter + RRF fusion
        fused = self._rrf_fuse(
            vec_results,
            fts_results,
            prefer_tables=prefer_tables,
            vec_weight=route["vec_weight"],
            bm25_weight=route["bm25_weight"],
        )
        top_candidates = fused[:search_limit]

        if not top_candidates:
            return []

        # 5. Deduplicate
        top_candidates = self._deduplicate(top_candidates)
        if cancel_event is not None and cancel_event.is_set():
            return []

        # 6. Rerank
        _t1 = _time.time()
        result = self._rerank_or_fallback(
            query,
            top_candidates,
            output_limit,
            degradations,
            cancel_event=cancel_event,
        )
        result = self._expand_parent_context(result)
        self._attach_degradations(result, degradations)
        logger.info(f"[perf] rerank: {_time.time()-_t1:.2f}s ({len(top_candidates)} pairs)")
        logger.info(f"[perf] total_retrieve: {_time.time()-_t0:.2f}s ({len(result)} results)")
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
        """返回完整检索链路，用于本地调试和评估，不调用回答模型。"""
        import time as _time

        timings: dict[str, float] = {}
        degradations: list[dict] = []
        t0 = _time.perf_counter()

        route = QueryRouter.classify(query)
        search_limit = route["top_k_retrieval"]
        rerank_limit = route["top_k_rerank"]
        mode = self._normalize_retrieval_mode(retrieval_mode)

        query_vec_list: list[float] | None = None
        if mode != "full_text":
            instructed_query = f"Instruct: {QUERY_INSTRUCTION}\nQuery: {query}"
            try:
                query_vec = self.embed_model.encode(
                    [instructed_query],
                    normalize_embeddings=True,
                    convert_to_numpy=True,
                )[0]
                query_vec_list = query_vec.tolist() if hasattr(query_vec, "tolist") else list(query_vec)
            except Exception as exc:
                logger.warning("[retriever] debug vector encode failed", exc_info=True)
                degradations.append(self._degradation("vector", exc))
        timings["embed_ms"] = round((_time.perf_counter() - t0) * 1000, 2)

        t1 = _time.perf_counter()
        vec_results: list[dict] = []
        if query_vec_list is not None:
            try:
                vec_results = self._vector_search(query_vec_list, file_filter, limit=search_limit)
            except Exception as exc:
                logger.warning("[retriever] debug vector search failed", exc_info=True)
                degradations.append(self._degradation("vector", exc))
        timings["vector_ms"] = round((_time.perf_counter() - t1) * 1000, 2)

        t1 = _time.perf_counter()
        fts_results: list[dict] = []
        try:
            fts_results = self._fts_search(
                query,
                file_filter,
                limit=search_limit,
                degradations=degradations,
            )
        except Exception as exc:
            logger.warning("[retriever] debug FTS search failed", exc_info=True)
            degradations.append(self._degradation("fts", exc))
        timings["fts_ms"] = round((_time.perf_counter() - t1) * 1000, 2)

        fused = self._rrf_fuse(
            vec_results,
            fts_results,
            prefer_tables=prefer_tables,
            vec_weight=route["vec_weight"],
            bm25_weight=route["bm25_weight"],
        )
        top_candidates = fused[:search_limit]
        deduped = self._deduplicate(top_candidates)

        reranked: list[dict] = []
        parent_expanded: list[dict] = []
        if include_rerank and deduped:
            t1 = _time.perf_counter()
            reranked = self._rerank_or_fallback(
                query,
                [dict(item) for item in deduped],
                rerank_limit,
                degradations,
            )
            parent_expanded = self._expand_parent_context([dict(item) for item in reranked])
            timings["rerank_ms"] = round((_time.perf_counter() - t1) * 1000, 2)
        else:
            timings["rerank_ms"] = 0.0
            parent_expanded = self._expand_parent_context([dict(item) for item in deduped])
        self._attach_degradations(parent_expanded, degradations)

        timings["total_ms"] = round((_time.perf_counter() - t0) * 1000, 2)
        return {
            "query": query,
            "file_filter": file_filter or [],
            "retrieval_mode": mode,
            "prefer_tables": prefer_tables,
            "router": route,
            "top_k_retrieval": search_limit,
            "top_k_rerank": rerank_limit,
            "status": "degraded" if degradations else "ok",
            "degradations": degradations,
            "timings": timings,
            "stages": {
                "vector": [self._debug_item(item, max_text_chars) for item in vec_results],
                "fts": [self._debug_item(item, max_text_chars) for item in fts_results],
                "fused": [self._debug_item(item, max_text_chars) for item in top_candidates],
                "deduped": [self._debug_item(item, max_text_chars) for item in deduped],
                "reranked": [self._debug_item(item, max_text_chars) for item in reranked],
                "parent_expanded": [
                    self._debug_item(item, max_text_chars) for item in parent_expanded
                ],
            },
        }

    # ------------------------------------------------------------------
    # Vector search
    # ------------------------------------------------------------------

    def _vector_search(
        self,
        query_vec: list[float],
        file_filter: list[str] | None,
        limit: int | None = None,
    ) -> list[RetrievalResult]:
        results = self._vector_store.search(
            collection_name=COLLECTION_NAME,
            query=query_vec,
            file_filter=file_filter,
            limit=limit or self.top_k_retrieval,
        )
        return [
            {
                "qdrant_id": hit.id,
                "score": hit.score,
                **hit.payload,
            }
            for hit in results
        ]

    def close(self) -> None:
        self._vector_store.close()

    # ------------------------------------------------------------------
    # FTS5 keyword search (replaces pickle BM25)
    # ------------------------------------------------------------------

    def _fts_search(
        self,
        query: str,
        file_filter: list[str] | None,
        limit: int | None = None,
        degradations: list[dict] | None = None,
    ) -> list[dict]:
        limit = limit or self.top_k_retrieval
        tokens = self._tokenize(query)
        rows: list[dict] = []

        if tokens:
            # Layer 1: jieba 精确匹配
            escaped = [t.replace('"', "") for t in tokens]
            fts_query = " OR ".join(f'"{t}"' for t in escaped if t)
            if fts_query:
                try:
                    rows = self._store.search_fts(fts_query, file_filter, limit=limit)
                except Exception:
                    logger.warning("[retriever] FTS5 exact search failed", exc_info=True)

        if not rows:
            # Layer 2: trigram 子串匹配（OCR 错字、简繁混用容错）
            logger.debug("[retriever] FTS5 exact empty → trigram fallback")
            try:
                rows = self._store.search_fts_trigram(query, file_filter, limit=limit)
            except Exception:
                logger.warning("[retriever] FTS5 trigram search failed", exc_info=True)

        if not rows:
            return []

        top_ids = [r["qdrant_id"] for r in rows]
        score_map = {r["qdrant_id"]: r["score"] for r in rows}
        row_payloads = {r["qdrant_id"]: self._fts_row_item(r) for r in rows}

        try:
            fetched = self._qdrant.retrieve(
                collection_name=COLLECTION_NAME,
                ids=top_ids,
                with_payload=True,
            )
            id_to_payload = {p.id: p.payload for p in fetched}
        except Exception as exc:
            logger.warning("[retriever] Qdrant payload fetch failed; using SQLite FTS payloads", exc_info=True)
            if degradations is not None:
                degradations.append(self._degradation("fts_payload", exc))
            id_to_payload = {}

        results = []
        for qid in top_ids:
            payload = {**row_payloads.get(qid, {}), **(id_to_payload.get(qid) or {})}
            if not payload:
                continue
            results.append({"qdrant_id": qid, "score": score_map[qid], **payload})
        return results

    @staticmethod
    def _fts_row_item(row: dict) -> dict:
        raw_text = row.get("raw_text", "") or ""
        return {
            "file_name": row.get("file_name", ""),
            "file_path": row.get("file_path", ""),
            "page_num": row.get("page_num", 0),
            "section": row.get("section", ""),
            "chunk_type": row.get("chunk_type", ""),
            "text": raw_text,
            "raw_text": raw_text,
            "child_text": raw_text,
            "parent_id": row.get("parent_id", 0),
            "parent_text": row.get("parent_text", ""),
            "contextual_prefix": row.get("contextual_prefix", ""),
            "char_count": row.get("char_count", len(raw_text)),
        }

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

    def fetch_chunks_by_ids(self, qdrant_ids: list[int], max_text_chars: int = 500) -> dict[int, dict]:
        """按 Qdrant id 批量获取 chunk payload，用于调试展示。"""
        if not qdrant_ids:
            return {}
        max_text_chars = max(0, max_text_chars)

        result: dict[int, dict] = {}
        for i in range(0, len(qdrant_ids), 100):
            batch = qdrant_ids[i:i + 100]
            records = self._qdrant.retrieve(
                collection_name=COLLECTION_NAME,
                ids=batch,
                with_payload=True,
            )
            for record in records:
                payload = dict(record.payload or {})
                text = payload.get("text", "")
                payload["text_preview"] = text[:max_text_chars]
                payload["text_length"] = len(text)
                payload.pop("text", None)
                result[int(record.id)] = payload
        return result

    # ------------------------------------------------------------------
    # RRF fusion
    # ------------------------------------------------------------------

    @staticmethod
    def _rrf_fuse(
        vec_results: list[dict],
        fts_results: list[dict],
        k: int = 60,
        prefer_tables: bool = False,
        vec_score_threshold: float = 0.3,
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
            # Only apply threshold to items that were in vector results;
            # BM25-only items (not in vec_scores) should pass through.
            if qid in vec_scores and vs < vec_score_threshold:
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
            parent_id = item.get("parent_id", 0)
            if parent_id:
                key = (item.get("file_path", ""), parent_id)
            else:
                key = (item.get("file_path", ""), item.get("page_num", 0), item.get("text", "")[:128])
            if key not in seen or item.get("rrf_score", 0) > seen[key].get("rrf_score", 0):
                seen[key] = item
        return list(seen.values())

    def _expand_parent_context(self, candidates: list[dict]) -> list[dict]:
        if not candidates:
            return []

        qdrant_ids = [int(item["qdrant_id"]) for item in candidates if item.get("qdrant_id") is not None]
        try:
            stored_contexts = self._store.get_chunk_context_by_qdrant_ids(qdrant_ids)
        except Exception:
            logger.debug("[retriever] parent context lookup failed", exc_info=True)
            stored_contexts = {}

        expanded: list[dict] = []
        seen: set[tuple] = set()
        for item in candidates:
            qid = int(item["qdrant_id"]) if item.get("qdrant_id") is not None else 0
            stored = stored_contexts.get(qid, {})
            child_text = stored.get("raw_text") or item.get("raw_text") or item.get("child_text") or item.get("text", "")
            parent_text = stored.get("parent_text") or item.get("parent_text") or child_text
            parent_id = stored.get("parent_id") or item.get("parent_id", 0)
            key = (
                item.get("file_path", ""),
                parent_id or item.get("page_num", 0),
                (parent_text or child_text)[:128],
            )
            if key in seen:
                continue
            seen.add(key)

            expanded_item = dict(item)
            expanded_item["matched_text"] = child_text
            expanded_item["child_text"] = child_text
            expanded_item["text"] = parent_text
            expanded_item["parent_id"] = parent_id
            expanded_item["parent_text_length"] = len(parent_text)
            expanded_item["contextual_prefix"] = (
                stored.get("contextual_prefix") or item.get("contextual_prefix", "")
            )
            expanded.append(expanded_item)

        return expanded

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
        if cancel_event is not None and cancel_event.is_set():
            return []
        try:
            reranked = self._rerank(
                query,
                candidates,
                cancel_event=cancel_event,
                top_k=top_k,
            )
            if reranked:
                return reranked
            if cancel_event is not None and cancel_event.is_set():
                return []
            degradations.append({
                "stage": "reranker",
                "status": "degraded",
                "error_type": "EmptyRerankResult",
                "message": "Reranker returned no results; using fused candidates.",
            })
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
        text = item.get("text", "") or ""
        return {
            "qdrant_id": item.get("qdrant_id"),
            "file_name": item.get("file_name", ""),
            "file_path": item.get("file_path", ""),
            "page_num": item.get("page_num", 0),
            "section": item.get("section", ""),
            "chunk_type": item.get("chunk_type", ""),
            "score": item.get("score"),
            "vec_score": item.get("vec_score"),
            "rrf_score": item.get("rrf_score"),
            "rerank_score": item.get("rerank_score"),
            "char_count": item.get("char_count", len(text)),
            "parent_id": item.get("parent_id", 0),
            "parent_text_length": item.get("parent_text_length", len(item.get("parent_text", "") or "")),
            "retrieval_status": item.get("retrieval_status", "ok"),
            "rerank_fallback": item.get("rerank_fallback", False),
            "matched_text_preview": (item.get("matched_text") or item.get("child_text") or "")[:max_text_chars],
            "text_preview": text[:max_text_chars],
            "text_length": len(text),
        }

    # ------------------------------------------------------------------
    # Rerank
    # ------------------------------------------------------------------

    def _rerank(
        self,
        query: str,
        candidates: list[dict],
        cancel_event=None,
        top_k: int | None = None,
    ) -> list[dict]:
        if cancel_event is not None and cancel_event.is_set():
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
