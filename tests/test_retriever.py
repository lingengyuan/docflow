"""
测试 HybridRetriever 的 RRF 融合和 BM25 逻辑（mock 向量检索和模型）。
"""

from src.query.retriever import HybridRetriever, QueryRouter


def make_item(qid: int, text: str = "sample", chunk_type: str = "text") -> dict:
    return {
        "qdrant_id": qid,
        "score": 0.9,
        "text": text,
        "file_name": "test.pdf",
        "file_path": "/tmp/test.pdf",
        "page_num": 1,
        "section": "",
        "chunk_type": chunk_type,
    }


class TestRRFFusion:
    def test_fuses_two_lists(self):
        vec = [make_item(1), make_item(2), make_item(3)]
        bm25 = [make_item(2), make_item(4), make_item(1)]
        result = HybridRetriever._rrf_fuse(vec, bm25)
        ids = [r["qdrant_id"] for r in result]
        # id=2 appears in both lists → highest RRF score
        assert ids[0] == 2

    def test_item_in_one_list_included(self):
        vec = [make_item(1)]
        bm25 = [make_item(2)]
        result = HybridRetriever._rrf_fuse(vec, bm25)
        ids = {r["qdrant_id"] for r in result}
        assert ids == {1, 2}

    def test_table_boost_applied(self):
        table_item = make_item(10, chunk_type="table")
        text_item = make_item(11, chunk_type="text")
        # table_item rank 2 in vec, text_item rank 1
        vec = [text_item, table_item]
        bm25 = []
        result_no_boost = HybridRetriever._rrf_fuse(vec, bm25, prefer_tables=False)
        result_boost = HybridRetriever._rrf_fuse(vec, bm25, prefer_tables=True)
        # Without boost: text_item (rank 1) wins
        assert result_no_boost[0]["qdrant_id"] == 11
        # With boost: table_item should be promoted
        assert result_boost[0]["qdrant_id"] == 10

    def test_empty_lists_return_empty(self):
        assert HybridRetriever._rrf_fuse([], []) == []

    def test_rrf_score_attached(self):
        vec = [make_item(1)]
        result = HybridRetriever._rrf_fuse(vec, [])
        assert "rrf_score" in result[0]
        assert result[0]["rrf_score"] > 0

    def test_moderate_vector_score_is_not_dropped_before_rerank(self):
        item = make_item(42)
        item["score"] = 0.35

        result = HybridRetriever._rrf_fuse([item], [])

        assert [r["qdrant_id"] for r in result] == [42]


class TestTableQueryDetection:
    def test_is_table_query(self):
        from src.query.engine import QueryEngine

        assert QueryEngine._is_table_query("Q3各区域销售数据汇总") is True
        assert QueryEngine._is_table_query("这份合同的违约条款是什么") is False
        assert QueryEngine._is_table_query("total sales amount") is True


class TestQueryRouter:
    def test_exact_query_uses_smaller_candidate_set(self):
        route = QueryRouter.classify('"INV2024" 2024-01 report.pdf')

        assert route["query_type"] == "exact"
        assert route["top_k_retrieval"] < 20
        assert route["top_k_rerank"] < 5

    def test_cross_document_query_uses_larger_candidate_set(self):
        route = QueryRouter.classify("比较多个文件里的 Phase 1 和 Phase 2 差异")

        assert route["query_type"] == "cross_document"
        assert route["top_k_retrieval"] > 20
        assert route["top_k_rerank"] > 5

    def test_mixed_language_technical_term_uses_keyword_route(self):
        route = QueryRouter.classify("扫描版 PDF 的 OCR 使用 glm-ocr 吗？")

        assert route["query_type"] == "keyword"
        assert route["bm25_weight"] > route["vec_weight"]

    def test_product_name_alone_does_not_force_keyword_route(self):
        route = QueryRouter.classify("DocFlow 当前主界面分成哪四个入口？")

        assert route["query_type"] == "semantic"


class FakeEmbedModel:
    def encode(self, texts, normalize_embeddings=True, convert_to_numpy=True):
        return [[0.1, 0.2, 0.3] for _ in texts]


class FakeDebugRetriever(HybridRetriever):
    def __init__(self):
        self.top_k_retrieval = 20
        self.top_k_rerank = 5
        self._store = None

    @property
    def embed_model(self):
        return FakeEmbedModel()

    def _vector_search(self, query_vec, file_filter, limit=None):
        return [
            make_item(1, text="alpha semantic match"),
            make_item(2, text="beta vector only"),
        ]

    def _fts_search(self, query, file_filter, limit=None, degradations=None):
        return [
            make_item(1, text="alpha keyword match"),
            make_item(3, text="gamma keyword only"),
        ]

    def _rerank(self, query, candidates, cancel_event=None, top_k=None):
        for i, item in enumerate(candidates):
            item["rerank_score"] = 1.0 - (i * 0.1)
        return candidates[: (top_k or self.top_k_rerank)]


class VectorFailingRetriever(FakeDebugRetriever):
    def _vector_search(self, query_vec, file_filter, limit=None):
        raise RuntimeError("qdrant down")


class FtsFailingRetriever(FakeDebugRetriever):
    def _fts_search(self, query, file_filter, limit=None, degradations=None):
        raise RuntimeError("sqlite locked")


class RerankFailingRetriever(FakeDebugRetriever):
    def _rerank(self, query, candidates, cancel_event=None, top_k=None):
        raise RuntimeError("reranker unavailable")


class TestDebugRetrieve:
    def test_debug_retrieve_returns_all_stages(self):
        retriever = FakeDebugRetriever()
        result = retriever.debug_retrieve("alpha question", include_rerank=True)

        assert result["query"] == "alpha question"
        assert set(result["stages"]) == {
            "vector",
            "fts",
            "fused",
            "deduped",
            "reranked",
            "parent_expanded",
        }
        assert result["stages"]["vector"][0]["qdrant_id"] == 1
        assert result["stages"]["fts"][1]["qdrant_id"] == 3
        assert result["stages"]["reranked"][0]["rerank_score"] == 1.0
        assert result["router"]["query_type"] == "balanced"
        assert "total_ms" in result["timings"]

    def test_debug_item_truncates_text(self):
        item = make_item(99, text="abcdef")
        debug_item = HybridRetriever._debug_item(item, max_text_chars=3)

        assert debug_item["text_preview"] == "abc"
        assert debug_item["text_length"] == 6

    def test_parent_context_expands_answer_text(self):
        retriever = FakeDebugRetriever()
        item = make_item(99, text="child hit")
        item["parent_id"] = 3
        item["parent_text"] = "child hit plus surrounding context"

        expanded = retriever._expand_parent_context([item])

        assert expanded[0]["text"] == "child hit plus surrounding context"
        assert expanded[0]["matched_text"] == "child hit"
        assert expanded[0]["parent_id"] == 3

    def test_vector_failure_keeps_fts_results(self):
        retriever = VectorFailingRetriever()

        result = retriever.debug_retrieve("alpha question", include_rerank=False)

        assert result["status"] == "degraded"
        assert result["degradations"][0]["stage"] == "vector"
        assert result["stages"]["vector"] == []
        assert result["stages"]["fts"][0]["qdrant_id"] == 1
        assert result["stages"]["parent_expanded"][0]["retrieval_status"] == "degraded"

    def test_fts_failure_keeps_vector_results(self):
        retriever = FtsFailingRetriever()

        result = retriever.debug_retrieve("alpha question", include_rerank=False)

        assert result["status"] == "degraded"
        assert result["degradations"][0]["stage"] == "fts"
        assert result["stages"]["fts"] == []
        assert result["stages"]["vector"][0]["qdrant_id"] == 1

    def test_reranker_failure_returns_fused_candidates(self):
        retriever = RerankFailingRetriever()

        result = retriever.debug_retrieve("alpha question", include_rerank=True)

        assert result["status"] == "degraded"
        assert result["degradations"][0]["stage"] == "reranker"
        assert result["stages"]["reranked"][0]["rerank_fallback"] is True
        assert result["stages"]["parent_expanded"][0]["qdrant_id"] == 1

    def test_full_text_mode_skips_vector_stage(self):
        retriever = FakeDebugRetriever()

        result = retriever.debug_retrieve(
            "alpha question",
            retrieval_mode="full_text",
            include_rerank=False,
        )

        assert result["retrieval_mode"] == "full_text"
        assert result["stages"]["vector"] == []
        assert result["stages"]["fts"][0]["qdrant_id"] == 1
        assert result["status"] == "ok"


class TestFtsDegradation:
    def test_fts_search_uses_sqlite_payload_when_qdrant_fetch_fails(self):
        class Store:
            def search_fts(self, fts_query, file_filter, limit):
                return [
                    {
                        "qdrant_id": 42,
                        "score": 1.5,
                        "file_name": "note.md",
                        "file_path": "/tmp/note.md",
                        "page_num": 2,
                        "section": "A",
                        "chunk_type": "text",
                        "char_count": 11,
                        "parent_id": 7,
                        "raw_text": "sqlite text",
                        "parent_text": "parent sqlite text",
                        "contextual_prefix": "",
                    }
                ]

            def search_fts_trigram(self, query, file_filter, limit):
                return []

        class BrokenQdrant:
            def retrieve(self, **kwargs):
                raise RuntimeError("qdrant unavailable")

        retriever = object.__new__(HybridRetriever)
        retriever.top_k_retrieval = 20
        retriever._store = Store()
        retriever._qdrant = BrokenQdrant()

        degradations = []
        result = retriever._fts_search("sqlite", None, limit=5, degradations=degradations)

        assert result[0]["qdrant_id"] == 42
        assert result[0]["text"] == "sqlite text"
        assert result[0]["parent_text"] == "parent sqlite text"
        assert degradations[0]["stage"] == "fts_payload"
