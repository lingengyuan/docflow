"""
测试 HybridRetriever 的 RRF 融合和 BM25 逻辑（mock 向量检索和模型）。
"""

import pytest
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

    def _fts_search(self, query, file_filter, limit=None):
        return [
            make_item(1, text="alpha keyword match"),
            make_item(3, text="gamma keyword only"),
        ]

    def _rerank(self, query, candidates, cancel_event=None, top_k=None):
        for i, item in enumerate(candidates):
            item["rerank_score"] = 1.0 - (i * 0.1)
        return candidates[: (top_k or self.top_k_rerank)]


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
