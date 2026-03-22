"""
测试 HybridRetriever 的 RRF 融合和 BM25 逻辑（mock 向量检索和模型）。
"""

import pytest
from src.query.retriever import HybridRetriever


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
