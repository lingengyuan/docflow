"""
测试 StructuredChunker 三层切块策略。
"""

import pytest
from src.ingest.chunker import StructuredChunker, Chunk


CTX = dict(file_name="test.pdf", file_path="/tmp/test.pdf", page_num=1)


class TestTableDetection:
    def test_splits_markdown_table(self):
        c = StructuredChunker()
        text = "Intro text.\n\n| Col1 | Col2 |\n|---|---|\n| A | 1 |\n| B | 2 |\n\nPost text."
        segments = c._split_tables(text)
        types = [t for t, _ in segments]
        assert "table" in types
        assert "text" in types

    def test_no_table_returns_single_text(self):
        c = StructuredChunker()
        text = "Just some normal text.\nMore text here."
        segments = c._split_tables(text)
        assert all(t == "text" for t, _ in segments)

    def test_pipe_line_without_separator_is_not_table(self):
        c = StructuredChunker()
        text = "Intro\n| not | a | real | table |\nOutro"
        segments = c._split_tables(text)
        assert segments == [("text", text)]

    def test_is_table_line(self):
        assert StructuredChunker._is_table_line("| A | B | C |")
        assert StructuredChunker._is_table_line("|---|---|---|")
        assert not StructuredChunker._is_table_line("Normal line")
        assert not StructuredChunker._is_table_line("")


class TestHeaderSplitting:
    def test_splits_on_markdown_headers(self):
        c = StructuredChunker()
        text = "Intro\n\n# Chapter 1\n\nContent 1.\n\n## Section 1.1\n\nDetails."
        sections = c._split_by_headers(text)
        headers = [h for h, _ in sections]
        assert any("Chapter 1" in h for h in headers)
        assert any("Section 1.1" in h for h in headers)

    def test_header_level(self):
        assert StructuredChunker._header_level("# H1") == 1
        assert StructuredChunker._header_level("## H2") == 2
        assert StructuredChunker._header_level("### H3") == 3


class TestRecursiveSplit:
    def test_invalid_overlap_rejected(self):
        with pytest.raises(ValueError):
            StructuredChunker(chunk_size=10, chunk_overlap=10)

    def test_short_text_not_split(self):
        c = StructuredChunker(chunk_size=512)
        text = "Short text."
        result = c._recursive_split(text)
        assert result == ["Short text."]

    def test_long_text_is_split(self):
        c = StructuredChunker(chunk_size=10, chunk_overlap=1)
        text = "word " * 100  # 500 chars, way over max_chars=20
        result = c._recursive_split(text)
        assert len(result) > 1

    def test_overlap_applied(self):
        c = StructuredChunker(chunk_size=5, chunk_overlap=2)
        text = "a" * 30
        result = c._hard_split(text, max_chars=10, overlap=3)
        assert len(result) > 1
        # For non-final chunks: tail of chunk N appears at start of chunk N+1
        # (last chunk may be shorter than overlap, so skip that pair)
        for i in range(len(result) - 2):
            overlap_tail = result[i][-3:]
            assert result[i + 1].startswith(overlap_tail)


class TestTableSummary:
    def test_generates_summary(self):
        table = "| 区域 | Q3销售额 | 增长率 |\n|---|---|---|\n| 华东 | 2,450,000 | 15% |\n| 华北 | 1,890,000 | 8% |"
        summary = StructuredChunker._table_summary(table, ["第三章", "销售数据"])
        assert "[表格]" in summary
        assert "2 行" in summary
        assert "区域" in summary

    def test_empty_table_returns_empty(self):
        assert StructuredChunker._table_summary("", []) == ""


class TestChunkPage:
    def test_returns_chunks_for_plain_text(self):
        c = StructuredChunker()
        chunks = c.chunk_page("Hello world. This is a test document.", **CTX)
        assert len(chunks) >= 1
        assert all(isinstance(ch, Chunk) for ch in chunks)
        assert all(ch.chunk_type == "text" for ch in chunks)

    def test_table_produces_two_chunks(self):
        c = StructuredChunker()
        text = "Report.\n\n| Name | Value |\n|---|---|\n| A | 1 |\n| B | 2 |"
        chunks = c.chunk_page(text, **CTX)
        types = {ch.chunk_type for ch in chunks}
        assert "table" in types
        assert "table_summary" in types

    def test_section_breadcrumb_attached(self):
        c = StructuredChunker()
        text = "# Chapter One\n\nSome content here."
        chunks = c.chunk_page(text, **CTX)
        text_chunks = [ch for ch in chunks if ch.chunk_type == "text"]
        assert any("Chapter One" in ch.section for ch in text_chunks)

    def test_char_count_set(self):
        c = StructuredChunker()
        chunks = c.chunk_page("Hello world.", **CTX)
        assert all(ch.char_count == len(ch.text) for ch in chunks)

    def test_metadata_propagated(self):
        c = StructuredChunker()
        chunks = c.chunk_page("Some content.", file_name="doc.pdf",
                               file_path="/docs/doc.pdf", page_num=3)
        assert all(ch.file_name == "doc.pdf" for ch in chunks)
        assert all(ch.page_num == 3 for ch in chunks)
