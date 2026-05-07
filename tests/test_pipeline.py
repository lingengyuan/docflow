from pathlib import Path

import numpy as np

from src.ingest.chunker import Chunk
from src.ingest.pipeline import IngestPipeline
from src.ingest.pdf_analyzer import PageContent, ParsedDocument
from src.ingest.store import DocStore


class FakeParser:
    def parse(self, file_path: Path) -> ParsedDocument:
        return ParsedDocument(
            file_path=file_path,
            file_name=file_path.name,
            total_pages=1,
            is_scanned=False,
            pages=[PageContent(page_num=1, text="shared chunk", headers=[])],
            metadata={},
        )


class FakeRegistry:
    supported_extensions = [".txt"]

    def __init__(self):
        self.parser = FakeParser()

    def supports(self, path: Path) -> bool:
        return path.suffix == ".txt"

    def resolve(self, path: Path) -> FakeParser:
        return self.parser


class FakeChunker:
    def chunk_page(self, text, file_name, file_path, page_num, is_ocr=False):
        return [
            Chunk(
                text=text,
                chunk_type="text",
                file_name=file_name,
                file_path=file_path,
                page_num=page_num,
            )
        ]


class FakeEmbedder:
    def __init__(self):
        self.encode_calls: list[list[str]] = []
        self.embedding_model_name = "fake-embedding"
        self.embedding_cache_key = "torch::fake-embedding"
        self.min_next_ids: list[int | None] = []
        self.qdrant_max_id = -1

    def encode_texts(self, texts, progress_callback=None):
        self.encode_calls.append(list(texts))
        if progress_callback:
            progress_callback(
                {
                    "encoded_texts": len(texts),
                    "total_texts": len(texts),
                    "batch_size": len(texts) or 1,
                }
        )
        return np.asarray([[0.1, 0.2, 0.3] for _ in texts], dtype=np.float32)

    def upsert_embeddings(self, chunks, dense_vecs, min_next_id=None):
        self.min_next_ids.append(min_next_id)
        start = min_next_id or 100
        return list(range(start, start + len(chunks)))

    def delete_file_vectors(self, qdrant_ids):
        return None

    def max_point_id(self):
        return self.qdrant_max_id


def _make_file(path: Path):
    path.write_text("shared chunk", encoding="utf-8")
    return path


def test_embedding_cache_reuses_duplicate_chunks_across_batches(tmp_path):
    store = DocStore(tmp_path / "docflow.db")
    pipeline = IngestPipeline(
        registry=FakeRegistry(),
        chunker=FakeChunker(),
        embedder=FakeEmbedder(),
        store=store,
        use_embedding_cache=True,
    )

    file_a = _make_file(tmp_path / "a.txt")
    file_b = _make_file(tmp_path / "b.txt")
    file_c = _make_file(tmp_path / "c.txt")

    prepared_a = pipeline.prepare_file(file_a)
    prepared_b = pipeline.prepare_file(file_b)
    results_ab = pipeline.ingest_prepared_batch([prepared_a, prepared_b])

    assert [r["status"] for r in results_ab] == ["done", "done"]
    assert pipeline.embedder.encode_calls == [["shared chunk"]]

    prepared_c = pipeline.prepare_file(file_c)
    result_c = pipeline.ingest_prepared_batch([prepared_c])[0]

    assert result_c["status"] == "done"
    assert pipeline.embedder.encode_calls == [["shared chunk"]]
    cached = store.get_cached_embeddings(
        pipeline.embedder.embedding_cache_key,
        [DocStore.compute_text_hash("shared chunk")],
    )
    assert len(cached) == 1


def test_parent_context_groups_adjacent_chunks(tmp_path):
    store = DocStore(tmp_path / "docflow.db")
    pipeline = IngestPipeline(
        registry=FakeRegistry(),
        chunker=FakeChunker(),
        embedder=FakeEmbedder(),
        store=store,
        parent_context_chars=1000,
    )
    chunks = [
        Chunk("first", "text", "note.md", "/tmp/note.md", 1, section="A"),
        Chunk("second", "text", "note.md", "/tmp/note.md", 1, section="A"),
        Chunk("third", "text", "note.md", "/tmp/note.md", 1, section="B"),
    ]

    pipeline._prepare_chunk_contexts(chunks)

    assert chunks[0].parent_id == chunks[1].parent_id
    assert chunks[0].parent_text == "first\n\nsecond"
    assert chunks[2].parent_id != chunks[0].parent_id


def test_contextual_prefix_uses_embedding_text_without_changing_raw_text(tmp_path):
    store = DocStore(tmp_path / "docflow.db")
    embedder = FakeEmbedder()
    pipeline = IngestPipeline(
        registry=FakeRegistry(),
        chunker=FakeChunker(),
        embedder=embedder,
        store=store,
        use_embedding_cache=False,
        contextual_prefix_enabled=True,
    )
    chunk = Chunk("raw body", "text", "note.md", "/tmp/note.md", 1, section="Plan")
    pipeline._prepare_chunk_contexts([chunk])

    pipeline._build_vectors([chunk])

    assert chunk.raw_text == "raw body"
    assert chunk.text == "raw body"
    assert chunk.contextual_prefix == "File: note.md | Section: Plan"
    assert embedder.encode_calls == [["File: note.md | Section: Plan\n\nraw body"]]


def test_ingest_advances_vector_id_floor_from_sqlite_and_qdrant(tmp_path):
    store = DocStore(tmp_path / "docflow.db")
    old_source = _make_file(tmp_path / "old.txt")
    old_file_id = store.upsert_file(old_source, old_source.name, "old-hash", status="done")
    store.add_chunks(
        old_file_id,
        [
            {
                "qdrant_id": 20,
                "chunk_type": "text",
                "page_num": 1,
                "section": "",
                "char_count": 5,
                "raw_text": "old",
            }
        ],
    )
    store.set_chunk_count(old_source, 1)

    embedder = FakeEmbedder()
    embedder.qdrant_max_id = 30
    pipeline = IngestPipeline(
        registry=FakeRegistry(),
        chunker=FakeChunker(),
        embedder=embedder,
        store=store,
        use_embedding_cache=False,
    )

    result = pipeline.ingest(_make_file(tmp_path / "new.txt"))

    assert result["status"] == "done"
    assert embedder.min_next_ids == [31]
    indexed = store.get_file_by_path(tmp_path / "new.txt")
    assert store.get_file_qdrant_ids(indexed["id"]) == [31]
