from __future__ import annotations

from pathlib import Path
from time import monotonic

import numpy as np

from src.ingest.chunker import Chunk
from src.ingest.pdf_analyzer import PageContent, ParsedDocument
from src.ingest.pipeline import IngestPipeline
from src.ingest.store import DocStore


class TextFileParser:
    def parse(self, file_path: Path) -> ParsedDocument:
        text = file_path.read_text(encoding="utf-8")
        return ParsedDocument(
            file_path=file_path,
            file_name=file_path.name,
            total_pages=1,
            is_scanned=False,
            pages=[PageContent(page_num=1, text=text, headers=[])],
            metadata={},
        )


class TextRegistry:
    supported_extensions = [".txt"]

    def supports(self, path: Path) -> bool:
        return path.suffix == ".txt"

    def resolve(self, path: Path) -> TextFileParser:
        return TextFileParser()


class SingleChunker:
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


class TrackingEmbedder:
    def __init__(self):
        self.embedding_model_name = "incremental-test"
        self.embedding_cache_key = "test::incremental"
        self.deleted_vectors: list[list[int]] = []
        self.encoded_texts: list[list[str]] = []

    def encode_texts(self, texts, progress_callback=None):
        self.encoded_texts.append(list(texts))
        return np.asarray([[0.1, 0.2, 0.3] for _ in texts], dtype=np.float32)

    def upsert_embeddings(self, chunks, dense_vecs, min_next_id=None):
        start = int(min_next_id or 0)
        return list(range(start, start + len(chunks)))

    def delete_file_vectors(self, qdrant_ids):
        self.deleted_vectors.append(list(qdrant_ids))

    def max_point_id(self):
        return -1


def test_incremental_index_add_modify_delete_cycle_finishes_under_limit(tmp_path):
    store = DocStore(tmp_path / "docflow.db")
    embedder = TrackingEmbedder()
    pipeline = IngestPipeline(
        registry=TextRegistry(),
        chunker=SingleChunker(),
        embedder=embedder,
        store=store,
        use_embedding_cache=False,
    )
    source = tmp_path / "daily-note.txt"

    started = monotonic()
    source.write_text("alpha source evidence", encoding="utf-8")
    added = pipeline.ingest(source)

    assert added["status"] == "done"
    indexed = store.get_file_by_path(source.resolve())
    assert indexed is not None
    assert store.list_file_chunks(indexed["id"])[0]["raw_text"] == "alpha source evidence"

    source.write_text("beta source evidence", encoding="utf-8")
    modified = pipeline.ingest(source)

    assert modified["status"] == "done"
    indexed = store.get_file_by_path(source.resolve())
    assert indexed is not None
    assert store.list_file_chunks(indexed["id"])[0]["raw_text"] == "beta source evidence"
    assert embedder.deleted_vectors == [[0]]

    source.unlink()
    removed = store.cleanup_deleted_files()

    assert removed == [{"file_name": "daily-note.txt", "qdrant_ids": [1]}]
    assert store.get_file_by_path(source.resolve()) is None
    assert monotonic() - started < 5
