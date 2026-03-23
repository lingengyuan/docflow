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

    def upsert_embeddings(self, chunks, dense_vecs):
        return list(range(100, 100 + len(chunks)))

    def delete_file_vectors(self, qdrant_ids):
        return None


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
    cached = store.get_cached_embeddings("fake-embedding", [DocStore.compute_text_hash("shared chunk")])
    assert len(cached) == 1
