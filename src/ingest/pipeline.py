"""
IngestPipeline — 将 ParserRegistry + StructuredChunker + Embedder + DocStore 串联。

支持格式：.pdf / .md / .markdown / .txt / .docx

使用方式：
    pipeline = IngestPipeline.from_config("config.yaml")
    pipeline.ingest("/path/to/doc.pdf")
    pipeline.ingest("/path/to/note.md")
"""

from __future__ import annotations

import logging
from pathlib import Path

import yaml

from src.ingest.chunker import StructuredChunker
from src.ingest.embedder import Embedder
from src.ingest.parsers import ParserRegistry
from src.ingest.store import DocStore

logger = logging.getLogger(__name__)


def _fts_tokenize(text: str) -> str:
    """jieba 分词 → 空格分隔字符串，用于 FTS5 全文索引。"""
    import jieba
    return " ".join(t for t in jieba.cut(text.lower()) if t.strip())


class IngestPipeline:
    def __init__(
        self,
        registry: ParserRegistry,
        chunker: StructuredChunker,
        embedder: Embedder,
        store: DocStore,
    ):
        self.registry = registry
        self.chunker = chunker
        self.embedder = embedder
        self.store = store

    @classmethod
    def from_config(cls, config_path: str | Path) -> "IngestPipeline":
        with open(config_path) as f:
            cfg = yaml.safe_load(f)

        db_path = Path(cfg["paths"]["db_path"]).expanduser()

        registry = ParserRegistry.from_config(cfg)
        chunker = StructuredChunker(
            chunk_size=cfg["chunking"]["chunk_size"],
            chunk_overlap=cfg["chunking"]["chunk_overlap"],
        )
        id_counter = Path(
            cfg["paths"].get("id_counter", "qdrant_id_counter.txt")
        ).expanduser()
        embedder = Embedder(
            qdrant_host=cfg["qdrant"]["host"],
            qdrant_port=cfg["qdrant"]["port"],
            embedding_model=cfg["embedding"]["model"],
            batch_size=cfg["embedding"]["batch_size"],
            device=cfg["embedding"]["device"],
            id_counter_path=id_counter,
        )
        store = DocStore(db_path)

        return cls(registry, chunker, embedder, store)

    def ingest(self, file_path: str | Path) -> dict:
        """
        处理单个文件（PDF / MD / TXT / DOCX）。
        Returns: {"status": "done"|"skipped"|"error"|"unsupported", "file": ..., "chunks": N}
        """
        path = Path(file_path).expanduser().resolve()

        if not self.registry.supports(path):
            logger.info(f"Skip (unsupported): {path.name}")
            return {"status": "unsupported", "file": path.name, "chunks": 0}

        if not self.store.needs_ingest(path):
            logger.info(f"Skip (unchanged): {path.name}")
            return {"status": "skipped", "file": path.name, "chunks": 0}

        file_hash = DocStore.compute_hash(path)

        file_id = self.store.upsert_file(
            file_path=path,
            file_name=path.name,
            file_hash=file_hash,
            status="processing",
        )
        self.store.set_status(path, "processing")

        try:
            # Delete old vectors if re-ingesting
            old_qdrant_ids = self.store.get_file_qdrant_ids(file_id)
            if old_qdrant_ids:
                logger.info(f"  Re-ingesting: removing {len(old_qdrant_ids)} old vectors for {path.name}")
                self.embedder.delete_file_vectors(old_qdrant_ids)

            # Parse file using registry
            logger.info(f"Parsing: {path.name}")
            parser = self.registry.resolve(path)
            doc = parser.parse(path)

            self.store.upsert_file(
                file_path=path,
                file_name=path.name,
                file_hash=file_hash,
                status="processing",
                total_pages=doc.total_pages,
                is_scanned=doc.is_scanned,
            )

            # Chunk all pages
            all_chunks = []
            for page in doc.pages:
                page_chunks = self.chunker.chunk_page(
                    text=page.text,
                    file_name=doc.file_name,
                    file_path=str(doc.file_path),
                    page_num=page.page_num,
                    is_ocr=page.is_ocr,
                )
                all_chunks.extend(page_chunks)

            logger.info(f"  {path.name}: {len(all_chunks)} chunks from {doc.total_pages} pages "
                        f"({'scanned' if doc.is_scanned else 'native'})")

            # Embed & store vectors
            qdrant_ids = self.embedder.embed_chunks(all_chunks)

            # Save chunk metadata + FTS5 tokenized text to SQLite
            chunk_records = [
                {
                    "qdrant_id": qdrant_ids[i],
                    "chunk_type": all_chunks[i].chunk_type,
                    "page_num": all_chunks[i].page_num,
                    "section": all_chunks[i].section,
                    "char_count": all_chunks[i].char_count,
                    "tokenized_text": _fts_tokenize(all_chunks[i].text),
                }
                for i in range(len(all_chunks))
            ]
            self.store.add_chunks(file_id, chunk_records)
            self.store.set_chunk_count(path, len(all_chunks))
            self.store.set_status(path, "done")

            logger.info(f"  Done: {path.name} → {len(all_chunks)} chunks indexed")
            return {"status": "done", "file": path.name, "chunks": len(all_chunks)}

        except Exception as e:
            logger.exception(f"Error ingesting {path.name}")
            self.store.set_status(path, "error", error_msg=str(e))
            return {"status": "error", "file": path.name, "error": str(e)}
