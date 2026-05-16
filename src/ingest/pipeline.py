"""
IngestPipeline — 将 ParserRegistry + StructuredChunker + Embedder + DocStore 串联。

支持格式：.pdf / .md / .markdown / .txt / .docx / 代码文本 / 可选图片格式

使用方式：
    pipeline = IngestPipeline.from_config("config.yaml")
    pipeline.ingest("/path/to/doc.pdf")
    pipeline.ingest("/path/to/note.md")
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from time import perf_counter

from src.config import DocFlowSettings
from src.domain_types import FileStatus
from src.embedding_backend import embedding_backend_config_from_dict
from src.ingest import pipeline_batch, pipeline_context, pipeline_vectors
from src.ingest.chunker import Chunk, StructuredChunker
from src.ingest.embedder import Embedder
from src.ingest.parsers import ParserRegistry
from src.ingest.pdf_analyzer import ParsedDocument
from src.ingest.pipeline_context import is_cjk_dominant as _is_cjk_dominant
from src.ingest.pipeline_types import IngestMetrics, PreparedIngestFile, ProgressCallback
from src.ingest.store import DocStore

logger = logging.getLogger(__name__)

class IngestPipeline:
    def __init__(
        self,
        registry: ParserRegistry,
        chunker: StructuredChunker,
        embedder: Embedder,
        store: DocStore,
        use_embedding_cache: bool = True,
        contextual_prefix_enabled: bool = False,
        contextual_prefix_mode: str = "metadata",
        contextual_prefix_model: str = "",
        ollama_base_url: str = "http://localhost:11434",
        parent_context_chars: int = 2048,
    ):
        self.registry = registry
        self.chunker = chunker
        self.embedder = embedder
        self.store = store
        self.use_embedding_cache = use_embedding_cache
        self.contextual_prefix_enabled = contextual_prefix_enabled
        self.contextual_prefix_mode = contextual_prefix_mode
        self.contextual_prefix_model = contextual_prefix_model
        self.ollama_base_url = ollama_base_url.rstrip("/")
        self.parent_context_chars = parent_context_chars

    @classmethod
    def from_config(cls, config_path: str | Path, store: DocStore | None = None) -> IngestPipeline:
        settings = DocFlowSettings.from_file(config_path)
        cfg = settings.raw
        embedding_config = embedding_backend_config_from_dict(cfg, config_path)

        registry = ParserRegistry.from_config(cfg)
        chunker = StructuredChunker(
            chunk_size=settings.chunking.chunk_size,
            chunk_overlap=settings.chunking.chunk_overlap,
        )
        embedder = Embedder(
            qdrant_host=settings.qdrant.host,
            qdrant_port=settings.qdrant.port,
            collection_name=settings.qdrant.collection,
            batch_size=settings.embedding.batch_size,
            id_counter_path=settings.paths.id_counter,
            adaptive_batch_char_budget=settings.ingest.adaptive_batch_char_budget,
            adaptive_batch_max=settings.ingest.adaptive_batch_max,
            embedding_config=embedding_config,
        )
        shared_store = store or DocStore(settings.paths.db_path)

        return cls(
            registry,
            chunker,
            embedder,
            shared_store,
            use_embedding_cache=settings.ingest.embedding_cache,
            contextual_prefix_enabled=settings.ingest.contextual_prefix,
            contextual_prefix_mode=settings.ingest.contextual_prefix_mode,
            contextual_prefix_model=settings.ingest.contextual_prefix_model,
            ollama_base_url=settings.ollama.base_url,
            parent_context_chars=settings.ingest.parent_context_chars,
        )

    def _parse_document(self, path: Path) -> tuple[ParsedDocument, str, float]:
        parser = self.registry.resolve(path)
        parse_start = perf_counter()
        doc = parser.parse(path)
        parse_s = perf_counter() - parse_start
        tags_json = json.dumps(doc.metadata.get("tags", []), ensure_ascii=False)
        return doc, tags_json, parse_s

    def _chunk_document(self, doc: ParsedDocument) -> tuple[list[Chunk], bool, float]:
        chunk_start = perf_counter()
        all_chunks: list[Chunk] = []
        for page in doc.pages:
            page_chunks = self.chunker.chunk_page(
                text=page.text,
                file_name=doc.file_name,
                file_path=str(doc.file_path),
                page_num=page.page_num,
                is_ocr=page.is_ocr,
            )
            all_chunks.extend(page_chunks)
        self._prepare_chunk_contexts(all_chunks)
        chunk_s = perf_counter() - chunk_start
        sample_text = doc.pages[0].text if doc.pages else ""
        return all_chunks, _is_cjk_dominant(sample_text), chunk_s

    def _prepare_chunk_contexts(self, chunks: list[Chunk]) -> None:
        pipeline_context.prepare_chunk_contexts(self, chunks)

    def _assign_parent_contexts(self, chunks: list[Chunk]) -> None:
        pipeline_context.assign_parent_contexts(self, chunks)

    def _apply_contextual_prefixes(self, chunks: list[Chunk]) -> None:
        pipeline_context.apply_contextual_prefixes(self, chunks)

    @staticmethod
    def _should_prefix_chunk(chunk: Chunk) -> bool:
        return pipeline_context.should_prefix_chunk(chunk)

    def _build_contextual_prefix(self, chunk: Chunk) -> str:
        return pipeline_context.build_contextual_prefix(self, chunk)

    def _build_ollama_contextual_prefix(self, chunk: Chunk) -> str:
        return pipeline_context.build_ollama_contextual_prefix(self, chunk)

    def close(self) -> None:
        self.embedder.close()

    @staticmethod
    def _chunk_embedding_text(chunk: Chunk) -> str:
        return pipeline_vectors.chunk_embedding_text(chunk)

    def prepare_file(self, file_path: str | Path) -> PreparedIngestFile | dict:
        """
        预处理单个文件：完成 hash / parse / chunk，但不执行 embedding。
        Returns:
          - PreparedIngestFile: 可进入后续微批 embedding
          - dict: skipped / unsupported / error
        """
        path = Path(file_path).expanduser().resolve()

        if not self.registry.supports(path):
            logger.info(f"Skip (unsupported): {path.name}")
            return {"status": "unsupported", "file": path.name, "chunks": 0}

        need, cached_hash = self.store.needs_ingest(path)
        if not need:
            logger.info(f"Skip (unchanged): {path.name}")
            return {"status": "skipped", "file": path.name, "chunks": 0}

        file_hash = cached_hash or DocStore.compute_hash(path)
        mtime_ns = path.stat().st_mtime_ns

        file_id = self.store.upsert_file(
            file_path=path,
            file_name=path.name,
            file_hash=file_hash,
            status=FileStatus.PROCESSING,
            mtime_ns=mtime_ns,
        )

        try:
            logger.info(f"Parsing: {path.name}")
            doc, tags_json, parse_s = self._parse_document(path)
            all_chunks, is_cjk, chunk_s = self._chunk_document(doc)
            old_qdrant_ids = self.store.get_file_qdrant_ids(file_id)

            self.store.upsert_file(
                file_path=path,
                file_name=path.name,
                file_hash=file_hash,
                status=FileStatus.PROCESSING,
                total_pages=doc.total_pages,
                is_scanned=doc.is_scanned,
                tags=tags_json,
                mtime_ns=mtime_ns,
            )

            logger.info(
                f"  Prepared: {path.name} → {len(all_chunks)} chunks from {doc.total_pages} pages "
                f"({'scanned' if doc.is_scanned else 'native'})"
            )
            return PreparedIngestFile(
                path=path,
                file_id=file_id,
                file_hash=file_hash,
                mtime_ns=mtime_ns,
                doc=doc,
                tags_json=tags_json,
                chunks=all_chunks,
                is_cjk=is_cjk,
                old_qdrant_ids=old_qdrant_ids,
                metrics=IngestMetrics(
                    parse_s=parse_s,
                    chunk_s=chunk_s,
                    total_s=parse_s + chunk_s,
                    chunk_count=len(all_chunks),
                ),
            )
        except Exception as e:
            logger.exception(f"Error preparing {path.name}")
            self.store.set_status(path, FileStatus.ERROR, error_msg=str(e))
            return {"status": "error", "file": path.name, "error": str(e)}

    def _build_vectors(
        self,
        chunks: list[Chunk],
        progress_callback: ProgressCallback | None = None,
    ):
        return pipeline_vectors.build_vectors(self, chunks, progress_callback=progress_callback)

    def _log_perf(self, file_name: str, metrics: IngestMetrics):
        pipeline_batch.log_perf(file_name, metrics)

    def ingest_prepared_batch(
        self,
        prepared_files: list[PreparedIngestFile],
        progress_callback: ProgressCallback | None = None,
    ) -> list[dict]:
        return pipeline_batch.ingest_prepared_batch(
            self,
            prepared_files,
            progress_callback=progress_callback,
        )

    def _safe_next_qdrant_id(self) -> int:
        return pipeline_batch.safe_next_qdrant_id(self)

    def ingest(self, file_path: str | Path) -> dict:
        """
        处理单个文件（PDF / MD / TXT / DOCX）。
        Returns: {"status": "done"|"skipped"|"error"|"unsupported", "file": ..., "chunks": N}
        """
        prepared = self.prepare_file(file_path)
        if not isinstance(prepared, PreparedIngestFile):
            return prepared
        return self.ingest_prepared_batch([prepared])[0]
