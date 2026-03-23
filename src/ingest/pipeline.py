"""
IngestPipeline — 将 ParserRegistry + StructuredChunker + Embedder + DocStore 串联。

支持格式：.pdf / .md / .markdown / .txt / .docx

使用方式：
    pipeline = IngestPipeline.from_config("config.yaml")
    pipeline.ingest("/path/to/doc.pdf")
    pipeline.ingest("/path/to/note.md")
"""

from __future__ import annotations

import json
import logging
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from time import perf_counter
from typing import Callable

import numpy as np
import yaml

from src.embedding_backend import embedding_backend_config_from_dict
from src.ingest.chunker import Chunk, StructuredChunker
from src.ingest.embedder import Embedder
from src.ingest.parsers import ParserRegistry
from src.ingest.pdf_analyzer import ParsedDocument
from src.ingest.store import DocStore

logger = logging.getLogger(__name__)

ProgressCallback = Callable[[dict], None]


def _is_cjk_dominant(text: str, threshold: float = 0.2) -> bool:
    """CJK 字符占比超过阈值则视为中文主导。"""
    if not text:
        return False
    cjk_count = sum(1 for c in text if "\u4e00" <= c <= "\u9fff" or "\u3400" <= c <= "\u4dbf")
    return cjk_count / len(text) > threshold


def _fts_tokenize(text: str, is_cjk: bool | None = None) -> str:
    """
    分词 → 空格分隔字符串，用于 FTS5 精确匹配索引。
    中文主导：jieba 分词；英文主导：直接小写（利用 FTS5 unicode61 英文处理）。
    传入 is_cjk 可跳过逐 chunk 的语言检测。
    """
    if is_cjk is None:
        is_cjk = _is_cjk_dominant(text)
    if is_cjk:
        import jieba
        return " ".join(t for t in jieba.cut(text.lower()) if t.strip())
    return text.lower()


@dataclass
class IngestMetrics:
    parse_s: float = 0.0
    chunk_s: float = 0.0
    embed_s: float = 0.0
    qdrant_s: float = 0.0
    sqlite_s: float = 0.0
    total_s: float = 0.0
    cache_hits: int = 0
    cache_misses: int = 0
    chunk_count: int = 0

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class PreparedIngestFile:
    path: Path
    file_id: int
    file_hash: str
    mtime_ns: int
    doc: ParsedDocument
    tags_json: str
    chunks: list[Chunk]
    is_cjk: bool
    old_qdrant_ids: list[int]
    metrics: IngestMetrics


class IngestPipeline:
    def __init__(
        self,
        registry: ParserRegistry,
        chunker: StructuredChunker,
        embedder: Embedder,
        store: DocStore,
        use_embedding_cache: bool = True,
    ):
        self.registry = registry
        self.chunker = chunker
        self.embedder = embedder
        self.store = store
        self.use_embedding_cache = use_embedding_cache

    @classmethod
    def from_config(cls, config_path: str | Path, store: DocStore | None = None) -> "IngestPipeline":
        with open(config_path) as f:
            cfg = yaml.safe_load(f)

        db_path = Path(cfg["paths"]["db_path"]).expanduser()
        ingest_cfg = cfg.get("ingest", {})
        embedding_config = embedding_backend_config_from_dict(cfg, config_path)

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
            batch_size=cfg["embedding"]["batch_size"],
            id_counter_path=id_counter,
            adaptive_batch_char_budget=ingest_cfg.get("adaptive_batch_char_budget"),
            adaptive_batch_max=ingest_cfg.get("adaptive_batch_max"),
            embedding_config=embedding_config,
        )
        shared_store = store or DocStore(db_path)

        return cls(
            registry,
            chunker,
            embedder,
            shared_store,
            use_embedding_cache=ingest_cfg.get("embedding_cache", True),
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
        chunk_s = perf_counter() - chunk_start
        sample_text = doc.pages[0].text if doc.pages else ""
        return all_chunks, _is_cjk_dominant(sample_text), chunk_s

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
            status="processing",
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
                status="processing",
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
            self.store.set_status(path, "error", error_msg=str(e))
            return {"status": "error", "file": path.name, "error": str(e)}

    def _build_vectors(
        self,
        chunks: list[Chunk],
        progress_callback: ProgressCallback | None = None,
    ) -> tuple[np.ndarray, list[str], set[str], float]:
        if not chunks:
            return np.empty((0, 0), dtype=np.float32), [], set(), 0.0

        embed_start = perf_counter()
        text_hashes = [DocStore.compute_text_hash(chunk.text) for chunk in chunks]
        hash_counts = Counter(text_hashes)
        cached_vectors = (
            self.store.get_cached_embeddings(self.embedder.embedding_cache_key, text_hashes)
            if self.use_embedding_cache
            else {}
        )
        cached_hashes = set(cached_vectors.keys())
        cache_hits = sum(hash_counts[text_hash] for text_hash in cached_hashes)

        missing_hashes: list[str] = []
        missing_texts: list[str] = []
        seen_missing: set[str] = set()
        for text_hash, chunk in zip(text_hashes, chunks):
            if text_hash in cached_hashes or text_hash in seen_missing:
                continue
            seen_missing.add(text_hash)
            missing_hashes.append(text_hash)
            missing_texts.append(chunk.text)

        if progress_callback:
            progress_callback(
                {
                    "stage": "embedding",
                    "processed_chunks": cache_hits,
                    "total_chunks": len(chunks),
                    "cache_hits": cache_hits,
                    "cache_misses": len(chunks) - cache_hits,
                    "adaptive_batch_size": None,
                }
            )

        missing_progress: list[int] = []
        cumulative = 0
        for text_hash in missing_hashes:
            cumulative += hash_counts[text_hash]
            missing_progress.append(cumulative)

        def _on_encode(update: dict):
            processed = cache_hits
            if update["encoded_texts"]:
                processed += missing_progress[update["encoded_texts"] - 1]
            if progress_callback:
                progress_callback(
                    {
                        "stage": "embedding",
                        "processed_chunks": processed,
                        "total_chunks": len(chunks),
                        "cache_hits": cache_hits,
                        "cache_misses": len(chunks) - cache_hits,
                        "adaptive_batch_size": update["batch_size"],
                    }
                )

        vectors_by_hash: dict[str, np.ndarray] = {
            text_hash: np.asarray(vector, dtype=np.float32)
            for text_hash, vector in cached_vectors.items()
        }
        if missing_texts:
            encoded_vectors = self.embedder.encode_texts(
                missing_texts,
                progress_callback=_on_encode,
            )
            new_vectors = {
                missing_hashes[i]: np.asarray(encoded_vectors[i], dtype=np.float32)
                for i in range(len(missing_hashes))
            }
            vectors_by_hash.update(new_vectors)
            if self.use_embedding_cache:
                self.store.put_cached_embeddings(self.embedder.embedding_cache_key, new_vectors)
        elif progress_callback:
            progress_callback(
                {
                    "stage": "embedding",
                    "processed_chunks": len(chunks),
                    "total_chunks": len(chunks),
                    "cache_hits": cache_hits,
                    "cache_misses": len(chunks) - cache_hits,
                    "adaptive_batch_size": None,
                }
            )

        vectors = np.stack([vectors_by_hash[text_hash] for text_hash in text_hashes]).astype(np.float32)
        embed_s = perf_counter() - embed_start
        return vectors, text_hashes, cached_hashes, embed_s

    def _log_perf(self, file_name: str, metrics: IngestMetrics):
        logger.info(
            "[perf] %s parse=%.3fs chunk=%.3fs embed=%.3fs qdrant=%.3fs sqlite=%.3fs total=%.3fs "
            "chunks=%d cache_hits=%d cache_misses=%d",
            file_name,
            metrics.parse_s,
            metrics.chunk_s,
            metrics.embed_s,
            metrics.qdrant_s,
            metrics.sqlite_s,
            metrics.total_s,
            metrics.chunk_count,
            metrics.cache_hits,
            metrics.cache_misses,
        )

    def ingest_prepared_batch(
        self,
        prepared_files: list[PreparedIngestFile],
        progress_callback: ProgressCallback | None = None,
    ) -> list[dict]:
        if not prepared_files:
            return []

        total_chunks = sum(len(prepared.chunks) for prepared in prepared_files)
        all_chunks = [chunk for prepared in prepared_files for chunk in prepared.chunks]
        text_hashes: list[str] = []
        cached_hashes: set[str] = set()

        try:
            vectors, text_hashes, cached_hashes, embed_s = self._build_vectors(
                all_chunks,
                progress_callback=progress_callback,
            )

            qdrant_start = perf_counter()
            for prepared in prepared_files:
                if prepared.old_qdrant_ids:
                    logger.info(
                        f"  Re-ingesting: removing {len(prepared.old_qdrant_ids)} old vectors for {prepared.path.name}"
                    )
                    self.embedder.delete_file_vectors(prepared.old_qdrant_ids)
            qdrant_ids = self.embedder.upsert_embeddings(all_chunks, vectors)
            qdrant_s = perf_counter() - qdrant_start
        except Exception as e:
            logger.exception("Error embedding ingest batch")
            results = []
            for prepared in prepared_files:
                self.store.set_status(prepared.path, "error", error_msg=str(e))
                metrics = prepared.metrics
                metrics.embed_s = embed_s if "embed_s" in locals() else 0.0
                metrics.qdrant_s = 0.0
                metrics.total_s = metrics.parse_s + metrics.chunk_s + metrics.embed_s
                results.append(
                    {
                        "status": "error",
                        "file": prepared.path.name,
                        "error": str(e),
                        "metrics": metrics.to_dict(),
                    }
                )
            return results

        if progress_callback:
            progress_callback(
                {
                    "stage": "storing",
                    "processed_chunks": total_chunks,
                    "total_chunks": total_chunks,
                    "cache_hits": sum(1 for text_hash in text_hashes if text_hash in cached_hashes),
                    "cache_misses": sum(1 for text_hash in text_hashes if text_hash not in cached_hashes),
                    "adaptive_batch_size": None,
                }
            )

        results: list[dict] = []
        start = 0
        for prepared in prepared_files:
            end = start + len(prepared.chunks)
            file_qdrant_ids = qdrant_ids[start:end]
            file_text_hashes = text_hashes[start:end]
            metrics = prepared.metrics
            ratio = (len(prepared.chunks) / total_chunks) if total_chunks else 0.0
            metrics.embed_s = embed_s * ratio
            metrics.qdrant_s = qdrant_s * ratio
            metrics.cache_hits = sum(1 for text_hash in file_text_hashes if text_hash in cached_hashes)
            metrics.cache_misses = len(file_text_hashes) - metrics.cache_hits

            if progress_callback:
                progress_callback(
                    {
                        "stage": "storing",
                        "processed_chunks": total_chunks,
                        "total_chunks": total_chunks,
                        "current_file": prepared.path.name,
                        "current_path": str(prepared.path),
                        "cache_hits": metrics.cache_hits,
                        "cache_misses": metrics.cache_misses,
                        "adaptive_batch_size": None,
                    }
                )

            sqlite_start = perf_counter()
            try:
                chunk_records = [
                    {
                        "qdrant_id": file_qdrant_ids[i],
                        "chunk_type": prepared.chunks[i].chunk_type,
                        "page_num": prepared.chunks[i].page_num,
                        "section": prepared.chunks[i].section,
                        "char_count": prepared.chunks[i].char_count,
                        "tokenized_text": _fts_tokenize(prepared.chunks[i].text, is_cjk=prepared.is_cjk),
                        "raw_text": prepared.chunks[i].text,
                    }
                    for i in range(len(prepared.chunks))
                ]
                self.store.add_chunks(prepared.file_id, chunk_records)
                self.store.set_chunk_count(prepared.path, len(prepared.chunks))
                self.store.set_status(prepared.path, "done")
                metrics.sqlite_s = perf_counter() - sqlite_start
                metrics.total_s = (
                    metrics.parse_s
                    + metrics.chunk_s
                    + metrics.embed_s
                    + metrics.qdrant_s
                    + metrics.sqlite_s
                )
                self._log_perf(prepared.path.name, metrics)
                logger.info(f"  Done: {prepared.path.name} → {len(prepared.chunks)} chunks indexed")
                results.append(
                    {
                        "status": "done",
                        "file": prepared.path.name,
                        "chunks": len(prepared.chunks),
                        "metrics": metrics.to_dict(),
                    }
                )
            except Exception as e:
                logger.exception(f"Error finalizing {prepared.path.name}")
                self.store.set_status(prepared.path, "error", error_msg=str(e))
                self.embedder.delete_file_vectors(file_qdrant_ids)
                metrics.sqlite_s = perf_counter() - sqlite_start
                metrics.total_s = (
                    metrics.parse_s
                    + metrics.chunk_s
                    + metrics.embed_s
                    + metrics.qdrant_s
                    + metrics.sqlite_s
                )
                results.append(
                    {
                        "status": "error",
                        "file": prepared.path.name,
                        "error": str(e),
                        "metrics": metrics.to_dict(),
                    }
                )
            start = end

        return results

    def ingest(self, file_path: str | Path) -> dict:
        """
        处理单个文件（PDF / MD / TXT / DOCX）。
        Returns: {"status": "done"|"skipped"|"error"|"unsupported", "file": ..., "chunks": N}
        """
        prepared = self.prepare_file(file_path)
        if not isinstance(prepared, PreparedIngestFile):
            return prepared
        return self.ingest_prepared_batch([prepared])[0]

    def benchmark_file(self, file_path: str | Path) -> dict:
        """
        端到端 benchmark（dry-run）：parse / chunk / embed 全部执行，但不写入 Qdrant / SQLite chunk 元数据。
        """
        path = Path(file_path).expanduser().resolve()
        if not self.registry.supports(path):
            return {"status": "unsupported", "file": path.name, "chunks": 0}

        try:
            doc, _, parse_s = self._parse_document(path)
            chunks, _, chunk_s = self._chunk_document(doc)
            vectors, text_hashes, cached_hashes, embed_s = self._build_vectors(chunks)
            metrics = IngestMetrics(
                parse_s=parse_s,
                chunk_s=chunk_s,
                embed_s=embed_s,
                total_s=parse_s + chunk_s + embed_s,
                chunk_count=len(chunks),
                cache_hits=sum(1 for text_hash in text_hashes if text_hash in cached_hashes),
                cache_misses=sum(1 for text_hash in text_hashes if text_hash not in cached_hashes),
            )
            if vectors.size and self.embedder._vector_dim is None:
                self.embedder._vector_dim = vectors.shape[1]
            self._log_perf(f"{path.name} (benchmark)", metrics)
            return {
                "status": "benchmarked",
                "file": path.name,
                "pages": doc.total_pages,
                "chunks": len(chunks),
                "metrics": metrics.to_dict(),
                "dry_run": True,
            }
        except Exception as e:
            logger.exception(f"Error benchmarking {path.name}")
            return {"status": "error", "file": path.name, "error": str(e)}
