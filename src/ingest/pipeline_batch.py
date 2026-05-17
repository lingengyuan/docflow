"""Prepared ingest batch finalization."""

from __future__ import annotations

import logging
from time import perf_counter

from src.domain_types import FileStatus
from src.ingest.pipeline_context import fts_tokenize
from src.ingest.pipeline_types import IngestMetrics, PreparedIngestFile, ProgressCallback

logger = logging.getLogger(__name__)


def log_perf(file_name: str, metrics: IngestMetrics) -> None:
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
    pipeline,
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
        vectors, text_hashes, cached_hashes, embed_s = pipeline._build_vectors(
            all_chunks,
            progress_callback=progress_callback,
        )

        qdrant_start = perf_counter()
        for prepared in prepared_files:
            if prepared.old_qdrant_ids:
                logger.info(
                    "  Re-ingesting: removing "
                    f"{len(prepared.old_qdrant_ids)} old vectors for {prepared.path.name}"
                )
                pipeline.embedder.delete_file_vectors(prepared.old_qdrant_ids)
        qdrant_ids = pipeline.embedder.upsert_embeddings(
            all_chunks,
            vectors,
            min_next_id=pipeline._safe_next_qdrant_id(),
        )
        qdrant_s = perf_counter() - qdrant_start
    except Exception as e:
        logger.exception("Error embedding ingest batch")
        error_results = []
        for prepared in prepared_files:
            pipeline.store.set_status(prepared.path, FileStatus.ERROR, error_msg=str(e))
            metrics = prepared.metrics
            metrics.embed_s = embed_s if "embed_s" in locals() else 0.0
            metrics.qdrant_s = 0.0
            metrics.total_s = metrics.parse_s + metrics.chunk_s + metrics.embed_s
            error_results.append(
                {
                    "status": "error",
                    "file": prepared.path.name,
                    "error": str(e),
                    "metrics": metrics.to_dict(),
                }
            )
        return error_results

    if progress_callback:
        progress_callback(
            {
                "stage": "storing",
                "processed_chunks": total_chunks,
                "total_chunks": total_chunks,
                "cache_hits": sum(1 for text_hash in text_hashes if text_hash in cached_hashes),
                "cache_misses": sum(
                    1 for text_hash in text_hashes if text_hash not in cached_hashes
                ),
                "adaptive_batch_size": None,
            }
        )

    results: list[dict] = []
    start = 0
    for prepared in prepared_files:
        end = start + len(prepared.chunks)
        file_qdrant_ids = qdrant_ids[start:end]
        file_text_hashes = text_hashes[start:end]
        result = _finalize_prepared_file(
            pipeline,
            prepared,
            file_qdrant_ids,
            file_text_hashes,
            total_chunks=total_chunks,
            embed_s=embed_s,
            qdrant_s=qdrant_s,
            cached_hashes=cached_hashes,
            progress_callback=progress_callback,
        )
        results.append(result)
        start = end

    return results


def _finalize_prepared_file(
    pipeline,
    prepared: PreparedIngestFile,
    file_qdrant_ids: list[int],
    file_text_hashes: list[str],
    *,
    total_chunks: int,
    embed_s: float,
    qdrant_s: float,
    cached_hashes: set[str],
    progress_callback: ProgressCallback | None,
) -> dict:
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
                "parent_id": prepared.chunks[i].parent_id,
                "raw_text": prepared.chunks[i].raw_text,
                "embedding_text": prepared.chunks[i].embedding_text,
                "parent_text": prepared.chunks[i].parent_text,
                "contextual_prefix": prepared.chunks[i].contextual_prefix,
                "tokenized_text": fts_tokenize(prepared.chunks[i].raw_text, is_cjk=prepared.is_cjk),
            }
            for i in range(len(prepared.chunks))
        ]
        pipeline.store.add_chunks(prepared.file_id, chunk_records)
        pipeline.store.set_chunk_count(prepared.path, len(prepared.chunks))
        pipeline.store.set_status(prepared.path, FileStatus.DONE)
        metrics.sqlite_s = perf_counter() - sqlite_start
        metrics.total_s = _total_metrics_s(metrics)
        log_perf(prepared.path.name, metrics)
        logger.info(f"  Done: {prepared.path.name} -> {len(prepared.chunks)} chunks indexed")
        return {
            "status": "done",
            "file": prepared.path.name,
            "chunks": len(prepared.chunks),
            "metrics": metrics.to_dict(),
        }
    except Exception as e:
        logger.exception(f"Error finalizing {prepared.path.name}")
        pipeline.store.set_status(prepared.path, FileStatus.ERROR, error_msg=str(e))
        pipeline.embedder.delete_file_vectors(file_qdrant_ids)
        metrics.sqlite_s = perf_counter() - sqlite_start
        metrics.total_s = _total_metrics_s(metrics)
        return {
            "status": "error",
            "file": prepared.path.name,
            "error": str(e),
            "metrics": metrics.to_dict(),
        }


def safe_next_qdrant_id(pipeline) -> int:
    sqlite_max = pipeline.store.max_qdrant_id()
    max_point_id = getattr(pipeline.embedder, "max_point_id", None)
    qdrant_max = max_point_id() if callable(max_point_id) else -1
    return max(sqlite_max, qdrant_max) + 1


def _total_metrics_s(metrics: IngestMetrics) -> float:
    return metrics.parse_s + metrics.chunk_s + metrics.embed_s + metrics.qdrant_s + metrics.sqlite_s
