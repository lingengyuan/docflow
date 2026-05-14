"""Embedding vector preparation for ingest pipelines."""

from __future__ import annotations

from collections import Counter
from time import perf_counter

import numpy as np

from src.ingest.chunker import Chunk
from src.ingest.pipeline_types import ProgressCallback
from src.ingest.store import DocStore


def chunk_embedding_text(chunk: Chunk) -> str:
    return chunk.embedding_text or chunk.raw_text or chunk.text


def build_vectors(
    pipeline,
    chunks: list[Chunk],
    progress_callback: ProgressCallback | None = None,
) -> tuple[np.ndarray, list[str], set[str], float]:
    if not chunks:
        return np.empty((0, 0), dtype=np.float32), [], set(), 0.0

    embed_start = perf_counter()
    embedding_texts = [chunk_embedding_text(chunk) for chunk in chunks]
    text_hashes = [DocStore.compute_text_hash(text) for text in embedding_texts]
    hash_counts = Counter(text_hashes)
    cached_vectors = (
        pipeline.store.get_cached_embeddings(pipeline.embedder.embedding_cache_key, text_hashes)
        if pipeline.use_embedding_cache
        else {}
    )
    cached_hashes = set(cached_vectors.keys())
    cache_hits = sum(hash_counts[text_hash] for text_hash in cached_hashes)

    missing_hashes: list[str] = []
    missing_texts: list[str] = []
    seen_missing: set[str] = set()
    for text_hash, text in zip(text_hashes, embedding_texts, strict=False):
        if text_hash in cached_hashes or text_hash in seen_missing:
            continue
        seen_missing.add(text_hash)
        missing_hashes.append(text_hash)
        missing_texts.append(text)

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
        encoded_vectors = pipeline.embedder.encode_texts(
            missing_texts,
            progress_callback=_on_encode,
        )
        new_vectors = {
            missing_hashes[i]: np.asarray(encoded_vectors[i], dtype=np.float32)
            for i in range(len(missing_hashes))
        }
        vectors_by_hash.update(new_vectors)
        if pipeline.use_embedding_cache:
            pipeline.store.put_cached_embeddings(pipeline.embedder.embedding_cache_key, new_vectors)
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

    vectors = np.stack([vectors_by_hash[text_hash] for text_hash in text_hashes]).astype(
        np.float32
    )
    embed_s = perf_counter() - embed_start
    return vectors, text_hashes, cached_hashes, embed_s
