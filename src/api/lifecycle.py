"""Application lifespan management for DocFlow API."""

from __future__ import annotations

from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI

from src.api.model_tasks import ModelTaskTimeout
from src.api.runtime import get_api_runtime
from src.config import DocFlowSettings
from src.ingest.pipeline import IngestPipeline
from src.ingest.queue import IngestQueue
from src.ingest.store import DocStore
from src.ingest.watcher import FolderWatcher, _is_excluded
from src.query.engine import QueryEngine


def _api():
    return get_api_runtime()


def _warmup_models():
    """Warm up embedding, reranker, and local LLM models."""
    api = _api()
    try:
        em = api.query_engine.retriever.embed_model
        warmup_query = (
            "Instruct: Retrieve relevant text passages that answer the query.\nQuery: warmup"
        )
        em.encode([warmup_query], normalize_embeddings=True, convert_to_numpy=True)
        api.logger.info("[warmup] Embedding model ready")
    except Exception as e:
        api.logger.warning(f"[warmup] Embedding warmup failed (non-fatal): {e}")
    try:
        rr = api.query_engine.retriever.reranker
        rr.compute_score([["warmup query", "warmup passage for reranker initialization."]])
        api.logger.info("[warmup] MLX reranker ready")
    except Exception as e:
        api.logger.warning(f"[warmup] Reranker warmup failed (non-fatal): {e}")
    if api.query_engine.generator.backend == "mlx":
        try:
            api.query_engine.generator._load_mlx_model()
            api.logger.info(f"[warmup] MLX LLM ready: {api.query_engine.generator.mlx_model_name}")
        except Exception as e:
            api.logger.warning(f"[warmup] MLX LLM warmup failed (non-fatal): {e}")


@asynccontextmanager
async def lifespan(app: FastAPI):
    api = _api()
    settings = DocFlowSettings.from_file(api.CONFIG_PATH)

    backend = settings.llm.backend
    if backend == "mlx":
        llm_options = list(
            dict.fromkeys(
                [
                    settings.llm.mlx_model,
                    settings.llm.mlx_model_enhanced,
                ]
            )
        )
    elif backend == "claude":
        llm_options = list(
            dict.fromkeys(
                [
                    settings.llm.claude_model,
                ]
            )
        )
    else:
        llm_options = list(
            dict.fromkeys(
                [
                    settings.llm.ollama_model,
                    settings.llm.ollama_model_enhanced,
                ]
            )
        )
    api.llm_options = [m for m in llm_options if m]
    api.app_state.llm_options = api.llm_options

    api.watch_dirs = api._parse_watch_dirs(settings.raw, config_path=settings.config_path)
    api.app_state.watch_dirs = api.watch_dirs

    api.store = DocStore(settings.paths.db_path)
    api.app_state.store = api.store

    n_reset = api.store.reset_processing_files()
    if n_reset:
        api.logger.info(
            f"[startup] Reset {n_reset} interrupted file(s) from 'processing' -> 'error'"
        )

    api.pipeline = IngestPipeline.from_config(api.CONFIG_PATH, store=api.store)
    api.query_engine = QueryEngine.from_config(api.CONFIG_PATH, store=api.store)
    api.app_state.pipeline = api.pipeline
    api.app_state.query_engine = api.query_engine

    api.ingest_queue = IngestQueue(
        api.pipeline,
        on_done=None,
        ml_executor=None,
        parse_workers=settings.ingest.parse_workers,
        microbatch_max_files=settings.ingest.microbatch_max_files,
        microbatch_max_chunks=settings.ingest.microbatch_max_chunks,
        microbatch_linger_ms=settings.ingest.microbatch_linger_ms,
        should_pause_background=lambda: api.model_tasks.is_foreground_active(
            grace_s=api.FOREGROUND_PAUSE_GRACE_S
        ),
        pause_check_interval_ms=settings.ingest.pause_check_interval_ms,
    )
    api.ingest_queue.start()
    api.app_state.ingest_queue = api.ingest_queue

    api.watcher = FolderWatcher(api.pipeline, api.watch_dirs, ingest_queue=api.ingest_queue)
    api.watcher.start()
    api.app_state.watcher = api.watcher
    api._sync_app_state()

    api.logger.info("Warming up embedding and reranker models...")
    try:
        await api.model_tasks.run(
            "warmup", _warmup_models, timeout_s=max(api.MODEL_TASK_TIMEOUT_S, 180)
        )
    except ModelTaskTimeout as exc:
        api.logger.warning("[warmup] Timed out: %s", exc)
    api.logger.info("Models ready.")

    shared_embed = api.query_engine.retriever._embed_model
    if shared_embed is not None:
        api.pipeline.embedder._model = shared_embed
        api.pipeline.embedder._vector_dim = shared_embed.get_sentence_embedding_dimension()
        api.pipeline.embedder._ensure_collection(api.pipeline.embedder._vector_dim)
        api.logger.info("[embedder] Shared embedding model instance with ingest pipeline")

    try:
        filled = api.store.backfill_fts(api.query_engine.retriever._qdrant)
        if filled > 0:
            api.logger.info(f"[migration] FTS5 backfill: {filled} chunks indexed")
    except Exception as e:
        api.logger.warning(f"[migration] FTS5 backfill failed (non-fatal): {e}")

    removed = api.store.cleanup_deleted_files()
    if removed:
        qdrant = api.query_engine.retriever._qdrant
        for r in removed:
            if r["qdrant_ids"]:
                try:
                    qdrant.delete(
                        collection_name=settings.qdrant.collection,
                        points_selector=r["qdrant_ids"],
                    )
                except Exception as e:
                    api.logger.warning(
                        f"[cleanup] Failed to delete vectors for {r['file_name']}: {e}"
                    )
            api.logger.info(
                f"[cleanup] Removed deleted file: {r['file_name']} ({len(r['qdrant_ids'])} vectors)"
            )

    supported_exts = api.pipeline.registry.supported_extensions
    all_files: list[Path] = []
    for wd in api.watch_dirs:
        for ext in wd.extensions if wd.extensions else supported_exts:
            pattern = f"**/*{ext}" if wd.recursive else f"*{ext}"
            all_files.extend(f for f in wd.path.glob(pattern) if not _is_excluded(f))
    if all_files:
        api.ingest_queue.submit_many(all_files)

    yield

    if api.watcher:
        api.watcher.stop()
    if api.ingest_queue:
        api.ingest_queue.stop()
    if api.query_engine:
        api.query_engine.close()
    if api.pipeline:
        api.pipeline.close()
    if api.store:
        api.store.close()
    api.model_tasks.shutdown()
    api.app_state.clear_runtime()
