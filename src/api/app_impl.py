"""
DocFlow FastAPI 后端。
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import queue
import shutil
import sys
import threading
import types
from contextlib import asynccontextmanager
from pathlib import Path
from time import perf_counter

import yaml
from fastapi import FastAPI, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, Response, StreamingResponse
from fastapi.staticfiles import StaticFiles

from src.api.health_checks import (
    _aggregate_health_status,
    _check_models,
    _check_ollama,
    _check_qdrant,
    _check_sqlite,
    _health_actions,
    _health_capabilities,
    _health_groups,
    _timed_check,
    configure_health_checks,
)
from src.api.model_tasks import ModelTaskController, ModelTaskTimeout
from src.api.routes import imports as imports_routes
from src.api.routes import knowledge as knowledge_routes
from src.api.routes import library as library_routes
from src.api.routes import maintenance as maintenance_routes
from src.api.routes import obsidian as obsidian_routes
from src.api.routes import query as query_routes
from src.api.routes import settings as settings_routes
from src.api.schemas import (
    AnswerFeedbackRequest,
    AnswerNoteRequest,
    BatchFavoriteRequest,
    BatchMetadataRequest,
    BatchRebuildRequest,
    ConversationCreateRequest,
    DebugRetrieveRequest,
    FileMetadataRequest,
    KnowledgeOutputRequest,
    LLMSwitchRequest,
    NoteCreateRequest,
    ObsidianRelatedRequest,
    QueryOptions,
    QueryRequest,
    QueryResponse,
    ResearchRequest,
    ResearchResponse,
    SummarizeRequest,
    WebImportRequest,
)
from src.api.services.health_service import HealthService
from src.api.services.import_service import ImportService
from src.api.services.knowledge_service import KnowledgeService
from src.api.services.query_service import QueryService
from src.api.state import AppState
from src.domain_types import FileStatus
from src.ingest.imports import (
    build_answer_note_markdown,
    build_knowledge_output_markdown,
    build_quick_note_markdown,
    fetch_webpage_markdown,
)
from src.ingest.pipeline import IngestPipeline
from src.ingest.queue import IngestQueue
from src.ingest.store import DocStore
from src.ingest.watcher import FolderWatcher, WatchDir, _is_excluded
from src.knowledge_outputs import (
    get_knowledge_output_type,
    knowledge_output_tags,
)
from src.maintenance.startup import ensure_config_file
from src.model_cache import (
    assert_model_download_allowed,
    configured_model_names,
    hf_cache_dir,
    hf_model_cache_path,
    is_hf_model_cached,
    is_remote_model_reference,
)
from src.query.engine import QueryEngine

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

logging.getLogger("transformers.tokenization_utils_base").setLevel(logging.ERROR)
logging.getLogger("FlagEmbedding").setLevel(logging.WARNING)

CONFIG_PATH = ensure_config_file(Path(__file__).parent.parent.parent / "config.yaml")
COLLECTION_NAME = "docflow"
MODEL_TASK_TIMEOUT_S = float(os.getenv("DOCFLOW_MODEL_TASK_TIMEOUT_S", "90"))
STREAM_FIRST_CONTENT_TIMEOUT_S = float(os.getenv("DOCFLOW_STREAM_FIRST_CONTENT_TIMEOUT_S", "60"))
STREAM_IDLE_TIMEOUT_S = float(os.getenv("DOCFLOW_STREAM_IDLE_TIMEOUT_S", "60"))
STREAM_QUEUE_POLL_S = 0.25
FOREGROUND_PAUSE_GRACE_S = float(os.getenv("DOCFLOW_FOREGROUND_PAUSE_GRACE_S", "5"))
INGEST_PAUSE_CHECK_INTERVAL_MS = int(os.getenv("DOCFLOW_INGEST_PAUSE_CHECK_INTERVAL_MS", "500"))
MODEL_TIMEOUT_MESSAGE = "模型任务超时，请稍后重试；系统已释放后续请求。"


# ---------------------------------------------------------------------------
# Global state (initialized in lifespan)
# ---------------------------------------------------------------------------

pipeline: IngestPipeline | None = None
ingest_queue: IngestQueue | None = None
query_engine: QueryEngine | None = None
store: DocStore | None = None
watcher: FolderWatcher | None = None
watch_dirs: list[WatchDir] = []
llm_options: list[str] = []
llm_switch_state: dict = {
    "state": "idle",
    "model": None,
    "message": "",
    "started_at": None,
    "finished_at": None,
}

model_tasks = ModelTaskController(thread_name_prefix="ml-inference", logger=logger)
app_state = AppState(config_path=CONFIG_PATH, model_tasks=model_tasks)
llm_switch_state = app_state.llm_switch_state
query_service = QueryService()
import_service = ImportService()
knowledge_service = KnowledgeService()
health_service = HealthService()
configure_health_checks(store_getter=lambda: store)


def _sync_app_state() -> None:
    app_state.pipeline = pipeline
    app_state.ingest_queue = ingest_queue
    app_state.query_engine = query_engine
    app_state.store = store
    app_state.watcher = watcher
    app_state.watch_dirs = watch_dirs
    app_state.llm_options = llm_options
    app_state.model_tasks = model_tasks




def _parse_watch_dirs(cfg: dict) -> list[WatchDir]:
    """解析 config.yaml 中的 watch_dirs 配置（列表或兼容旧版单字符串）。"""
    paths_cfg = cfg.get("paths", {})
    raw = paths_cfg.get("watch_dirs", paths_cfg.get("watch_dir"))
    if raw is None:
        raw = "~/Documents/DocFlow"

    if isinstance(raw, str):
        # 兼容旧版单目录配置
        return [WatchDir(path=Path(raw).expanduser(), recursive=False)]

    result: list[WatchDir] = []
    for entry in raw:
        if isinstance(entry, str):
            result.append(WatchDir(path=Path(entry).expanduser(), recursive=False))
        else:
            result.append(
                WatchDir(
                    path=Path(entry["path"]).expanduser(),
                    recursive=entry.get("recursive", False),
                    extensions=entry.get("extensions", []),
                )
            )
    return result


def _configured_model_names(cfg: dict) -> dict[str, str]:
    return configured_model_names(cfg)


def _hf_cache_dir() -> Path:
    return hf_cache_dir()


def _safe_path_size(path: Path, *, max_entries: int = 100_000) -> int:
    return health_service.safe_path_size(path, max_entries=max_entries)


def _unique_existing_paths(paths: list[Path]) -> list[Path]:
    result: list[Path] = []
    seen: set[str] = set()
    for path in paths:
        expanded = path.expanduser()
        try:
            key = str(expanded.resolve())
        except OSError:
            key = str(expanded)
        if key in seen or not expanded.exists():
            continue
        seen.add(key)
        result.append(expanded)
    return result


def _config_path(raw: str | Path) -> Path:
    path = Path(raw).expanduser()
    if path.is_absolute():
        return path
    return CONFIG_PATH.parent / path


def _configured_model_cache_paths(cfg: dict) -> list[Path]:
    paths: list[Path] = []
    model_names = {name for name in _configured_model_names(cfg).values() if name}
    for model_name in model_names:
        if is_remote_model_reference(model_name):
            paths.append(hf_model_cache_path(model_name))

    onnx_cache_dir = cfg.get("embedding", {}).get("onnx_cache_dir")
    if onnx_cache_dir:
        paths.append(_config_path(onnx_cache_dir))

    if any("/" not in name for name in model_names):
        paths.append(Path.home() / ".ollama" / "models")

    return _unique_existing_paths(paths)


def _source_file_usage(files: list[dict]) -> dict:
    return health_service.source_file_usage(files)


def _app_data_paths(cfg: dict) -> list[Path]:
    paths_cfg = cfg.get("paths", {})
    candidates = [
        _config_path(paths_cfg.get("db_path", "docflow.db")),
        _config_path(paths_cfg.get("id_counter", "qdrant_id_counter.txt")),
        _config_path("qdrant_storage"),
    ]
    db_path = _config_path(paths_cfg.get("db_path", "docflow.db"))
    candidates.extend(
        [
            Path(f"{db_path}-wal"),
            Path(f"{db_path}-shm"),
        ]
    )
    return _unique_existing_paths(candidates)


def _collect_storage_usage(cfg: dict, doc_store: DocStore) -> dict:
    return health_service.collect_storage_usage(
        cfg,
        doc_store,
        configured_model_cache_paths=_configured_model_cache_paths,
        app_data_paths=_app_data_paths,
        disk_usage=shutil.disk_usage,
    )


def _is_hf_model_cached(model_name: str) -> bool:
    return is_hf_model_cached(model_name)


def _llm_model_status(model_name: str) -> dict:
    cached = _is_hf_model_cached(model_name) if is_remote_model_reference(model_name) else True
    return {
        "model": model_name,
        "cached": cached,
        "available": cached,
        "current": bool(query_engine and query_engine.generator.current_model == model_name),
        "detail": "本地缓存可用。" if cached else "本地缓存缺失，切换前需要先准备模型。",
        "actions": [] if cached else [f"联网后准备模型缓存：{model_name}"],
    }


def _set_llm_switch_state(state: str, *, model: str | None = None, message: str = ""):
    llm_switch_state.set(state, model=model, message=message)


def _load_mlx_model_candidate(model_name: str):
    from mlx_lm import load

    with open(CONFIG_PATH) as f:
        cfg = yaml.safe_load(f) or {}
    allow_model_download = bool(cfg.get("privacy", {}).get("allow_model_download", False))
    assert_model_download_allowed(
        model_name,
        allow_model_download,
        purpose="answer",
    )
    return load(model_name)


@asynccontextmanager
async def lifespan(app: FastAPI):
    global pipeline, ingest_queue, query_engine, store, watcher, watch_dirs, llm_options
    with open(CONFIG_PATH) as f:
        cfg = yaml.safe_load(f)

    llm_cfg = cfg.get("llm", {})
    ingest_cfg = cfg.get("ingest", {})
    backend = llm_cfg.get("backend", "local")
    if backend == "mlx":
        llm_options = list(
            dict.fromkeys(
                [
                    llm_cfg.get("mlx_model", ""),
                    llm_cfg.get("mlx_model_enhanced", ""),
                ]
            )
        )
    elif backend == "claude":
        llm_options = list(
            dict.fromkeys(
                [
                    llm_cfg.get("claude_model", ""),
                    llm_cfg.get("claude_model_enhanced", ""),
                ]
            )
        )
    else:
        llm_options = list(
            dict.fromkeys(
                [
                    llm_cfg.get("ollama_model", cfg["ollama"]["llm_model"]),
                    llm_cfg.get(
                        "ollama_model_enhanced", cfg["ollama"].get("llm_model_enhanced", "")
                    ),
                ]
            )
        )
    llm_options = [m for m in llm_options if m]
    app_state.llm_options = llm_options

    db_path = Path(cfg["paths"]["db_path"]).expanduser()
    watch_dirs = _parse_watch_dirs(cfg)
    app_state.watch_dirs = watch_dirs

    store = DocStore(db_path)
    app_state.store = store

    # 清理上次崩溃遗留的 processing 状态，确保启动扫描能重新处理这些文件
    n_reset = store.reset_processing_files()
    if n_reset:
        logger.info(f"[startup] Reset {n_reset} interrupted file(s) from 'processing' → 'error'")

    pipeline = IngestPipeline.from_config(CONFIG_PATH, store=store)
    query_engine = QueryEngine.from_config(CONFIG_PATH, store=store)
    app_state.pipeline = pipeline
    app_state.query_engine = query_engine

    # CPU embedding 不需要走前台模型任务控制器，直接在 ingest worker 线程里跑
    # 前台模型任务控制器只保留给 MLX 推理（reranker、LLM）
    ingest_queue = IngestQueue(
        pipeline,
        on_done=None,
        ml_executor=None,
        parse_workers=ingest_cfg.get("parse_workers", 2),
        microbatch_max_files=ingest_cfg.get("microbatch_max_files", 8),
        microbatch_max_chunks=ingest_cfg.get("microbatch_max_chunks", 128),
        microbatch_linger_ms=ingest_cfg.get("microbatch_linger_ms", 75),
        should_pause_background=lambda: model_tasks.is_foreground_active(
            grace_s=FOREGROUND_PAUSE_GRACE_S
        ),
        pause_check_interval_ms=ingest_cfg.get(
            "pause_check_interval_ms", INGEST_PAUSE_CHECK_INTERVAL_MS
        ),
    )
    ingest_queue.start()
    app_state.ingest_queue = ingest_queue

    watcher = FolderWatcher(pipeline, watch_dirs, ingest_queue=ingest_queue)
    watcher.start()
    app_state.watcher = watcher
    _sync_app_state()

    logger.info("Warming up embedding and reranker models...")
    try:
        await model_tasks.run("warmup", _warmup_models, timeout_s=max(MODEL_TASK_TIMEOUT_S, 180))
    except ModelTaskTimeout as exc:
        logger.warning("[warmup] Timed out: %s", exc)
    logger.info("Models ready.")

    # 共享 embedding model 实例
    shared_embed = query_engine.retriever._embed_model
    if shared_embed is not None:
        pipeline.embedder._model = shared_embed
        pipeline.embedder._vector_dim = shared_embed.get_sentence_embedding_dimension()
        pipeline.embedder._ensure_collection(pipeline.embedder._vector_dim)
        logger.info("[embedder] Shared embedding model instance with ingest pipeline")

    # FTS5 backfill migration: 若旧 DB 有 chunks 但无 FTS5 记录，从 Qdrant 回填
    try:
        filled = store.backfill_fts(query_engine.retriever._qdrant)
        if filled > 0:
            logger.info(f"[migration] FTS5 backfill: {filled} chunks indexed")
    except Exception as e:
        logger.warning(f"[migration] FTS5 backfill failed (non-fatal): {e}")

    # 清理磁盘上已删除的文件（DB + Qdrant 向量）
    removed = store.cleanup_deleted_files()
    if removed:
        qdrant = query_engine.retriever._qdrant
        for r in removed:
            if r["qdrant_ids"]:
                try:
                    qdrant.delete(
                        collection_name=cfg["qdrant"].get("collection", "docflow"),
                        points_selector=r["qdrant_ids"],
                    )
                except Exception as e:
                    logger.warning(f"[cleanup] Failed to delete vectors for {r['file_name']}: {e}")
            logger.info(
                f"[cleanup] Removed deleted file: {r['file_name']} ({len(r['qdrant_ids'])} vectors)"
            )

    # Background scan: enqueue existing files (skip .obsidian/.trash/.git)
    supported_exts = pipeline.registry.supported_extensions
    all_files: list[Path] = []
    for wd in watch_dirs:
        for ext in wd.extensions if wd.extensions else supported_exts:
            pattern = f"**/*{ext}" if wd.recursive else f"*{ext}"
            all_files.extend(f for f in wd.path.glob(pattern) if not _is_excluded(f))
    if all_files:
        ingest_queue.submit_many(all_files)

    yield

    if watcher:
        watcher.stop()
    if ingest_queue:
        ingest_queue.stop()
    if query_engine:
        query_engine.close()
    if pipeline:
        pipeline.close()
    if store:
        store.close()
    model_tasks.shutdown()
    app_state.clear_runtime()


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------

app = FastAPI(title="DocFlow", lifespan=lifespan)


# ---------------------------------------------------------------------------
# API handlers
# ---------------------------------------------------------------------------


def _resolve_query_options(req) -> QueryOptions:
    scope_mode = str(getattr(req, "scope_mode", None) or "all").strip().lower().replace("-", "_")
    retrieval_mode = _normalize_retrieval_mode(getattr(req, "retrieval_mode", None))
    if scope_mode == "full_text":
        retrieval_mode = "full_text"

    legacy_filter = _clean_file_filter(getattr(req, "file_filter", None))
    if legacy_filter:
        return QueryOptions(
            file_filter=legacy_filter,
            retrieval_mode=retrieval_mode,
            scope={
                "mode": "file_filter",
                "label": "指定文件",
                "files": legacy_filter,
                "retrieval_mode": retrieval_mode,
            },
        )

    if scope_mode in {"all", "full_text"}:
        label = "全文模式" if scope_mode == "full_text" else "全部知识库"
        return QueryOptions(
            file_filter=[],
            retrieval_mode=retrieval_mode,
            scope={"mode": scope_mode, "label": label, "retrieval_mode": retrieval_mode},
        )

    if scope_mode == "collection":
        if store is None:
            raise HTTPException(503, "File store not ready")
        collection = str(getattr(req, "collection", "") or "").strip()
        if not collection:
            raise HTTPException(400, "Collection is required for collection scope")
        files = store.list_files(status=FileStatus.DONE, collection=collection)
        file_names = _unique_file_names(files)
        if not file_names:
            raise HTTPException(404, f"No indexed files found in collection: {collection}")
        return QueryOptions(
            file_filter=file_names,
            retrieval_mode=retrieval_mode,
            scope={
                "mode": "collection",
                "label": collection,
                "collection": collection,
                "file_count": len(file_names),
                "retrieval_mode": retrieval_mode,
            },
        )

    if scope_mode == "file":
        file_name = str(getattr(req, "file_name", "") or "").strip()
        file_id = getattr(req, "file_id", None)
        if file_id is not None:
            if store is None:
                raise HTTPException(503, "File store not ready")
            record = store.get_file_by_id(int(file_id))
            if record is None or record.get("status") != "done":
                raise HTTPException(404, "Indexed file not found")
            file_name = str(record.get("file_name") or "").strip()
        if not file_name:
            raise HTTPException(400, "File is required for file scope")
        return QueryOptions(
            file_filter=[file_name],
            retrieval_mode=retrieval_mode,
            scope={
                "mode": "file",
                "label": file_name,
                "file_id": file_id,
                "file_name": file_name,
                "retrieval_mode": retrieval_mode,
            },
        )

    raise HTTPException(400, f"Unsupported query scope: {scope_mode}")


def _clean_file_filter(values: list[str] | None) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for value in values or []:
        file_name = str(value or "").strip()
        if not file_name or file_name in seen:
            continue
        seen.add(file_name)
        result.append(file_name)
    return result


def _unique_file_names(files: list[dict]) -> list[str]:
    return _clean_file_filter([str(item.get("file_name") or "") for item in files])


def _normalize_retrieval_mode(mode: str | None) -> str:
    normalized = str(mode or "hybrid").strip().lower().replace("-", "_")
    if normalized in {"fts", "fulltext", "full_text"}:
        return "full_text"
    return "hybrid"


async def query(req: QueryRequest):
    if query_engine is None:
        raise HTTPException(503, "Query engine not ready")
    options = _resolve_query_options(req)
    conversation_id = _resolve_conversation_id(req.conversation_id)
    conversation_context = _conversation_context(conversation_id)
    retrieval_query = _build_retrieval_query(req.question, conversation_context)
    file_filter_json = json.dumps(options.file_filter, ensure_ascii=False)
    if store is not None and conversation_id is not None:
        store.add_message(
            conversation_id=conversation_id,
            role="user",
            content=req.question,
            file_filter_json=file_filter_json,
        )
    try:
        result = await model_tasks.run(
            "query",
            lambda: query_engine.query(
                req.question,
                file_filter=options.file_filter or None,
                retrieval_mode=options.retrieval_mode,
                conversation_context=conversation_context,
                retrieval_query=retrieval_query,
            ),
            timeout_s=MODEL_TASK_TIMEOUT_S,
        )
    except ModelTaskTimeout as exc:
        logger.warning("[api/query] timeout id=%s question=%r", exc.task_id, req.question[:80])
        raise HTTPException(504, MODEL_TIMEOUT_MESSAGE) from exc
    citations_data = query_service.response_citations(result.citations)
    history_id = None
    if store is not None:
        history_id = store.add_history(
            question=req.question,
            answer=result.text,
            citations_json=json.dumps(citations_data, ensure_ascii=False),
            file_filter_json=file_filter_json,
            conversation_id=conversation_id or 0,
        )
        if conversation_id is not None:
            store.add_message(
                conversation_id=conversation_id,
                role="assistant",
                content=result.text,
                citations_json=json.dumps(citations_data, ensure_ascii=False),
                file_filter_json=file_filter_json,
            )
    return QueryResponse(
        answer=result.text,
        citations=citations_data,
        related_notes=getattr(result, "related_notes", []),
        history_id=history_id,
        conversation_id=conversation_id,
        scope=options.scope,
        reproducible=getattr(result, "reproducible", True),
    )


async def research(req: ResearchRequest):
    if query_engine is None:
        raise HTTPException(503, "Query engine not ready")
    options = _resolve_query_options(req)
    conversation_id = _resolve_conversation_id(req.conversation_id)
    conversation_context = _conversation_context(conversation_id)
    file_filter_json = json.dumps(options.file_filter, ensure_ascii=False)
    if store is not None and conversation_id is not None:
        store.add_message(
            conversation_id=conversation_id,
            role="user",
            content=req.question,
            file_filter_json=file_filter_json,
        )
    try:
        result = await model_tasks.run(
            "research",
            lambda: query_engine.deep_research(
                req.question,
                file_filter=options.file_filter or None,
                retrieval_mode=options.retrieval_mode,
                max_steps=req.max_steps,
                conversation_context=conversation_context,
            ),
            timeout_s=max(MODEL_TASK_TIMEOUT_S, 180),
        )
    except ModelTaskTimeout as exc:
        logger.warning("[api/research] timeout id=%s question=%r", exc.task_id, req.question[:80])
        raise HTTPException(504, MODEL_TIMEOUT_MESSAGE) from exc
    citations_data = query_service.response_citations(result.citations)
    history_id = None
    if store is not None:
        history_id = store.add_history(
            question=req.question,
            answer=result.text,
            citations_json=json.dumps(citations_data, ensure_ascii=False),
            file_filter_json=file_filter_json,
            conversation_id=conversation_id or 0,
        )
        if conversation_id is not None:
            store.add_message(
                conversation_id=conversation_id,
                role="assistant",
                content=result.text,
                citations_json=json.dumps(citations_data, ensure_ascii=False),
                file_filter_json=file_filter_json,
            )
    return ResearchResponse(
        answer=result.text,
        citations=citations_data,
        related_notes=getattr(result, "related_notes", []),
        research_steps=getattr(result, "research_steps", []),
        history_id=history_id,
        conversation_id=conversation_id,
        scope=options.scope,
        reproducible=getattr(result, "reproducible", True),
    )


async def query_stream(req: QueryRequest, request: Request):
    """SSE 流式查询：先返回 citations，再逐 token 返回答案。"""
    if query_engine is None:
        raise HTTPException(503, "Query engine not ready")

    options = _resolve_query_options(req)
    conversation_id = _resolve_conversation_id(req.conversation_id)
    conversation_context = _conversation_context(conversation_id)
    retrieval_query = _build_retrieval_query(req.question, conversation_context)
    file_filter_json = json.dumps(options.file_filter, ensure_ascii=False)
    if store is not None and conversation_id is not None:
        store.add_message(
            conversation_id=conversation_id,
            role="user",
            content=req.question,
            file_filter_json=file_filter_json,
        )

    loop = asyncio.get_event_loop()
    q: queue.Queue = queue.Queue()
    cancel = threading.Event()

    def _run():
        try:
            if conversation_id is not None:
                q.put(("conversation", {"conversation_id": conversation_id}))
            stream_result = query_engine.query_stream(
                req.question,
                file_filter=options.file_filter or None,
                retrieval_mode=options.retrieval_mode,
                cancel_event=cancel,
                conversation_context=conversation_context,
                retrieval_query=retrieval_query,
                include_related=True,
            )
            if len(stream_result) == 3:
                chunks, token_gen, related_notes = stream_result
            else:
                chunks, token_gen = stream_result
                related_notes = []
            if cancel.is_set():
                return
            citations_data = query_service.stream_citations(chunks)
            if cancel.is_set():
                return
            q.put(("citations", citations_data))
            q.put(("related_notes", related_notes))
            full_answer = []
            for token in token_gen:
                if cancel.is_set():
                    return
                full_answer.append(token)
                q.put(("token", token))
            answer_text = "".join(full_answer).strip()
            history_id = None
            if not cancel.is_set() and store is not None:
                history_id = store.add_history(
                    question=req.question,
                    answer=answer_text,
                    citations_json=json.dumps(citations_data, ensure_ascii=False),
                    file_filter_json=file_filter_json,
                    conversation_id=conversation_id or 0,
                )
                if conversation_id is not None:
                    store.add_message(
                        conversation_id=conversation_id,
                        role="assistant",
                        content=answer_text,
                        citations_json=json.dumps(citations_data, ensure_ascii=False),
                        file_filter_json=file_filter_json,
                    )
            if not cancel.is_set():
                q.put(
                    (
                        "done",
                        {
                            "history_id": history_id,
                            "conversation_id": conversation_id,
                        },
                    )
                )
        except Exception as e:
            if not cancel.is_set():
                q.put(("error", str(e)))

    task = model_tasks.submit("query_stream", _run)

    async def event_stream():
        first_content_at: float | None = None
        last_content_at = perf_counter()
        while True:
            if await request.is_disconnected():
                cancel.set()
                model_tasks.cancel_and_retire(task, reason="client disconnected")
                break
            try:
                event, data = await loop.run_in_executor(None, q.get, True, STREAM_QUEUE_POLL_S)
            except queue.Empty:
                now = perf_counter()
                if (
                    first_content_at is None
                    and now - task.started_at > STREAM_FIRST_CONTENT_TIMEOUT_S
                ):
                    cancel.set()
                    reason = (
                        f"stream first content timeout after {STREAM_FIRST_CONTENT_TIMEOUT_S:.1f}s"
                    )
                    model_tasks.cancel_and_retire(
                        task,
                        reason=reason,
                    )
                    yield (
                        "event: error\n"
                        f"data: {json.dumps(MODEL_TIMEOUT_MESSAGE, ensure_ascii=False)}\n\n"
                    )
                    break
                if first_content_at is not None and now - last_content_at > STREAM_IDLE_TIMEOUT_S:
                    cancel.set()
                    model_tasks.cancel_and_retire(
                        task,
                        reason=f"stream idle timeout after {STREAM_IDLE_TIMEOUT_S:.1f}s",
                    )
                    yield (
                        "event: error\n"
                        f"data: {json.dumps(MODEL_TIMEOUT_MESSAGE, ensure_ascii=False)}\n\n"
                    )
                    break
                if task.future.done() and q.empty():
                    break
                continue
            if event in ("citations", "related_notes", "token", "done", "error"):
                if first_content_at is None:
                    first_content_at = perf_counter()
                last_content_at = perf_counter()
            yield f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"
            if event in ("done", "error"):
                break

    return StreamingResponse(event_stream(), media_type="text/event-stream")


async def answer_feedback(req: AnswerFeedbackRequest):
    if store is None:
        raise HTTPException(503, "Store not ready")
    try:
        feedback = store.set_answer_feedback(req.history_id, req.rating, req.note or "")
    except KeyError as exc:
        raise HTTPException(404, "Answer history item not found") from exc
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc
    return {"ok": True, "feedback": feedback, "summary": store.get_feedback_summary()}


def _resolve_conversation_id(conversation_id: int | None) -> int | None:
    if store is None:
        return None
    if conversation_id is None:
        return store.create_conversation()
    if store.get_conversation(conversation_id) is None:
        raise HTTPException(404, "Conversation not found")
    return conversation_id


def _conversation_context(conversation_id: int | None, limit: int = 6) -> list[dict]:
    if store is None or conversation_id is None:
        return []
    return store.list_messages(conversation_id, limit=limit)


def _build_retrieval_query(question: str, conversation_context: list[dict]) -> str:
    return query_service.build_retrieval_query(question, conversation_context)


def _looks_like_followup(question: str) -> bool:
    return query_service.looks_like_followup(question)


async def list_conversations(limit: int = 50):
    if store is None:
        raise HTTPException(503, "Store not ready")
    return store.list_conversations(limit=limit)


async def create_conversation(req: ConversationCreateRequest):
    if store is None:
        raise HTTPException(503, "Store not ready")
    conversation_id = store.create_conversation(req.title)
    return store.get_conversation(conversation_id)


async def list_conversation_messages(conversation_id: int, limit: int = 100):
    if store is None:
        raise HTTPException(503, "Store not ready")
    if store.get_conversation(conversation_id) is None:
        raise HTTPException(404, "Conversation not found")
    items = store.list_messages(conversation_id, limit=limit)
    return query_service.decode_history_items(items)


async def delete_conversation(conversation_id: int):
    if store is None:
        raise HTTPException(503, "Store not ready")
    if not store.delete_conversation(conversation_id):
        raise HTTPException(404, "Conversation not found")
    return {"ok": True}


async def trigger_ingest():
    """手动触发全量扫描所有监控目录（异步，立即返回）。"""
    if ingest_queue is None or not watch_dirs:
        raise HTTPException(503, "Pipeline not ready")
    supported_exts = pipeline.registry.supported_extensions
    all_files: list[Path] = []
    for wd in watch_dirs:
        for ext in wd.extensions if wd.extensions else supported_exts:
            pattern = f"**/*{ext}" if wd.recursive else f"*{ext}"
            all_files.extend(f for f in wd.path.glob(pattern) if not _is_excluded(f))
    result = ingest_queue.submit_many(all_files)
    return {**result, "files": [p.name for p in all_files]}


async def queue_status():
    if ingest_queue is None:
        return {
            "queue_size": 0,
            "processing": None,
            "processing_files": [],
            "pending_files": [],
            "progress": None,
            "last_completed": None,
            "paused": False,
            "pause_reason": None,
            "paused_since": None,
            "foreground": model_tasks.status(),
        }
    return {**ingest_queue.status(), "foreground": model_tasks.status()}


def _warmup_models():
    """预热 embedding + reranker + LLM 模型。"""
    try:
        em = query_engine.retriever.embed_model
        warmup_query = (
            "Instruct: Retrieve relevant text passages that answer the query.\nQuery: warmup"
        )
        em.encode([warmup_query], normalize_embeddings=True, convert_to_numpy=True)
        logger.info("[warmup] Embedding model ready")
    except Exception as e:
        logger.warning(f"[warmup] Embedding warmup failed (non-fatal): {e}")
    try:
        rr = query_engine.retriever.reranker
        rr.compute_score([["warmup query", "warmup passage for reranker initialization."]])
        logger.info("[warmup] MLX reranker ready")
    except Exception as e:
        logger.warning(f"[warmup] Reranker warmup failed (non-fatal): {e}")
    if query_engine.generator.backend == "mlx":
        try:
            query_engine.generator._load_mlx_model()
            logger.info(f"[warmup] MLX LLM ready: {query_engine.generator.mlx_model_name}")
        except Exception as e:
            logger.warning(f"[warmup] MLX LLM warmup failed (non-fatal): {e}")


async def list_files(
    status: str | None = None,
    collection: str | None = None,
    tag: str | None = None,
    favorite: bool | None = None,
    kind: str | None = None,
    recent: bool | None = None,
):
    if store is None:
        raise HTTPException(503, "Store not ready")
    return store.list_files(
        status=status,
        collection=collection,
        tag=tag,
        favorite=favorite,
        kind=kind,
        recent=recent,
    )


async def library_meta():
    if store is None:
        raise HTTPException(503, "Store not ready")
    return store.list_library_facets()


async def storage_usage():
    if store is None:
        raise HTTPException(503, "Store not ready")
    with open(CONFIG_PATH) as f:
        cfg = yaml.safe_load(f)
    return _collect_storage_usage(cfg, store)


async def knowledge_overview(file_id: int | None = None):
    if store is None:
        raise HTTPException(503, "Store not ready")
    return knowledge_service.overview(store, active_file_id=file_id)


async def update_file_metadata(file_id: int, req: FileMetadataRequest):
    if store is None:
        raise HTTPException(503, "Store not ready")
    record = store.update_file_metadata(
        file_id,
        collection=req.collection,
        user_tags=req.user_tags,
    )
    if record is None:
        raise HTTPException(404, "File not found")
    return record


async def batch_favorite(req: BatchFavoriteRequest):
    if store is None:
        raise HTTPException(503, "Store not ready")
    if not req.file_ids:
        raise HTTPException(400, "No file IDs provided")
    changed = store.set_favorites(req.file_ids, favorited=req.favorited)
    return {"file_ids": changed, "favorited": req.favorited, "count": len(changed)}


async def batch_update_file_metadata(req: BatchMetadataRequest):
    if store is None:
        raise HTTPException(503, "Store not ready")
    if not req.file_ids:
        raise HTTPException(400, "No file IDs provided")
    records = store.update_files_metadata(
        req.file_ids,
        collection=req.collection,
        user_tags=req.user_tags,
    )
    return {"files": records, "count": len(records)}


async def batch_rebuild_files(req: BatchRebuildRequest):
    if store is None or ingest_queue is None:
        raise HTTPException(503, "Not ready")
    if not req.file_ids:
        raise HTTPException(400, "No file IDs provided")

    paths: list[Path] = []
    missing_ids: list[int] = []
    for file_id in dict.fromkeys(req.file_ids):
        record = store.get_file_by_id(file_id)
        if record is None:
            missing_ids.append(file_id)
            continue
        path = Path(record["file_path"])
        if path.exists():
            paths.append(path)
        else:
            missing_ids.append(file_id)

    if not paths:
        raise HTTPException(404, "No existing files found")
    result = ingest_queue.submit_many(paths)
    return {
        **result,
        "requested": len(req.file_ids),
        "files": [path.name for path in paths],
        "missing_ids": missing_ids,
    }


async def import_url(req: WebImportRequest):
    if store is None or ingest_queue is None:
        raise HTTPException(503, "Not ready")
    try:
        item = fetch_webpage_markdown(req.url, title=req.title)
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc
    except Exception as exc:
        raise HTTPException(502, f"Failed to fetch URL: {exc}") from exc
    return _write_import_and_enqueue(
        "web",
        item,
        collection=req.collection or "Web Imports",
        user_tags=req.user_tags or ["web"],
    )


async def create_note(req: NoteCreateRequest):
    if store is None or ingest_queue is None:
        raise HTTPException(503, "Not ready")
    try:
        item = build_quick_note_markdown(req.title, req.content, tags=req.user_tags or ["note"])
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc
    return _write_import_and_enqueue(
        "note",
        item,
        collection=req.collection or "Notes",
        user_tags=req.user_tags or ["note"],
    )


async def save_answer_note(req: AnswerNoteRequest):
    if store is None or ingest_queue is None:
        raise HTTPException(503, "Not ready")
    try:
        item = build_answer_note_markdown(
            req.title,
            req.answer,
            question=req.question,
            citations=req.citations or [],
            tags=req.user_tags or ["answer"],
        )
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc
    result = _write_import_and_enqueue(
        "answer",
        item,
        collection=req.collection or "Saved Answers",
        user_tags=req.user_tags or ["answer"],
    )
    note_file_id = int((result.get("file") or {}).get("id") or 0)
    source_file_ids = _source_file_ids_from_citations(req.citations or [])
    result["source_links"] = (
        store.replace_note_source_links(note_file_id, source_file_ids)
        if note_file_id and source_file_ids
        else []
    )
    return result


def _source_file_ids_from_citations(citations: list[dict]) -> list[int]:
    if store is None or not citations:
        return []
    files = store.list_files()
    by_path = {str(file.get("file_path") or ""): int(file["id"]) for file in files}
    by_name: dict[str, int] = {}
    for file in files:
        file_name = str(file.get("file_name") or "")
        if file_name and file_name not in by_name:
            by_name[file_name] = int(file["id"])
    source_ids = []
    for citation in citations:
        file_path = str(citation.get("file_path") or "")
        file_name = str(citation.get("file_name") or "")
        file_id = by_path.get(file_path) or by_name.get(file_name)
        if file_id:
            source_ids.append(file_id)
    return list(dict.fromkeys(source_ids))


async def create_knowledge_output(req: KnowledgeOutputRequest):
    if store is None or ingest_queue is None or query_engine is None:
        raise HTTPException(503, "Not ready")
    try:
        output = get_knowledge_output_type(req.output_type)
        source_text, source_files = _build_knowledge_output_source(req)
        title = req.title or output.label
        generated = await model_tasks.run(
            "knowledge_output",
            lambda: query_engine.generate_knowledge_output(output.id, title, source_text),
            timeout_s=MODEL_TASK_TIMEOUT_S,
        )
        user_tags = knowledge_output_tags(output.id, req.user_tags)
        item = build_knowledge_output_markdown(
            title,
            output.id,
            generated,
            source_files=source_files,
            tags=req.user_tags,
        )
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc
    except ModelTaskTimeout as exc:
        logger.warning("[api/knowledge-output] timeout id=%s type=%s", exc.task_id, req.output_type)
        raise HTTPException(504, MODEL_TIMEOUT_MESSAGE) from exc

    result = _write_import_and_enqueue(
        "knowledge",
        item,
        collection=req.collection or "Knowledge Outputs",
        user_tags=user_tags,
    )
    return {
        **result,
        "output_type": output.id,
        "source_files": source_files,
        "preview": generated[:500],
    }


async def obsidian_related_notes(req: ObsidianRelatedRequest):
    if query_engine is None:
        raise HTTPException(503, "Query engine not ready")
    query_text = " ".join(
        part.strip()
        for part in [req.selection or "", req.note_title or "", req.note_content or ""]
        if part and part.strip()
    ).strip()
    if not query_text:
        raise HTTPException(400, "Note content or selection is required")
    query_text = query_text[:4000]
    exclude = {
        value
        for value in {
            req.note_path,
            req.note_title,
            f"{req.note_title}.md" if req.note_title else "",
        }
        if value
    }
    limit = max(1, min(int(req.limit or 6), 12))
    chunks = query_engine.retriever.retrieve(
        query=query_text,
        file_filter=None,
        retrieval_mode=_normalize_retrieval_mode(req.retrieval_mode),
        prefer_tables=False,
        related_k=limit,
    )
    related_notes = QueryEngine._related_notes(
        [],
        chunks,
        extra_exclude_keys=exclude,
        limit=limit,
    )
    return {
        "related_notes": related_notes,
        "count": len(related_notes),
    }


def _build_knowledge_output_source(req: KnowledgeOutputRequest) -> tuple[str, list[str]]:
    _sync_app_state()
    return import_service.build_knowledge_output_source(app_state, req)


def _format_knowledge_file_context(file_name: str, chunks: list[dict]) -> str:
    return import_service.format_knowledge_file_context(file_name, chunks)


def _write_import_and_enqueue(
    prefix: str,
    item,
    collection: str,
    user_tags: list[str],
) -> dict:
    _sync_app_state()
    return import_service.write_import_and_enqueue(
        app_state,
        prefix=prefix,
        item=item,
        collection=collection,
        user_tags=user_tags,
    )


async def upload_file(file: UploadFile):
    """上传文件到第一个监控目录（支持所有已注册格式）。"""
    if not watch_dirs or ingest_queue is None:
        raise HTTPException(503, "Not ready")
    dest = import_service.safe_upload_destination(watch_dirs[0].path, file.filename or "")
    supported_exts = pipeline.registry.supported_extensions
    suffix = dest.suffix.lower()
    if suffix not in supported_exts:
        raise HTTPException(400, f"Unsupported file type: {suffix}. Supported: {supported_exts}")
    try:
        with dest.open("wb") as f:
            while chunk := await file.read(1024 * 1024):
                f.write(chunk)
    finally:
        await file.close()

    return ingest_queue.submit(dest)


async def create_demo_library():
    if ingest_queue is None:
        raise HTTPException(503, "Not ready")
    from src.maintenance.demo import create_demo_files

    result = create_demo_files(CONFIG_PATH)
    queued = []
    for item in result["files"]:
        queued.append(ingest_queue.submit(Path(item["path"])))
    return {**result, "queued": len(queued), "queue_results": queued}


async def preview_file(file_id: int):
    file_path, media_type = _resolve_preview_file(file_id)
    return FileResponse(str(file_path), media_type=media_type)


async def preview_file_head(file_id: int):
    file_path, media_type = _resolve_preview_file(file_id)
    return Response(
        status_code=200,
        headers={
            "content-length": str(file_path.stat().st_size),
            "content-type": media_type,
        },
    )


def _resolve_preview_file(file_id: int) -> tuple[Path, str]:
    if store is None:
        raise HTTPException(503, "Store not ready")
    record = store.get_file_by_id(file_id)
    if record is None:
        raise HTTPException(404, "File not found")
    file_path = Path(record["file_path"])
    if not file_path.exists():
        raise HTTPException(404, "File not found on disk")
    # 根据扩展名选择 MIME type
    suffix = file_path.suffix.lower()
    media_types = {
        ".pdf": "application/pdf",
        ".md": "text/markdown; charset=utf-8",
        ".txt": "text/plain; charset=utf-8",
        ".docx": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
    }
    media_type = media_types.get(suffix, "application/octet-stream")
    return file_path, media_type


async def list_file_chunks(file_id: int, max_text_chars: int = 500):
    """本地调试：查看文件实际切出的 chunk 和文本预览。"""
    if store is None or query_engine is None:
        raise HTTPException(503, "Not ready")
    record = store.get_file_by_id(file_id)
    if record is None:
        raise HTTPException(404, "File not found")

    chunk_rows = store.list_file_chunks(file_id)
    qdrant_ids = [row["qdrant_id"] for row in chunk_rows]
    payloads = query_engine.retriever.fetch_chunks_by_ids(
        qdrant_ids,
        max_text_chars=max(0, min(max_text_chars, 2000)),
    )
    chunks = []
    for row in chunk_rows:
        payload = payloads.get(row["qdrant_id"], {})
        chunks.append(
            {
                **row,
                "text_preview": payload.get("text_preview", ""),
                "text_length": payload.get("text_length", 0),
            }
        )
    return {"file": record, "chunks": chunks, "count": len(chunks)}


async def list_history(limit: int = 50):
    if store is None:
        raise HTTPException(503, "Store not ready")
    items = store.list_history(limit=limit)
    return query_service.decode_history_items(items)


async def search_history(q: str, limit: int = 20):
    if store is None:
        raise HTTPException(503, "Store not ready")
    items = store.search_history(q, limit=limit)
    return query_service.decode_history_items(items)


async def clear_history():
    if store is None:
        raise HTTPException(503, "Store not ready")
    store.clear_history()
    return {"ok": True}


async def list_favorites():
    if store is None:
        raise HTTPException(503, "Store not ready")
    return store.list_favorites()


async def toggle_favorite(file_id: int):
    if store is None:
        raise HTTPException(503, "Store not ready")
    added = store.toggle_favorite(file_id)
    return {"file_id": file_id, "favorited": added}


async def summarize_files(req: SummarizeRequest):
    if store is None or query_engine is None:
        raise HTTPException(503, "Not ready")
    if not req.file_ids:
        raise HTTPException(400, "No file IDs provided")

    summaries: list[str] = []
    for fid in req.file_ids:
        record = store.get_file_by_id(fid)
        if not record or record["status"] != FileStatus.DONE:
            continue
        qdrant_ids = store.get_file_qdrant_ids(fid)
        try:
            md = await model_tasks.run(
                "summarize",
                lambda record=record, qdrant_ids=qdrant_ids: query_engine.summarize_file(
                    record["file_name"],
                    qdrant_ids,
                ),
                timeout_s=MODEL_TASK_TIMEOUT_S,
            )
        except ModelTaskTimeout as exc:
            logger.warning("[api/summarize] timeout id=%s file_id=%s", exc.task_id, fid)
            raise HTTPException(504, MODEL_TIMEOUT_MESSAGE) from exc
        summaries.append(md)

    if not summaries:
        raise HTTPException(404, "No valid files found")

    combined = "\n\n---\n\n".join(summaries)
    return Response(
        content=combined,
        media_type="text/markdown; charset=utf-8",
        headers={"Content-Disposition": 'attachment; filename="docflow-summary.md"'},
    )


async def debug_retrieve(req: DebugRetrieveRequest):
    """本地调试：返回向量、全文、融合、去重、精排的完整检索链路。"""
    if query_engine is None:
        raise HTTPException(503, "Query engine not ready")
    options = _resolve_query_options(req)
    prefer_tables = query_engine._is_table_query(req.question)
    try:
        return await model_tasks.run(
            "debug_retrieve",
            lambda: query_engine.retriever.debug_retrieve(
                req.question,
                file_filter=options.file_filter or None,
                retrieval_mode=options.retrieval_mode,
                prefer_tables=prefer_tables,
                include_rerank=req.include_rerank,
                max_text_chars=max(0, min(req.max_text_chars, 2000)),
            ),
            timeout_s=MODEL_TASK_TIMEOUT_S,
        )
    except ModelTaskTimeout as exc:
        logger.warning(
            "[api/debug/retrieve] timeout id=%s question=%r", exc.task_id, req.question[:80]
        )
        raise HTTPException(504, MODEL_TIMEOUT_MESSAGE) from exc


async def get_llm():
    if query_engine is None:
        raise HTTPException(503, "Not ready")
    backend = query_engine.generator.backend
    return {
        "current": query_engine.generator.current_model,
        "options": llm_options,
        "models": [_llm_model_status(model) for model in llm_options],
        "backend": backend,
        "network_mode": "cloud" if backend == "claude" else "local",
        "privacy_notice": (
            "云端回答已启用。提问内容会发送到你配置的外部模型服务。"
            if backend == "claude"
            else "本地回答已启用。默认不会把提问发送到外部模型服务。"
        ),
        "switch": dict(llm_switch_state),
    }


async def set_llm(req: LLMSwitchRequest):
    if query_engine is None:
        raise HTTPException(503, "Not ready")
    if req.model not in llm_options:
        raise HTTPException(400, f"Unknown model: {req.model}. Available: {llm_options}")
    gen = query_engine.generator
    if req.model == gen.current_model:
        _set_llm_switch_state("idle", model=req.model, message="Already using this model")
        return {"ok": True, "model": req.model, "unchanged": True}
    model_status = _llm_model_status(req.model)
    if gen.backend == "mlx" and not model_status["available"]:
        message = f"Model is not cached locally: {req.model}"
        _set_llm_switch_state("error", model=req.model, message=message)
        raise HTTPException(409, message)
    _set_llm_switch_state("switching", model=req.model)
    if gen.backend == "mlx":
        try:
            loaded_model, loaded_tokenizer = await model_tasks.run(
                "llm_switch",
                lambda: _load_mlx_model_candidate(req.model),
                timeout_s=MODEL_TASK_TIMEOUT_S,
            )
            gen._mlx_model = loaded_model
            gen._mlx_tokenizer = loaded_tokenizer
            gen.mlx_model_name = req.model
        except ModelTaskTimeout as exc:
            logger.warning("[api/llm] switch timeout id=%s model=%s", exc.task_id, req.model)
            _set_llm_switch_state("error", model=req.model, message=MODEL_TIMEOUT_MESSAGE)
            raise HTTPException(504, MODEL_TIMEOUT_MESSAGE) from exc
        except Exception as exc:
            message = str(exc) or "Model switch failed"
            logger.exception("[api/llm] switch failed model=%s", req.model)
            _set_llm_switch_state("error", model=req.model, message=message)
            raise HTTPException(500, message) from exc
    elif gen.backend == "claude":
        gen.claude_model = req.model
    else:
        gen.ollama_model = req.model
    _set_llm_switch_state("idle", model=req.model, message="Switched")
    logger.info(f"[llm] Switched to {req.model}")
    return {"ok": True, "model": req.model}


async def list_sources():
    """返回所有监控目录配置。"""
    return [
        {
            "path": str(wd.path),
            "recursive": wd.recursive,
            "extensions": wd.extensions or pipeline.registry.supported_extensions,
        }
        for wd in watch_dirs
    ]


async def health():
    with open(CONFIG_PATH) as f:
        cfg = yaml.safe_load(f)
    return health_service.build_health(
        cfg,
        timed_check=_timed_check,
        check_sqlite=_check_sqlite,
        check_qdrant=_check_qdrant,
        check_ollama=_check_ollama,
        check_models=_check_models,
        health_capabilities=_health_capabilities,
        aggregate_health_status=_aggregate_health_status,
        health_groups=_health_groups,
        health_actions=_health_actions,
    )


















def _register_api_routes() -> None:
    app.include_router(
        query_routes.create_router(
            {
                "query": query,
                "research": research,
                "query_stream": query_stream,
                "answer_feedback": answer_feedback,
                "list_conversations": list_conversations,
                "create_conversation": create_conversation,
                "list_conversation_messages": list_conversation_messages,
                "delete_conversation": delete_conversation,
            }
        )
    )
    app.include_router(
        library_routes.create_router(
            {
                "trigger_ingest": trigger_ingest,
                "queue_status": queue_status,
                "list_files": list_files,
                "library_meta": library_meta,
                "storage_usage": storage_usage,
                "update_file_metadata": update_file_metadata,
                "batch_favorite": batch_favorite,
                "batch_update_file_metadata": batch_update_file_metadata,
                "batch_rebuild_files": batch_rebuild_files,
                "preview_file": preview_file,
                "preview_file_head": preview_file_head,
                "list_file_chunks": list_file_chunks,
                "list_history": list_history,
                "search_history": search_history,
                "clear_history": clear_history,
                "list_favorites": list_favorites,
                "toggle_favorite": toggle_favorite,
                "summarize_files": summarize_files,
            }
        )
    )
    app.include_router(
        imports_routes.create_router(
            {
                "import_url": import_url,
                "create_note": create_note,
                "save_answer_note": save_answer_note,
                "create_knowledge_output": create_knowledge_output,
                "upload_file": upload_file,
                "create_demo_library": create_demo_library,
            }
        )
    )
    app.include_router(
        knowledge_routes.create_router(
            {
                "knowledge_overview": knowledge_overview,
            }
        )
    )
    app.include_router(
        settings_routes.create_router(
            {
                "get_llm": get_llm,
                "set_llm": set_llm,
                "list_sources": list_sources,
                "health": health,
            }
        )
    )
    app.include_router(
        maintenance_routes.create_router(
            {
                "debug_retrieve": debug_retrieve,
            }
        )
    )
    app.include_router(
        obsidian_routes.create_router(
            {
                "obsidian_related_notes": obsidian_related_notes,
            }
        )
    )


_register_api_routes()


# ---------------------------------------------------------------------------
# Static files (frontend)
# ---------------------------------------------------------------------------

STATIC_DIR = Path(__file__).parent.parent.parent / "frontend"
if STATIC_DIR.exists():

    @app.get("/favicon.ico", include_in_schema=False)
    async def favicon_ico():
        return FileResponse(str(STATIC_DIR / "favicon.svg"), media_type="image/svg+xml")

    @app.head("/favicon.ico", include_in_schema=False)
    async def favicon_ico_head():
        favicon_path = STATIC_DIR / "favicon.svg"
        return Response(
            status_code=200,
            headers={
                "content-length": str(favicon_path.stat().st_size),
                "content-type": "image/svg+xml",
            },
        )

    app.mount("/", StaticFiles(directory=str(STATIC_DIR), html=True), name="frontend")


_STATE_FIELD_NAMES = {
    "pipeline",
    "ingest_queue",
    "query_engine",
    "store",
    "watcher",
    "watch_dirs",
    "llm_options",
    "model_tasks",
    "llm_switch_state",
}


class _ApiModule(types.ModuleType):
    def __getattribute__(self, name: str):
        if name in _STATE_FIELD_NAMES:
            state = types.ModuleType.__getattribute__(self, "app_state")
            return getattr(state, name)
        return types.ModuleType.__getattribute__(self, name)

    def __setattr__(self, name: str, value):
        if name in _STATE_FIELD_NAMES:
            state = types.ModuleType.__getattribute__(self, "app_state")
            setattr(state, name, value)
        types.ModuleType.__setattr__(self, name, value)


sys.modules[__name__].__class__ = _ApiModule
