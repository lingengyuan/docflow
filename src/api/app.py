"""
DocFlow FastAPI 后端。
"""

from __future__ import annotations

import logging
import os
import sqlite3
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from pathlib import Path
from time import perf_counter

import yaml
import json

from fastapi import FastAPI, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, Response, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from src.ingest.pipeline import IngestPipeline
from src.ingest.queue import IngestQueue
from src.ingest.store import DocStore
from src.ingest.watcher import FolderWatcher, WatchDir, _is_excluded
from src.query.engine import QueryEngine

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

logging.getLogger("transformers.tokenization_utils_base").setLevel(logging.ERROR)
logging.getLogger("FlagEmbedding").setLevel(logging.WARNING)

CONFIG_PATH = Path(__file__).parent.parent.parent / "config.yaml"
COLLECTION_NAME = "docflow"


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

ml_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="ml-inference")


def _timed_check(fn) -> dict:
    start = perf_counter()
    try:
        result = fn()
        if not isinstance(result, dict):
            result = {"status": "ok"}
    except Exception as exc:
        result = {"status": "unavailable", "error": str(exc)}
    result["latency_ms"] = round((perf_counter() - start) * 1000, 2)
    return result


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
            result.append(WatchDir(
                path=Path(entry["path"]).expanduser(),
                recursive=entry.get("recursive", False),
                extensions=entry.get("extensions", []),
            ))
    return result


def _configured_model_names(cfg: dict) -> dict[str, str]:
    ollama_cfg = cfg.get("ollama", {})
    llm_cfg = cfg.get("llm", {})
    embedding_cfg = cfg.get("embedding", {})
    reranker_cfg = cfg.get("reranker", {})
    vlm_cfg = cfg.get("vlm", {})
    ingest_cfg = cfg.get("ingest", {})
    return {
        "embedding": embedding_cfg.get("model", ""),
        "reranker": reranker_cfg.get("model", ""),
        "llm": llm_cfg.get("mlx_model") or llm_cfg.get("ollama_model") or ollama_cfg.get("llm_model", ""),
        "llm_enhanced": llm_cfg.get("mlx_model_enhanced") or ollama_cfg.get("llm_model_enhanced", ""),
        "ocr": ollama_cfg.get("ocr_model", ""),
        "contextual_prefix": ingest_cfg.get("contextual_prefix_model", ""),
        "vlm": vlm_cfg.get("model", ""),
    }


def _hf_cache_dir() -> Path:
    hub_cache = os.getenv("HUGGINGFACE_HUB_CACHE")
    if hub_cache:
        return Path(hub_cache).expanduser()
    hf_home = Path(os.getenv("HF_HOME", str(Path.home() / ".cache" / "huggingface"))).expanduser()
    return hf_home / "hub"


def _is_hf_model_cached(model_name: str) -> bool:
    if not model_name or "/" not in model_name:
        return False
    model_dir = _hf_cache_dir() / f"models--{model_name.replace('/', '--')}"
    snapshots_dir = model_dir / "snapshots"
    if not snapshots_dir.exists():
        return False
    return any(snap.is_dir() for snap in snapshots_dir.iterdir())


@asynccontextmanager
async def lifespan(app: FastAPI):
    global pipeline, ingest_queue, query_engine, store, watcher, watch_dirs, llm_options
    with open(CONFIG_PATH) as f:
        cfg = yaml.safe_load(f)

    llm_cfg = cfg.get("llm", {})
    ingest_cfg = cfg.get("ingest", {})
    backend = llm_cfg.get("backend", "local")
    if backend == "mlx":
        llm_options = list(dict.fromkeys([
            llm_cfg.get("mlx_model", ""),
            llm_cfg.get("mlx_model_enhanced", ""),
        ]))
    elif backend == "claude":
        llm_options = list(dict.fromkeys([
            llm_cfg.get("claude_model", ""),
            llm_cfg.get("claude_model_enhanced", ""),
        ]))
    else:
        llm_options = list(dict.fromkeys([
            llm_cfg.get("ollama_model", cfg["ollama"]["llm_model"]),
            llm_cfg.get("ollama_model_enhanced", cfg["ollama"].get("llm_model_enhanced", "")),
        ]))
    llm_options = [m for m in llm_options if m]

    db_path = Path(cfg["paths"]["db_path"]).expanduser()
    watch_dirs = _parse_watch_dirs(cfg)

    store = DocStore(db_path)

    # 清理上次崩溃遗留的 processing 状态，确保启动扫描能重新处理这些文件
    n_reset = store.reset_processing_files()
    if n_reset:
        logger.info(f"[startup] Reset {n_reset} interrupted file(s) from 'processing' → 'error'")

    pipeline = IngestPipeline.from_config(CONFIG_PATH, store=store)
    query_engine = QueryEngine.from_config(CONFIG_PATH, store=store)

    # CPU embedding 不需要走 MLX Metal 执行器，直接在 ingest worker 线程里跑
    # ml_executor 只保留给 MLX 推理（reranker、LLM）
    ingest_queue = IngestQueue(
        pipeline,
        on_done=None,
        ml_executor=None,
        parse_workers=ingest_cfg.get("parse_workers", 2),
        microbatch_max_files=ingest_cfg.get("microbatch_max_files", 8),
        microbatch_max_chunks=ingest_cfg.get("microbatch_max_chunks", 128),
        microbatch_linger_ms=ingest_cfg.get("microbatch_linger_ms", 75),
    )
    ingest_queue.start()

    watcher = FolderWatcher(pipeline, watch_dirs, ingest_queue=ingest_queue)
    watcher.start()

    import asyncio
    loop = asyncio.get_event_loop()
    logger.info("Warming up embedding and reranker models...")
    await loop.run_in_executor(ml_executor, _warmup_models)
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
            logger.info(f"[cleanup] Removed deleted file: {r['file_name']} ({len(r['qdrant_ids'])} vectors)")

    # Background scan: enqueue existing files (skip .obsidian/.trash/.git)
    supported_exts = pipeline.registry.supported_extensions
    all_files: list[Path] = []
    for wd in watch_dirs:
        for ext in (wd.extensions if wd.extensions else supported_exts):
            pattern = f"**/*{ext}" if wd.recursive else f"*{ext}"
            all_files.extend(f for f in wd.path.glob(pattern) if not _is_excluded(f))
    if all_files:
        ingest_queue.submit_many(all_files)

    yield

    if watcher:
        watcher.stop()
    if ingest_queue:
        ingest_queue.stop()


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------

app = FastAPI(title="DocFlow", lifespan=lifespan)


# ---------------------------------------------------------------------------
# API routes
# ---------------------------------------------------------------------------

class QueryRequest(BaseModel):
    question: str
    file_filter: list[str] | None = None
    conversation_id: int | None = None


class DebugRetrieveRequest(BaseModel):
    question: str
    file_filter: list[str] | None = None
    include_rerank: bool = True
    max_text_chars: int = 300


class QueryResponse(BaseModel):
    answer: str
    citations: list[dict]
    conversation_id: int | None = None


class ConversationCreateRequest(BaseModel):
    title: str = ""


@app.post("/api/query", response_model=QueryResponse)
async def query(req: QueryRequest):
    if query_engine is None:
        raise HTTPException(503, "Query engine not ready")
    conversation_id = _resolve_conversation_id(req.conversation_id)
    conversation_context = _conversation_context(conversation_id)
    retrieval_query = _build_retrieval_query(req.question, conversation_context)
    if store is not None and conversation_id is not None:
        store.add_message(
            conversation_id=conversation_id,
            role="user",
            content=req.question,
            file_filter_json=json.dumps(req.file_filter or [], ensure_ascii=False),
        )
    import asyncio
    loop = asyncio.get_event_loop()
    result = await loop.run_in_executor(
        ml_executor,
        lambda: query_engine.query(
            req.question,
            file_filter=req.file_filter,
            conversation_context=conversation_context,
            retrieval_query=retrieval_query,
        ),
    )
    seen_files: dict[str, dict] = {}
    for c in result.citations:
        key = c.file_path or c.file_name
        if key not in seen_files or c.score > seen_files[key]["score"]:
            seen_files[key] = {
                "file_name": c.file_name,
                "file_path": c.file_path,
                "page_num": c.page_num,
                "section": c.section,
                "snippet": c.snippet,
                "score": round(c.score, 4),
            }
    citations_data = list(seen_files.values())
    if store is not None:
        store.add_history(
            question=req.question,
            answer=result.text,
            citations_json=json.dumps(citations_data, ensure_ascii=False),
            file_filter_json=json.dumps(req.file_filter or [], ensure_ascii=False),
            conversation_id=conversation_id or 0,
        )
        if conversation_id is not None:
            store.add_message(
                conversation_id=conversation_id,
                role="assistant",
                content=result.text,
                citations_json=json.dumps(citations_data, ensure_ascii=False),
                file_filter_json=json.dumps(req.file_filter or [], ensure_ascii=False),
            )
    return QueryResponse(answer=result.text, citations=citations_data, conversation_id=conversation_id)


@app.post("/api/query/stream")
async def query_stream(req: QueryRequest, request: Request):
    """SSE 流式查询：先返回 citations，再逐 token 返回答案。"""
    if query_engine is None:
        raise HTTPException(503, "Query engine not ready")
    import asyncio, queue, threading

    conversation_id = _resolve_conversation_id(req.conversation_id)
    conversation_context = _conversation_context(conversation_id)
    retrieval_query = _build_retrieval_query(req.question, conversation_context)
    if store is not None and conversation_id is not None:
        store.add_message(
            conversation_id=conversation_id,
            role="user",
            content=req.question,
            file_filter_json=json.dumps(req.file_filter or [], ensure_ascii=False),
        )

    loop = asyncio.get_event_loop()
    q: queue.Queue = queue.Queue()
    cancel = threading.Event()

    def _run():
        try:
            if conversation_id is not None:
                q.put(("conversation", {"conversation_id": conversation_id}))
            chunks, token_gen = query_engine.query_stream(
                req.question,
                file_filter=req.file_filter,
                cancel_event=cancel,
                conversation_context=conversation_context,
                retrieval_query=retrieval_query,
            )
            if cancel.is_set():
                return
            seen_files: dict[str, dict] = {}
            for c in chunks:
                key = c.get("file_path") or c["file_name"]
                score = c.get("rerank_score", c.get("rrf_score", 0.0))
                if key not in seen_files or score > seen_files[key]["score"]:
                    seen_files[key] = {
                        "file_name": c["file_name"],
                        "file_path": c.get("file_path", ""),
                        "page_num": c["page_num"],
                        "section": c.get("section", ""),
                        "snippet": c["text"][:200],
                        "score": round(score, 4),
                    }
            citations_data = list(seen_files.values())
            if cancel.is_set():
                return
            q.put(("citations", citations_data))
            full_answer = []
            for token in token_gen:
                if cancel.is_set():
                    return
                full_answer.append(token)
                q.put(("token", token))
            answer_text = "".join(full_answer).strip()
            if not cancel.is_set() and store is not None:
                store.add_history(
                    question=req.question,
                    answer=answer_text,
                    citations_json=json.dumps(citations_data, ensure_ascii=False),
                    file_filter_json=json.dumps(req.file_filter or [], ensure_ascii=False),
                    conversation_id=conversation_id or 0,
                )
                if conversation_id is not None:
                    store.add_message(
                        conversation_id=conversation_id,
                        role="assistant",
                        content=answer_text,
                        citations_json=json.dumps(citations_data, ensure_ascii=False),
                        file_filter_json=json.dumps(req.file_filter or [], ensure_ascii=False),
                    )
            if not cancel.is_set():
                q.put(("done", ""))
        except Exception as e:
            if not cancel.is_set():
                q.put(("error", str(e)))

    ml_executor.submit(_run)

    async def event_stream():
        while True:
            if await request.is_disconnected():
                cancel.set()
                break
            try:
                event, data = await loop.run_in_executor(None, q.get, True, 1)
            except queue.Empty:
                continue
            yield f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"
            if event in ("done", "error"):
                break

    return StreamingResponse(event_stream(), media_type="text/event-stream")


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
    if not _looks_like_followup(question):
        return question
    previous_user_questions = [
        message["content"]
        for message in conversation_context
        if message.get("role") == "user" and message.get("content")
    ]
    if not previous_user_questions:
        return question
    return f"{previous_user_questions[-1]}\n{question}"


def _looks_like_followup(question: str) -> bool:
    q = question.strip().lower()
    markers = (
        "展开", "继续", "上面", "刚才", "前面", "这个", "那个", "这点", "第二点",
        "第三点", "第一点", "it", "that", "this", "above", "previous",
    )
    return any(marker in q for marker in markers)


@app.get("/api/conversations")
async def list_conversations(limit: int = 50):
    if store is None:
        raise HTTPException(503, "Store not ready")
    return store.list_conversations(limit=limit)


@app.post("/api/conversations")
async def create_conversation(req: ConversationCreateRequest):
    if store is None:
        raise HTTPException(503, "Store not ready")
    conversation_id = store.create_conversation(req.title)
    return store.get_conversation(conversation_id)


@app.get("/api/conversations/{conversation_id}/messages")
async def list_conversation_messages(conversation_id: int, limit: int = 100):
    if store is None:
        raise HTTPException(503, "Store not ready")
    if store.get_conversation(conversation_id) is None:
        raise HTTPException(404, "Conversation not found")
    items = store.list_messages(conversation_id, limit=limit)
    for item in items:
        try:
            item["citations"] = json.loads(item["citations"])
        except Exception:
            item["citations"] = []
        try:
            item["file_filter"] = json.loads(item["file_filter"])
        except Exception:
            item["file_filter"] = []
    return items


@app.delete("/api/conversations/{conversation_id}")
async def delete_conversation(conversation_id: int):
    if store is None:
        raise HTTPException(503, "Store not ready")
    if not store.delete_conversation(conversation_id):
        raise HTTPException(404, "Conversation not found")
    return {"ok": True}


@app.post("/api/ingest")
async def trigger_ingest():
    """手动触发全量扫描所有监控目录（异步，立即返回）。"""
    if ingest_queue is None or not watch_dirs:
        raise HTTPException(503, "Pipeline not ready")
    supported_exts = pipeline.registry.supported_extensions
    all_files: list[Path] = []
    for wd in watch_dirs:
        for ext in (wd.extensions if wd.extensions else supported_exts):
            pattern = f"**/*{ext}" if wd.recursive else f"*{ext}"
            all_files.extend(f for f in wd.path.glob(pattern) if not _is_excluded(f))
    result = ingest_queue.submit_many(all_files)
    return {**result, "files": [p.name for p in all_files]}


@app.get("/api/queue")
async def queue_status():
    if ingest_queue is None:
        return {
            "queue_size": 0,
            "processing": None,
            "processing_files": [],
            "pending_files": [],
            "progress": None,
            "last_completed": None,
        }
    return ingest_queue.status()


def _warmup_models():
    """预热 embedding + reranker + LLM 模型。"""
    try:
        em = query_engine.retriever.embed_model
        warmup_query = "Instruct: Retrieve relevant text passages that answer the query.\nQuery: warmup"
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


@app.get("/api/files")
async def list_files(status: str | None = None):
    if store is None:
        raise HTTPException(503, "Store not ready")
    return store.list_files(status=status)


@app.post("/api/upload")
async def upload_file(file: UploadFile):
    """上传文件到第一个监控目录（支持所有已注册格式）。"""
    if not watch_dirs or ingest_queue is None:
        raise HTTPException(503, "Not ready")
    original_name = file.filename or ""
    safe_name = Path(original_name).name
    if not safe_name:
        raise HTTPException(400, "Missing filename")
    supported_exts = pipeline.registry.supported_extensions
    suffix = Path(safe_name).suffix.lower()
    if suffix not in supported_exts:
        raise HTTPException(400, f"Unsupported file type: {suffix}. Supported: {supported_exts}")

    dest = watch_dirs[0].path / safe_name
    try:
        with dest.open("wb") as f:
            while chunk := await file.read(1024 * 1024):
                f.write(chunk)
    finally:
        await file.close()

    return ingest_queue.submit(dest)


@app.get("/api/file/{file_id}/preview")
async def preview_file(file_id: int):
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
    return FileResponse(str(file_path), media_type=media_type)


@app.get("/api/file/{file_id}/chunks")
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
        chunks.append({
            **row,
            "text_preview": payload.get("text_preview", ""),
            "text_length": payload.get("text_length", 0),
        })
    return {"file": record, "chunks": chunks, "count": len(chunks)}


@app.get("/api/history")
async def list_history(limit: int = 50):
    if store is None:
        raise HTTPException(503, "Store not ready")
    items = store.list_history(limit=limit)
    for item in items:
        try:
            item["citations"] = json.loads(item["citations"])
        except Exception:
            item["citations"] = []
        try:
            item["file_filter"] = json.loads(item["file_filter"])
        except Exception:
            item["file_filter"] = []
    return items


@app.get("/api/history/search")
async def search_history(q: str, limit: int = 20):
    if store is None:
        raise HTTPException(503, "Store not ready")
    items = store.search_history(q, limit=limit)
    for item in items:
        try:
            item["citations"] = json.loads(item["citations"])
        except Exception:
            item["citations"] = []
        try:
            item["file_filter"] = json.loads(item["file_filter"])
        except Exception:
            item["file_filter"] = []
    return items


@app.delete("/api/history")
async def clear_history():
    if store is None:
        raise HTTPException(503, "Store not ready")
    store.clear_history()
    return {"ok": True}


@app.get("/api/favorites")
async def list_favorites():
    if store is None:
        raise HTTPException(503, "Store not ready")
    return store.list_favorites()


@app.post("/api/favorites/{file_id}")
async def toggle_favorite(file_id: int):
    if store is None:
        raise HTTPException(503, "Store not ready")
    added = store.toggle_favorite(file_id)
    return {"file_id": file_id, "favorited": added}


class SummarizeRequest(BaseModel):
    file_ids: list[int]


@app.post("/api/summarize")
async def summarize_files(req: SummarizeRequest):
    if store is None or query_engine is None:
        raise HTTPException(503, "Not ready")
    if not req.file_ids:
        raise HTTPException(400, "No file IDs provided")

    import asyncio
    loop = asyncio.get_event_loop()

    summaries: list[str] = []
    for fid in req.file_ids:
        record = store.get_file_by_id(fid)
        if not record or record["status"] != "done":
            continue
        qdrant_ids = store.get_file_qdrant_ids(fid)
        md = await loop.run_in_executor(
            ml_executor, query_engine.summarize_file, record["file_name"], qdrant_ids
        )
        summaries.append(md)

    if not summaries:
        raise HTTPException(404, "No valid files found")

    combined = "\n\n---\n\n".join(summaries)
    return Response(
        content=combined,
        media_type="text/markdown; charset=utf-8",
        headers={"Content-Disposition": 'attachment; filename="docflow-summary.md"'},
    )


@app.post("/api/debug/retrieve")
async def debug_retrieve(req: DebugRetrieveRequest):
    """本地调试：返回向量、全文、融合、去重、精排的完整检索链路。"""
    if query_engine is None:
        raise HTTPException(503, "Query engine not ready")
    import asyncio
    loop = asyncio.get_event_loop()
    prefer_tables = query_engine._is_table_query(req.question)
    return await loop.run_in_executor(
        ml_executor,
        lambda: query_engine.retriever.debug_retrieve(
            req.question,
            file_filter=req.file_filter,
            prefer_tables=prefer_tables,
            include_rerank=req.include_rerank,
            max_text_chars=max(0, min(req.max_text_chars, 2000)),
        ),
    )


@app.get("/api/llm")
async def get_llm():
    if query_engine is None:
        raise HTTPException(503, "Not ready")
    return {
        "current": query_engine.generator.current_model,
        "options": llm_options,
        "backend": query_engine.generator.backend,
    }


class LLMSwitchRequest(BaseModel):
    model: str


@app.post("/api/llm")
async def set_llm(req: LLMSwitchRequest):
    if query_engine is None:
        raise HTTPException(503, "Not ready")
    if req.model not in llm_options:
        raise HTTPException(400, f"Unknown model: {req.model}. Available: {llm_options}")
    gen = query_engine.generator
    if gen.backend == "mlx":
        import asyncio
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(ml_executor, gen._load_mlx_model, req.model)
    elif gen.backend == "claude":
        gen.claude_model = req.model
    else:
        gen.ollama_model = req.model
    logger.info(f"[llm] Switched to {req.model}")
    return {"ok": True, "model": req.model}


@app.get("/api/sources")
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


@app.get("/api/health")
async def health():
    with open(CONFIG_PATH) as f:
        cfg = yaml.safe_load(f)
    checks = {
        "api": {"status": "ok"},
        "sqlite": _timed_check(lambda: _check_sqlite(cfg)),
        "qdrant": _timed_check(lambda: _check_qdrant(cfg)),
        "ollama": _timed_check(lambda: _check_ollama(cfg)),
        "models": _check_models(cfg),
    }
    capabilities = _health_capabilities(cfg, checks)
    status = _aggregate_health_status(checks)
    return {
        "status": status,
        "checks": checks,
        "capabilities": capabilities,
    }


def _check_sqlite(cfg: dict) -> dict:
    active_store = store
    if active_store is not None:
        with active_store._conn() as conn:
            conn.execute("SELECT 1").fetchone()
            conn.execute("CREATE TEMP TABLE IF NOT EXISTS health_check(value INTEGER)")
            conn.execute("DELETE FROM health_check")
            conn.execute("INSERT INTO health_check(value) VALUES (1)")
            fts_tables = [
                row["name"]
                for row in conn.execute(
                    """
                    SELECT name FROM sqlite_master
                    WHERE type = 'table'
                      AND name IN ('chunks_fts', 'chunks_fts_trigram', 'history_fts')
                    ORDER BY name
                    """
                ).fetchall()
            ]
        required_fts_tables = {"chunks_fts", "chunks_fts_trigram", "history_fts"}
        missing_fts_tables = sorted(required_fts_tables - set(fts_tables))
        return {
            "status": "ok" if not missing_fts_tables else "unavailable",
            "mode": "runtime",
            "write_check": "ok",
            "fts_tables": fts_tables,
            "missing_fts_tables": missing_fts_tables,
            "quick_check": "skipped during app runtime",
            "note": "Use `python main.py doctor --strict` for a full SQLite integrity check.",
        }
    else:
        db_path = Path(cfg["paths"]["db_path"]).expanduser()
        conn = sqlite3.connect(db_path)
        try:
            conn.execute("SELECT 1").fetchone()
            conn.execute("CREATE TEMP TABLE IF NOT EXISTS health_check(value INTEGER)")
            conn.execute("INSERT INTO health_check(value) VALUES (1)")
            quick_check = conn.execute("PRAGMA quick_check").fetchone()[0]
        finally:
            conn.close()

    status = "ok" if quick_check == "ok" else "unavailable"
    return {"status": status, "mode": "offline", "quick_check": quick_check}


def _check_qdrant(cfg: dict) -> dict:
    from qdrant_client import QdrantClient

    client = QdrantClient(host=cfg["qdrant"]["host"], port=cfg["qdrant"]["port"], timeout=2)
    collection = cfg.get("qdrant", {}).get("collection", COLLECTION_NAME)
    info = client.get_collection(collection)
    return {
        "status": "ok",
        "collection": collection,
        "points_count": getattr(info, "points_count", 0),
        "vectors_count": getattr(info, "vectors_count", None),
    }


def _check_ollama(cfg: dict) -> dict:
    import requests

    ollama_cfg = cfg.get("ollama", {})
    ingest_cfg = cfg.get("ingest", {})
    llm_cfg = cfg.get("llm", {})
    base_url = ollama_cfg.get("base_url", "http://localhost:11434").rstrip("/")
    response = requests.get(f"{base_url}/api/tags", timeout=2)
    response.raise_for_status()
    data = response.json()
    installed = set()
    for item in data.get("models", []):
        name = item.get("name", "")
        if not name:
            continue
        installed.add(name)
        installed.add(name.split(":", 1)[0])

    required = {
        "ocr": ollama_cfg.get("ocr_model", ""),
        "contextual_prefix": (
            ingest_cfg.get("contextual_prefix_model", "")
            if ingest_cfg.get("contextual_prefix_mode") == "ollama"
            else ""
        ),
        "llm": llm_cfg.get("ollama_model", ollama_cfg.get("llm_model", ""))
        if llm_cfg.get("backend", "local") == "local"
        else "",
    }
    models = {}
    missing = []
    for purpose, model in required.items():
        if not model:
            continue
        available = model in installed or model.split(":", 1)[0] in installed
        models[purpose] = {"model": model, "available": available}
        if not available:
            missing.append(model)

    status = "ok" if not missing else "degraded"
    return {
        "status": status,
        "base_url": base_url,
        "models": models,
        "missing_models": missing,
    }


def _check_models(cfg: dict) -> dict:
    names = _configured_model_names(cfg)
    local_models = {
        name: {
            "model": model,
            "cached": _is_hf_model_cached(model),
        }
        for name, model in names.items()
        if model and "/" in model
    }
    missing = [
        item["model"]
        for item in local_models.values()
        if not item["cached"]
    ]
    return {
        "status": "ok" if not missing else "degraded",
        "local_cache": local_models,
        "missing_local_cache": missing,
        "note": "Local model check only inspects cache folders and does not download models.",
    }


def _health_capabilities(cfg: dict, checks: dict) -> dict:
    ollama_models = checks["ollama"].get("models", {})
    ingest_cfg = cfg.get("ingest", {})
    contextual_prefix_enabled = ingest_cfg.get("contextual_prefix", False)
    contextual_prefix_available = (
        contextual_prefix_enabled
        and (
            ingest_cfg.get("contextual_prefix_mode") != "ollama"
            or ollama_models.get("contextual_prefix", {}).get("available", False)
        )
    )
    return {
        "query": checks["sqlite"]["status"] == "ok" and checks["qdrant"]["status"] == "ok",
        "ingest": checks["sqlite"]["status"] == "ok" and checks["qdrant"]["status"] == "ok",
        "ocr": ollama_models.get("ocr", {}).get("available", False),
        "contextual_prefix": contextual_prefix_available,
        "contextual_prefix_enabled": contextual_prefix_enabled,
        "contextual_prefix_mode": ingest_cfg.get("contextual_prefix_mode", "metadata"),
    }


def _aggregate_health_status(checks: dict) -> str:
    critical = [checks["sqlite"]["status"], checks["qdrant"]["status"]]
    if any(status != "ok" for status in critical):
        return "unavailable"
    optional = [checks["ollama"]["status"], checks["models"]["status"]]
    if any(status != "ok" for status in optional):
        return "degraded"
    return "ok"


# ---------------------------------------------------------------------------
# Static files (frontend)
# ---------------------------------------------------------------------------

STATIC_DIR = Path(__file__).parent.parent.parent / "frontend"
if STATIC_DIR.exists():
    app.mount("/", StaticFiles(directory=str(STATIC_DIR), html=True), name="frontend")
