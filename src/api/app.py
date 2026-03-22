"""
DocFlow FastAPI 后端。
"""

from __future__ import annotations

import logging
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from pathlib import Path

import yaml
import json

from fastapi import FastAPI, HTTPException, UploadFile
from fastapi.responses import FileResponse, Response, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from src.ingest.pipeline import IngestPipeline
from src.ingest.queue import IngestQueue
from src.ingest.store import DocStore
from src.ingest.watcher import FolderWatcher, WatchDir
from src.query.engine import QueryEngine

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

logging.getLogger("transformers.tokenization_utils_base").setLevel(logging.ERROR)
logging.getLogger("FlagEmbedding").setLevel(logging.WARNING)

CONFIG_PATH = Path(__file__).parent.parent.parent / "config.yaml"


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


@asynccontextmanager
async def lifespan(app: FastAPI):
    global pipeline, ingest_queue, query_engine, store, watcher, watch_dirs, llm_options
    with open(CONFIG_PATH) as f:
        cfg = yaml.safe_load(f)

    llm_cfg = cfg.get("llm", {})
    if llm_cfg.get("backend", "local") == "mlx":
        llm_options = list(dict.fromkeys([
            llm_cfg.get("mlx_model", ""),
            llm_cfg.get("mlx_model_enhanced", ""),
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
    pipeline = IngestPipeline.from_config(CONFIG_PATH)
    query_engine = QueryEngine.from_config(CONFIG_PATH)

    # FTS5 取代 BM25 pickle，无需热重载回调（增量写入，始终最新）
    ingest_queue = IngestQueue(
        pipeline,
        on_done=None,
        ml_executor=ml_executor,
    )
    ingest_queue.start()

    watcher = FolderWatcher(pipeline, watch_dirs)
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

    # Background scan: enqueue existing files
    supported_exts = pipeline.registry.supported_extensions
    all_files: list[Path] = []
    for wd in watch_dirs:
        for ext in (wd.extensions if wd.extensions else supported_exts):
            pattern = f"**/*{ext}" if wd.recursive else f"*{ext}"
            all_files.extend(wd.path.glob(pattern))
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


class QueryResponse(BaseModel):
    answer: str
    citations: list[dict]


@app.post("/api/query", response_model=QueryResponse)
async def query(req: QueryRequest):
    if query_engine is None:
        raise HTTPException(503, "Query engine not ready")
    import asyncio
    loop = asyncio.get_event_loop()
    result = await loop.run_in_executor(
        ml_executor, lambda: query_engine.query(req.question, file_filter=req.file_filter)
    )
    seen_files: dict[str, dict] = {}
    for c in result.citations:
        if c.file_name not in seen_files or c.score > seen_files[c.file_name]["score"]:
            seen_files[c.file_name] = {
                "file_name": c.file_name,
                "page_num": c.page_num,
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
        )
    return QueryResponse(answer=result.text, citations=citations_data)


@app.post("/api/query/stream")
async def query_stream(req: QueryRequest):
    """SSE 流式查询：先返回 citations，再逐 token 返回答案。"""
    if query_engine is None:
        raise HTTPException(503, "Query engine not ready")
    import asyncio, queue, threading

    loop = asyncio.get_event_loop()
    q: queue.Queue = queue.Queue()

    def _run():
        try:
            chunks, token_gen = query_engine.query_stream(
                req.question, file_filter=req.file_filter
            )
            seen_files: dict[str, dict] = {}
            for c in chunks:
                fn = c["file_name"]
                score = c.get("rerank_score", c.get("rrf_score", 0.0))
                if fn not in seen_files or score > seen_files[fn]["score"]:
                    seen_files[fn] = {
                        "file_name": fn,
                        "page_num": c["page_num"],
                        "snippet": c["text"][:200],
                        "score": round(score, 4),
                    }
            citations_data = list(seen_files.values())
            q.put(("citations", citations_data))
            full_answer = []
            for token in token_gen:
                full_answer.append(token)
                q.put(("token", token))
            answer_text = "".join(full_answer).strip()
            if store is not None:
                store.add_history(
                    question=req.question,
                    answer=answer_text,
                    citations_json=json.dumps(citations_data, ensure_ascii=False),
                    file_filter_json=json.dumps(req.file_filter or [], ensure_ascii=False),
                )
            q.put(("done", ""))
        except Exception as e:
            q.put(("error", str(e)))

    ml_executor.submit(_run)

    async def event_stream():
        while True:
            try:
                event, data = await loop.run_in_executor(None, q.get, True, 600)
            except Exception:
                break
            yield f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"
            if event in ("done", "error"):
                break

    return StreamingResponse(event_stream(), media_type="text/event-stream")


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
            all_files.extend(wd.path.glob(pattern))
    result = ingest_queue.submit_many(all_files)
    return {**result, "files": [p.name for p in all_files]}


@app.get("/api/queue")
async def queue_status():
    if ingest_queue is None:
        return {"queue_size": 0, "processing": None, "pending_files": []}
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
    supported_exts = pipeline.registry.supported_extensions
    suffix = Path(file.filename).suffix.lower()
    if suffix not in supported_exts:
        raise HTTPException(400, f"Unsupported file type: {suffix}. Supported: {supported_exts}")

    dest = watch_dirs[0].path / file.filename
    content = await file.read()
    dest.write_bytes(content)

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
    return {"status": "ok"}


# ---------------------------------------------------------------------------
# Static files (frontend)
# ---------------------------------------------------------------------------

STATIC_DIR = Path(__file__).parent.parent.parent / "frontend"
if STATIC_DIR.exists():
    app.mount("/", StaticFiles(directory=str(STATIC_DIR), html=True), name="frontend")
