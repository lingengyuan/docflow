"""
DocFlow FastAPI 后端。
"""

from __future__ import annotations

import logging
import os
import shutil
import sys
import types
from pathlib import Path
from typing import TYPE_CHECKING

from fastapi import FastAPI
from fastapi.responses import FileResponse, Response
from fastapi.staticfiles import StaticFiles

from src.api.handlers.import_handlers import (
    create_demo_library,
    create_knowledge_output,
    create_note,
    import_url,
    save_answer_note,
    upload_file,
)
from src.api.handlers.library_handlers import (
    batch_favorite,
    batch_rebuild_files,
    batch_update_file_metadata,
    clear_history,
    knowledge_overview,
    library_meta,
    list_favorites,
    list_file_chunks,
    list_files,
    list_history,
    preview_file,
    preview_file_head,
    queue_status,
    search_history,
    storage_usage,
    summarize_files,
    toggle_favorite,
    trigger_ingest,
    update_file_metadata,
)
from src.api.handlers.maintenance_handlers import debug_retrieve
from src.api.handlers.obsidian_handlers import obsidian_related_notes
from src.api.handlers.query_handlers import (
    answer_feedback,
    create_conversation,
    delete_conversation,
    list_conversation_messages,
    list_conversations,
    query,
    query_stream,
    research,
)
from src.api.handlers.settings_handlers import get_llm, health, list_sources, set_llm
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
from src.api.lifecycle import lifespan
from src.api.model_tasks import ModelTaskController
from src.api.routes import imports as imports_routes
from src.api.routes import knowledge as knowledge_routes
from src.api.routes import library as library_routes
from src.api.routes import maintenance as maintenance_routes
from src.api.routes import obsidian as obsidian_routes
from src.api.routes import query as query_routes
from src.api.routes import settings as settings_routes
from src.api.runtime_helpers import (
    _app_data_paths,
    _collect_storage_usage,
    _config_path,
    _configured_model_cache_paths,
    _configured_model_names,
    _hf_cache_dir,
    _is_hf_model_cached,
    _llm_model_status,
    _load_mlx_model_candidate,
    _parse_watch_dirs,
    _safe_path_size,
    _set_llm_switch_state,
    _source_file_usage,
    _unique_existing_paths,
)
from src.api.services.health_service import HealthService
from src.api.services.import_service import ImportService
from src.api.services.knowledge_service import KnowledgeService
from src.api.services.query_service import QueryService
from src.api.state import AppState
from src.ingest.imports import (
    build_answer_note_markdown,
    build_knowledge_output_markdown,
    build_quick_note_markdown,
    fetch_webpage_markdown,
)
from src.knowledge_outputs import get_knowledge_output_type, knowledge_output_tags
from src.maintenance.startup import ensure_config_file

if TYPE_CHECKING:
    from src.ingest.pipeline import IngestPipeline
    from src.ingest.queue import IngestQueue
    from src.ingest.store import DocStore
    from src.ingest.watcher import FolderWatcher, WatchDir
    from src.query.engine import QueryEngine

__all__ = [
    "app",
    "_aggregate_health_status",
    "_app_data_paths",
    "_check_models",
    "_check_ollama",
    "_check_qdrant",
    "_check_sqlite",
    "_collect_storage_usage",
    "_config_path",
    "_configured_model_cache_paths",
    "_configured_model_names",
    "_health_actions",
    "_health_capabilities",
    "_health_groups",
    "_hf_cache_dir",
    "_is_hf_model_cached",
    "_llm_model_status",
    "_load_mlx_model_candidate",
    "_parse_watch_dirs",
    "_safe_path_size",
    "_set_llm_switch_state",
    "_source_file_usage",
    "_timed_check",
    "_unique_existing_paths",
    "build_answer_note_markdown",
    "build_knowledge_output_markdown",
    "build_quick_note_markdown",
    "fetch_webpage_markdown",
    "get_knowledge_output_type",
    "knowledge_output_tags",
    "shutil",
]

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


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------

app = FastAPI(title="DocFlow", lifespan=lifespan)


# ---------------------------------------------------------------------------
# API handlers
# ---------------------------------------------------------------------------


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
