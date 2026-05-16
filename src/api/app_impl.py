"""
DocFlow FastAPI 后端。
"""

from __future__ import annotations

import logging
import os
import shutil
import sys
import types

from fastapi import FastAPI

from src.api.app_routes import register_api_routes
from src.api.app_static import STATIC_DIR as STATIC_DIR
from src.api.app_static import mount_static_frontend
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
from src.api.runtime import ApiRuntime, configure_api_runtime, get_api_runtime
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
from src.api.state import AppContext
from src.ingest.imports import (
    build_answer_note_markdown,
    build_knowledge_output_markdown,
    build_quick_note_markdown,
    fetch_webpage_markdown,
)
from src.knowledge_outputs import get_knowledge_output_type, knowledge_output_tags
from src.maintenance.startup import ensure_config_file

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

CONFIG_PATH = ensure_config_file(os.getenv("DOCFLOW_CONFIG", "config.yaml"))
COLLECTION_NAME = "docflow"
MODEL_TASK_TIMEOUT_S = float(os.getenv("DOCFLOW_MODEL_TASK_TIMEOUT_S", "90"))
STREAM_FIRST_CONTENT_TIMEOUT_S = float(os.getenv("DOCFLOW_STREAM_FIRST_CONTENT_TIMEOUT_S", "60"))
STREAM_IDLE_TIMEOUT_S = float(os.getenv("DOCFLOW_STREAM_IDLE_TIMEOUT_S", "60"))
STREAM_QUEUE_POLL_S = 0.25
FOREGROUND_PAUSE_GRACE_S = float(os.getenv("DOCFLOW_FOREGROUND_PAUSE_GRACE_S", "5"))
INGEST_PAUSE_CHECK_INTERVAL_MS = int(os.getenv("DOCFLOW_INGEST_PAUSE_CHECK_INTERVAL_MS", "500"))
MODEL_TIMEOUT_MESSAGE = "模型任务超时，请稍后重试；系统已释放后续请求。"


# ---------------------------------------------------------------------------
# Application context (initialized in lifespan)
# ---------------------------------------------------------------------------

_initial_model_tasks = ModelTaskController(thread_name_prefix="ml-inference", logger=logger)
app_context = AppContext(config_path=CONFIG_PATH, model_tasks=_initial_model_tasks)
app_state = app_context
del _initial_model_tasks
query_service = QueryService()
import_service = ImportService()
knowledge_service = KnowledgeService()
health_service = HealthService()
configure_health_checks(store_getter=lambda: app_context.store)
configure_api_runtime(
    ApiRuntime(
        app_context=app_context,
        CONFIG_PATH=CONFIG_PATH,
        MODEL_TASK_TIMEOUT_S=MODEL_TASK_TIMEOUT_S,
        STREAM_FIRST_CONTENT_TIMEOUT_S=STREAM_FIRST_CONTENT_TIMEOUT_S,
        STREAM_IDLE_TIMEOUT_S=STREAM_IDLE_TIMEOUT_S,
        STREAM_QUEUE_POLL_S=STREAM_QUEUE_POLL_S,
        FOREGROUND_PAUSE_GRACE_S=FOREGROUND_PAUSE_GRACE_S,
        INGEST_PAUSE_CHECK_INTERVAL_MS=INGEST_PAUSE_CHECK_INTERVAL_MS,
        MODEL_TIMEOUT_MESSAGE=MODEL_TIMEOUT_MESSAGE,
        logger=logger,
        query_service=query_service,
        import_service=import_service,
        knowledge_service=knowledge_service,
        health_service=health_service,
        shutil=shutil,
        fetch_webpage_markdown=fetch_webpage_markdown,
        build_quick_note_markdown=build_quick_note_markdown,
        build_answer_note_markdown=build_answer_note_markdown,
        build_knowledge_output_markdown=build_knowledge_output_markdown,
        get_knowledge_output_type=get_knowledge_output_type,
        knowledge_output_tags=knowledge_output_tags,
        _collect_storage_usage=_collect_storage_usage,
        _configured_model_cache_paths=_configured_model_cache_paths,
        _app_data_paths=_app_data_paths,
        _is_hf_model_cached=_is_hf_model_cached,
        _parse_watch_dirs=_parse_watch_dirs,
        _llm_model_status=_llm_model_status,
        _load_mlx_model_candidate=_load_mlx_model_candidate,
        _set_llm_switch_state=_set_llm_switch_state,
        _timed_check=_timed_check,
        _check_sqlite=_check_sqlite,
        _check_qdrant=_check_qdrant,
        _check_ollama=_check_ollama,
        _check_models=_check_models,
        _health_capabilities=_health_capabilities,
        _aggregate_health_status=_aggregate_health_status,
        _health_groups=_health_groups,
        _health_actions=_health_actions,
    )
)


def _sync_app_state() -> None:
    return None


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------

app = FastAPI(title="DocFlow", lifespan=lifespan)


register_api_routes(app)
mount_static_frontend(app)


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
            state = types.ModuleType.__getattribute__(self, "app_context")
            return getattr(state, name)
        return types.ModuleType.__getattribute__(self, name)

    def __setattr__(self, name: str, value):
        if name in _STATE_FIELD_NAMES:
            state = types.ModuleType.__getattribute__(self, "app_context")
            setattr(state, name, value)
            return
        try:
            runtime = get_api_runtime()
        except RuntimeError:
            runtime = None
        if runtime is not None and hasattr(runtime, name):
            setattr(runtime, name, value)
        types.ModuleType.__setattr__(self, name, value)


sys.modules[__name__].__class__ = _ApiModule
