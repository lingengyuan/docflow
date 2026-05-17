"""Runtime access point for API handlers and lifecycle code."""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType
from typing import Any

from src.api.state import AppContext, LLMSwitchState, WatchDirLike


@dataclass
class ApiRuntime:
    app_context: AppContext
    CONFIG_PATH: Path
    MODEL_TASK_TIMEOUT_S: float
    STREAM_FIRST_CONTENT_TIMEOUT_S: float
    STREAM_IDLE_TIMEOUT_S: float
    STREAM_QUEUE_POLL_S: float
    FOREGROUND_PAUSE_GRACE_S: float
    INGEST_PAUSE_CHECK_INTERVAL_MS: int
    MODEL_TIMEOUT_MESSAGE: str
    logger: logging.Logger
    query_service: Any
    import_service: Any
    knowledge_service: Any
    health_service: Any
    shutil: ModuleType
    fetch_webpage_markdown: Callable[..., Any]
    build_quick_note_markdown: Callable[..., Any]
    build_answer_note_markdown: Callable[..., Any]
    build_knowledge_output_markdown: Callable[..., Any]
    get_knowledge_output_type: Callable[..., Any]
    knowledge_output_tags: Callable[..., Any]
    _collect_storage_usage: Callable[..., Any]
    _configured_model_cache_paths: Callable[..., Any]
    _app_data_paths: Callable[..., Any]
    _is_hf_model_cached: Callable[..., Any]
    _parse_watch_dirs: Callable[..., Any]
    _llm_model_status: Callable[..., Any]
    _load_mlx_model_candidate: Callable[..., Any]
    _set_llm_switch_state: Callable[..., Any]
    _timed_check: Callable[..., Any]
    _check_sqlite: Callable[..., Any]
    _check_qdrant: Callable[..., Any]
    _check_ollama: Callable[..., Any]
    _check_models: Callable[..., Any]
    _health_capabilities: Callable[..., Any]
    _aggregate_health_status: Callable[..., Any]
    _health_groups: Callable[..., Any]
    _health_actions: Callable[..., Any]

    @property
    def app_state(self) -> AppContext:
        return self.app_context

    @property
    def pipeline(self) -> Any:
        return self.app_context.pipeline

    @pipeline.setter
    def pipeline(self, value: Any) -> None:
        self.app_context.pipeline = value

    @property
    def ingest_queue(self) -> Any:
        return self.app_context.ingest_queue

    @ingest_queue.setter
    def ingest_queue(self, value: Any) -> None:
        self.app_context.ingest_queue = value

    @property
    def query_engine(self) -> Any:
        return self.app_context.query_engine

    @query_engine.setter
    def query_engine(self, value: Any) -> None:
        self.app_context.query_engine = value

    @property
    def store(self) -> Any:
        return self.app_context.store

    @store.setter
    def store(self, value: Any) -> None:
        self.app_context.store = value

    @property
    def watcher(self) -> Any:
        return self.app_context.watcher

    @watcher.setter
    def watcher(self, value: Any) -> None:
        self.app_context.watcher = value

    @property
    def watch_dirs(self) -> list[WatchDirLike]:
        return self.app_context.watch_dirs

    @watch_dirs.setter
    def watch_dirs(self, value: list[WatchDirLike]) -> None:
        self.app_context.watch_dirs = value

    @property
    def llm_options(self) -> list[str]:
        return self.app_context.llm_options

    @llm_options.setter
    def llm_options(self, value: list[str]) -> None:
        self.app_context.llm_options = value

    @property
    def model_tasks(self) -> Any:
        return self.app_context.model_tasks

    @property
    def llm_switch_state(self) -> LLMSwitchState:
        return self.app_context.llm_switch_state

    def _sync_app_state(self) -> None:
        return None


_runtime: ApiRuntime | None = None


def configure_api_runtime(runtime: ApiRuntime) -> None:
    global _runtime
    _runtime = runtime


def get_api_runtime() -> ApiRuntime:
    if _runtime is None:
        raise RuntimeError("API runtime has not been configured")
    return _runtime
