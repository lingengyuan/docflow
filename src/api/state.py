"""Runtime state shared by DocFlow API routes and services."""

from __future__ import annotations

import logging
import threading
from collections.abc import Iterator, MutableMapping
from dataclasses import dataclass, field
from pathlib import Path
from time import time
from typing import Any, Literal, Protocol, TypedDict, cast

from src.api.model_tasks import ModelTaskController

logger = logging.getLogger(__name__)

LLMSwitchStatus = Literal["idle", "switching", "error"]


class LLMSwitchSnapshot(TypedDict):
    state: LLMSwitchStatus
    model: str | None
    message: str
    started_at: float | None
    finished_at: float | None


class WatchDirLike(Protocol):
    path: Path
    recursive: bool
    extensions: list[str]


class LLMSwitchState(MutableMapping[str, Any]):
    """Thread-safe model switch status exposed as a mapping for compatibility."""

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._data: dict[str, Any] = {
            "state": "idle",
            "model": None,
            "message": "",
            "started_at": None,
            "finished_at": None,
        }

    def set(
        self,
        state: LLMSwitchStatus,
        *,
        model: str | None = None,
        message: str = "",
    ) -> None:
        now = time()
        with self._lock:
            self._data.update(
                {
                    "state": state,
                    "model": model,
                    "message": message,
                    "started_at": now if state == "switching" else self._data.get("started_at"),
                    "finished_at": None if state == "switching" else now,
                }
            )

    def snapshot(self) -> LLMSwitchSnapshot:
        with self._lock:
            return cast(LLMSwitchSnapshot, dict(self._data))

    def __getitem__(self, key: str) -> Any:
        with self._lock:
            return self._data[key]

    def __setitem__(self, key: str, value: Any) -> None:
        with self._lock:
            self._data[key] = value

    def __delitem__(self, key: str) -> None:
        with self._lock:
            del self._data[key]

    def __iter__(self) -> Iterator[str]:
        return iter(self.snapshot())

    def __len__(self) -> int:
        with self._lock:
            return len(self._data)


@dataclass
class AppContext:
    """Container for long-lived API runtime dependencies."""

    config_path: Path
    model_tasks: ModelTaskController
    pipeline: Any = None
    ingest_queue: Any = None
    query_engine: Any = None
    store: Any = None
    watcher: Any = None
    watch_dirs: list[WatchDirLike] = field(default_factory=list)
    llm_options: list[str] = field(default_factory=list)
    llm_switch_state: LLMSwitchState = field(default_factory=LLMSwitchState)

    def set_runtime(
        self,
        *,
        pipeline: Any,
        ingest_queue: Any,
        query_engine: Any,
        store: Any,
        watcher: Any,
        watch_dirs: list[WatchDirLike],
        llm_options: list[str],
    ) -> None:
        self.pipeline = pipeline
        self.ingest_queue = ingest_queue
        self.query_engine = query_engine
        self.store = store
        self.watcher = watcher
        self.watch_dirs = watch_dirs
        self.llm_options = llm_options

    def clear_runtime(self) -> None:
        self.pipeline = None
        self.ingest_queue = None
        self.query_engine = None
        self.store = None
        self.watcher = None
        self.watch_dirs = []
        self.llm_options = []


AppState = AppContext
