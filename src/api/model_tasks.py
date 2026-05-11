"""Bounded execution for foreground model-backed API tasks."""

from __future__ import annotations

import asyncio
import logging
import threading
import uuid
from collections.abc import Callable
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from time import perf_counter, time
from typing import TypeVar

T = TypeVar("T")


class ModelTaskTimeout(TimeoutError):
    """Raised when a model task exceeds its API-facing time budget."""

    def __init__(self, task_id: str, name: str, timeout_s: float):
        super().__init__(f"model task {name} timed out after {timeout_s:.1f}s")
        self.task_id = task_id
        self.name = name
        self.timeout_s = timeout_s


@dataclass(frozen=True)
class SubmittedModelTask:
    task_id: str
    name: str
    future: Future
    started_at: float


class ModelTaskController:
    """Owns the single foreground model worker and can retire it after hangs."""

    def __init__(
        self,
        *,
        max_workers: int = 1,
        thread_name_prefix: str = "ml-inference",
        logger: logging.Logger | None = None,
    ):
        self._max_workers = max_workers
        self._thread_name_prefix = thread_name_prefix
        self._logger = logger or logging.getLogger(__name__)
        self._lock = threading.Lock()
        self._generation = 0
        self._foreground_count = 0
        self._last_started_at: float | None = None
        self._last_finished_at: float | None = None
        self._executor = self._new_executor()

    def _new_executor(self) -> ThreadPoolExecutor:
        return ThreadPoolExecutor(
            max_workers=self._max_workers,
            thread_name_prefix=f"{self._thread_name_prefix}-{self._generation}",
        )

    def submit(self, name: str, fn: Callable[[], T]) -> SubmittedModelTask:
        task_id = uuid.uuid4().hex[:12]
        started_at = perf_counter()
        self._mark_submitted()

        def _wrapped() -> T:
            self._logger.info("[model-task] start id=%s name=%s", task_id, name)
            try:
                return fn()
            finally:
                duration_ms = round((perf_counter() - started_at) * 1000, 2)
                self._logger.info(
                    "[model-task] finish id=%s name=%s duration_ms=%.2f",
                    task_id,
                    name,
                    duration_ms,
                )
                self._mark_finished()

        with self._lock:
            executor = self._executor
        try:
            future = executor.submit(_wrapped)
        except Exception:
            self._mark_finished()
            raise
        future.add_done_callback(self._mark_cancelled)
        return SubmittedModelTask(task_id=task_id, name=name, future=future, started_at=started_at)

    async def run(self, name: str, fn: Callable[[], T], *, timeout_s: float) -> T:
        task = self.submit(name, fn)
        try:
            return await asyncio.wait_for(asyncio.wrap_future(task.future), timeout=timeout_s)
        except TimeoutError as exc:
            task.future.cancel()
            self.retire(reason=f"timeout after {timeout_s:.1f}s", task=task)
            raise ModelTaskTimeout(task.task_id, name, timeout_s) from exc

    def retire(self, *, reason: str, task: SubmittedModelTask | None = None) -> None:
        with self._lock:
            old_executor = self._executor
            self._generation += 1
            self._executor = self._new_executor()
        task_ref = f" id={task.task_id} name={task.name}" if task else ""
        self._logger.warning("[model-task] retire executor%s reason=%s", task_ref, reason)
        old_executor.shutdown(wait=False, cancel_futures=True)

    def cancel_and_retire(self, task: SubmittedModelTask, *, reason: str) -> None:
        task.future.cancel()
        if not task.future.done():
            self.retire(reason=reason, task=task)
        else:
            self._logger.info(
                "[model-task] cancel id=%s name=%s reason=%s",
                task.task_id,
                task.name,
                reason,
            )

    def shutdown(self) -> None:
        with self._lock:
            executor = self._executor
            self._generation += 1
            self._executor = self._new_executor()
        executor.shutdown(wait=False, cancel_futures=True)

    def is_foreground_active(self, *, grace_s: float = 0.0) -> bool:
        with self._lock:
            if self._foreground_count > 0:
                return True
            last_finished_at = self._last_finished_at
        if grace_s <= 0 or last_finished_at is None:
            return False
        return time() - last_finished_at < grace_s

    def status(self) -> dict:
        with self._lock:
            return {
                "foreground_active": self._foreground_count > 0,
                "foreground_count": self._foreground_count,
                "last_started_at": self._last_started_at,
                "last_finished_at": self._last_finished_at,
            }

    def _mark_submitted(self) -> None:
        with self._lock:
            self._foreground_count += 1
            self._last_started_at = time()

    def _mark_finished(self) -> None:
        with self._lock:
            self._foreground_count = max(0, self._foreground_count - 1)
            self._last_finished_at = time()

    def _mark_cancelled(self, future: Future) -> None:
        if future.cancelled():
            self._mark_finished()
