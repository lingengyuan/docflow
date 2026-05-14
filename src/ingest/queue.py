"""
IngestQueue — 后台异步 ingest 任务队列。

将 ingest 从 HTTP 请求路径中解耦：API 调用立即返回，
实际处理由后台 worker 线程完成。
"""

from __future__ import annotations

import logging
import threading
from collections import deque
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from src.ingest import queue_prepared, queue_status
from src.ingest.pipeline import PreparedIngestFile

logger = logging.getLogger(__name__)


class IngestQueue:
    """
    线程安全的 ingest 任务队列。

    默认使用两阶段流水线：
      1. 预处理（parse / chunk）可并行
      2. embedding / qdrant / sqlite 串行微批执行
    """

    def __init__(
        self,
        pipeline,
        on_done=None,
        ml_executor=None,
        parse_workers: int = 2,
        microbatch_max_files: int = 8,
        microbatch_max_chunks: int = 128,
        microbatch_linger_ms: int = 75,
        should_pause_background: Callable[[], bool] | None = None,
        pause_check_interval_ms: int = 500,
    ):
        from src.ingest.pipeline import IngestPipeline

        self.pipeline: IngestPipeline = pipeline
        self._on_done = on_done
        self._ml_executor = ml_executor
        self._queue: deque[Path] = deque()
        self._tracked_paths: set[Path] = set()
        self._lock = threading.Lock()
        self._event = threading.Event()
        self._current: Path | None = None
        self._thread = threading.Thread(target=self._worker, daemon=True, name="ingest-worker")
        self._running = False

        self._prepared_mode = (
            hasattr(self.pipeline, "prepare_file")
            and hasattr(self.pipeline, "ingest_prepared_batch")
            and self._ml_executor is None
        )
        self._parse_workers = max(1, parse_workers)
        self._microbatch_max_files = max(1, microbatch_max_files)
        self._microbatch_max_chunks = max(1, microbatch_max_chunks)
        self._microbatch_linger_s = max(0.0, microbatch_linger_ms / 1000.0)
        self._should_pause_background = should_pause_background
        self._pause_check_s = max(0.05, pause_check_interval_ms / 1000.0)
        self._paused_reason: str | None = None
        self._paused_since: float | None = None

        self._prepare_executor = (
            ThreadPoolExecutor(
                max_workers=self._parse_workers,
                thread_name_prefix="ingest-prepare",
            )
            if self._prepared_mode
            else None
        )
        self._prepare_futures: dict = {}
        self._prepared: deque[PreparedIngestFile] = deque()
        self._prepared_ready_at: float | None = None
        self._active_batch: list[PreparedIngestFile] = []
        self._progress: dict | None = None
        self._last_completed: dict | None = None

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def start(self):
        self._running = True
        self._thread.start()
        logger.info("[queue] Ingest worker started")

    def stop(self):
        self._running = False
        self._event.set()
        self._thread.join(timeout=5)
        if self._prepare_executor is not None:
            self._prepare_executor.shutdown(wait=False, cancel_futures=True)
        logger.info("[queue] Ingest worker stopped")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def submit(self, pdf_path: Path) -> dict:
        """将单个文件加入队列。立即返回，不等待处理完成。"""
        pdf_path = Path(pdf_path).expanduser().resolve()
        added = False
        with self._lock:
            if pdf_path not in self._tracked_paths:
                self._tracked_paths.add(pdf_path)
                self._queue.append(pdf_path)
                added = True
        self._event.set()
        status = "queued" if added else "duplicate"
        logger.info(f"[queue] {status}: {pdf_path.name} (queue size: {self.queue_size})")
        return {"status": status, "file": pdf_path.name}

    def submit_many(self, pdf_paths: list[Path]) -> dict:
        """批量入队。返回入队数量。"""
        added = 0
        with self._lock:
            for p in pdf_paths:
                path = Path(p).expanduser().resolve()
                if path in self._tracked_paths:
                    continue
                self._tracked_paths.add(path)
                self._queue.append(path)
                added += 1
        if added:
            self._event.set()
        logger.info(f"[queue] Queued {added} files (queue size: {self.queue_size})")
        return {"queued": added}

    @property
    def queue_size(self) -> int:
        with self._lock:
            return len(self._pending_paths_locked())

    def status(self) -> dict:
        return queue_status.status(self)

    # ------------------------------------------------------------------
    # Worker
    # ------------------------------------------------------------------

    def _worker(self):
        while self._running:
            if self._prepared_mode:
                did_work = self._drain_prepared()
            else:
                did_work = self._drain_legacy()
            if did_work:
                continue
            timeout = self._pause_check_s if self._is_marked_paused() else 0.1
            self._event.wait(timeout=timeout)
            self._event.clear()

    def _drain_legacy(self) -> bool:
        while True:
            if self._is_background_paused():
                self._mark_paused("foreground_active")
                return False
            self._clear_pause()
            with self._lock:
                if not self._queue:
                    self._current = None
                    self._progress = None
                    return False
                path = self._queue.popleft()
                self._current = path
                self._progress = self._make_progress_locked(
                    stage="processing",
                    current_path=path,
                    processed_chunks=0,
                    total_chunks=0,
                )

            try:
                logger.info(f"[queue] Processing: {path.name}")
                if self._ml_executor:
                    future = self._ml_executor.submit(self.pipeline.ingest, path)
                    result = future.result()
                else:
                    result = self.pipeline.ingest(path)
                logger.info(f"[queue] Done: {result}")
                if self._on_done:
                    self._on_done()
                with self._lock:
                    self._last_completed = dict(result)
                    self._tracked_paths.discard(path)
            except Exception as e:
                logger.exception(f"[queue] Failed: {path.name}")
                with self._lock:
                    self._last_completed = {
                        "status": "error",
                        "file": path.name,
                        "error": str(e),
                    }
                    self._tracked_paths.discard(path)
            finally:
                with self._lock:
                    self._current = None
                    self._refresh_progress_locked()
            return True

    def _drain_prepared(self) -> bool:
        scheduled = self._schedule_prepare_tasks()

        if self._collect_prepare_results(timeout=0):
            return True

        if self._should_process_batch():
            if self._is_background_paused():
                self._mark_paused("foreground_active")
                return False
            self._clear_pause()
            batch = self._pop_prepared_batch()
            if batch:
                self._process_prepared_batch(batch)
                return True

        if self._should_wait_for_more_prepared():
            if self._collect_prepare_results(timeout=self._microbatch_linger_s):
                return True
            if self._should_process_batch():
                if self._is_background_paused():
                    self._mark_paused("foreground_active")
                    return False
                self._clear_pause()
                batch = self._pop_prepared_batch()
                if batch:
                    self._process_prepared_batch(batch)
                    return True

        if scheduled:
            return True

        if self._has_outstanding_preparation():
            return self._collect_prepare_results(timeout=0.05)

        with self._lock:
            if not self._tracked_paths:
                self._progress = None
                self._paused_reason = None
                self._paused_since = None
        return False

    # ------------------------------------------------------------------
    # Prepared pipeline helpers
    # ------------------------------------------------------------------

    def _schedule_prepare_tasks(self) -> bool:
        return queue_prepared.schedule_prepare_tasks(self)

    def _collect_prepare_results(self, timeout: float) -> bool:
        return queue_prepared.collect_prepare_results(self, timeout)

    def _process_prepared_batch(self, batch: list[PreparedIngestFile]):
        queue_prepared.process_prepared_batch(self, batch)

    def _should_wait_for_more_prepared(self) -> bool:
        return queue_prepared.should_wait_for_more_prepared(self)

    def _should_process_batch(self) -> bool:
        return queue_prepared.should_process_batch(self)

    def _pop_prepared_batch(self) -> list[PreparedIngestFile]:
        return queue_prepared.pop_prepared_batch(self)

    def _has_outstanding_preparation(self) -> bool:
        return queue_prepared.has_outstanding_preparation(self)

    def _reached_microbatch_limit_locked(self) -> bool:
        return queue_prepared.reached_microbatch_limit_locked(self)

    def _is_background_paused(self) -> bool:
        return queue_status.is_background_paused(self)

    def _mark_paused(self, reason: str):
        queue_status.mark_paused(self, reason)

    def _clear_pause(self):
        queue_status.clear_pause(self)

    def _is_marked_paused(self) -> bool:
        return queue_status.is_marked_paused(self)

    # ------------------------------------------------------------------
    # Status helpers
    # ------------------------------------------------------------------

    def _display_current_locked(self) -> Path | None:
        return queue_status.display_current_locked(self)

    def _pending_paths_locked(self) -> list[Path]:
        return queue_status.pending_paths_locked(self)

    def _make_progress_locked(
        self,
        stage: str,
        current_path: Path | None,
        processed_chunks: int,
        total_chunks: int,
        batch_files: list[str] | None = None,
    ) -> dict:
        return queue_status.make_progress_locked(
            self,
            stage,
            current_path,
            processed_chunks,
            total_chunks,
            batch_files=batch_files,
        )

    def _refresh_progress_locked(self):
        queue_status.refresh_progress_locked(self)
