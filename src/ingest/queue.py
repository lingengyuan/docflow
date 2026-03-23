"""
IngestQueue — 后台异步 ingest 任务队列。

将 ingest 从 HTTP 请求路径中解耦：API 调用立即返回，
实际处理由后台 worker 线程完成。
"""

from __future__ import annotations

import logging
import threading
import time
from collections import deque
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from pathlib import Path

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
        with self._lock:
            processing_files = [prepared.path.name for prepared in self._active_batch]
            if not processing_files:
                current = self._display_current_locked()
                processing_files = [current.name] if current else []
            return {
                "queue_size": len(self._pending_paths_locked()),
                "processing": processing_files[0] if processing_files else None,
                "processing_files": processing_files,
                "pending_files": [p.name for p in self._pending_paths_locked()],
                "progress": dict(self._progress) if self._progress else None,
                "last_completed": dict(self._last_completed) if self._last_completed else None,
            }

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
            self._event.wait(timeout=0.1)
            self._event.clear()

    def _drain_legacy(self) -> bool:
        while True:
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
            except Exception:
                logger.exception(f"[queue] Failed: {path.name}")
                with self._lock:
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
            batch = self._pop_prepared_batch()
            if batch:
                self._process_prepared_batch(batch)
                return True

        if self._should_wait_for_more_prepared():
            if self._collect_prepare_results(timeout=self._microbatch_linger_s):
                return True
            if self._should_process_batch():
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
        return False

    # ------------------------------------------------------------------
    # Prepared pipeline helpers
    # ------------------------------------------------------------------

    def _schedule_prepare_tasks(self) -> bool:
        if self._prepare_executor is None:
            return False

        scheduled = False
        while True:
            with self._lock:
                if len(self._prepare_futures) >= self._parse_workers or not self._queue:
                    self._refresh_progress_locked()
                    return scheduled
                path = self._queue.popleft()
                future = self._prepare_executor.submit(self.pipeline.prepare_file, path)
                self._prepare_futures[future] = path
                if self._prepared:
                    self._prepared_ready_at = time.monotonic()
                self._refresh_progress_locked()
                scheduled = True

    def _collect_prepare_results(self, timeout: float) -> bool:
        with self._lock:
            futures = list(self._prepare_futures.keys())
        if not futures:
            return False

        done, _ = wait(futures, timeout=timeout, return_when=FIRST_COMPLETED)
        if not done:
            return False

        with self._lock:
            done = [future for future in list(self._prepare_futures.keys()) if future.done()]

        handled_any = False
        for future in done:
            with self._lock:
                path = self._prepare_futures.pop(future, None)
            if path is None:
                continue

            handled_any = True
            try:
                result = future.result()
                if isinstance(result, PreparedIngestFile):
                    with self._lock:
                        self._prepared.append(result)
                        if self._prepared_ready_at is None:
                            self._prepared_ready_at = time.monotonic()
                        self._refresh_progress_locked()
                else:
                    logger.info(f"[queue] Done: {result}")
                    if self._on_done:
                        self._on_done()
                    with self._lock:
                        self._last_completed = dict(result)
                        self._tracked_paths.discard(path)
                        self._refresh_progress_locked()
            except Exception as e:
                logger.exception(f"[queue] Failed during prepare: {path.name}")
                with self._lock:
                    self._tracked_paths.discard(path)
                    self._last_completed = {
                        "status": "error",
                        "file": path.name,
                        "error": str(e),
                    }
                    self._refresh_progress_locked()

        return handled_any

    def _process_prepared_batch(self, batch: list[PreparedIngestFile]):
        total_chunks = sum(len(prepared.chunks) for prepared in batch)
        batch_files = [prepared.path.name for prepared in batch]
        logger.info(
            "[queue] Embedding batch: %d file(s), %d chunk(s): %s",
            len(batch),
            total_chunks,
            ", ".join(batch_files),
        )

        with self._lock:
            self._active_batch = batch
            self._progress = self._make_progress_locked(
                stage="embedding",
                current_path=batch[0].path if batch else None,
                processed_chunks=0,
                total_chunks=total_chunks,
                batch_files=batch_files,
            )

        def _on_progress(update: dict):
            with self._lock:
                if not self._active_batch:
                    return
                progress = dict(self._progress or {})
                progress.update(update)
                progress.setdefault("batch_files", batch_files)
                progress.setdefault("batch_size", len(batch_files))
                if "current_file" not in progress and batch:
                    progress["current_file"] = batch[0].path.name
                if "current_path" not in progress and batch:
                    progress["current_path"] = str(batch[0].path)
                progress["updated_at"] = time.time()
                self._progress = progress

        try:
            results = self.pipeline.ingest_prepared_batch(batch, progress_callback=_on_progress)
        except Exception as e:
            logger.exception("[queue] Failed batch")
            results = [
                {"status": "error", "file": prepared.path.name, "error": str(e)}
                for prepared in batch
            ]
        if len(results) < len(batch):
            results = list(results) + [
                {"status": "error", "file": prepared.path.name, "error": "Missing batch result"}
                for prepared in batch[len(results):]
            ]

        for prepared, result in zip(batch, results):
            logger.info(f"[queue] Done: {result}")
            if self._on_done:
                self._on_done()
            with self._lock:
                self._last_completed = dict(result)
                self._tracked_paths.discard(prepared.path)

        with self._lock:
            self._active_batch = []
            self._refresh_progress_locked()

    def _should_wait_for_more_prepared(self) -> bool:
        with self._lock:
            if not self._prepared:
                return False
            if self._reached_microbatch_limit_locked():
                return False
            ready_at = self._prepared_ready_at
        if ready_at is None:
            return False
        return time.monotonic() - ready_at < self._microbatch_linger_s

    def _should_process_batch(self) -> bool:
        with self._lock:
            if not self._prepared:
                return False
            if self._reached_microbatch_limit_locked():
                return True
            if self._prepared_ready_at is None:
                return False
            if not (self._prepare_futures or self._queue):
                return time.monotonic() - self._prepared_ready_at >= self._microbatch_linger_s
            return time.monotonic() - self._prepared_ready_at >= self._microbatch_linger_s

    def _pop_prepared_batch(self) -> list[PreparedIngestFile]:
        with self._lock:
            batch: list[PreparedIngestFile] = []
            chunk_total = 0
            while self._prepared:
                candidate = self._prepared[0]
                next_chunk_total = chunk_total + len(candidate.chunks)
                if batch and (
                    len(batch) >= self._microbatch_max_files
                    or next_chunk_total > self._microbatch_max_chunks
                ):
                    break
                batch.append(self._prepared.popleft())
                chunk_total = next_chunk_total
                if len(batch) >= self._microbatch_max_files or chunk_total >= self._microbatch_max_chunks:
                    break
            self._prepared_ready_at = time.monotonic() if self._prepared else None
            return batch

    def _has_outstanding_preparation(self) -> bool:
        with self._lock:
            return bool(self._prepare_futures)

    def _reached_microbatch_limit_locked(self) -> bool:
        if not self._prepared:
            return False
        chunk_total = sum(len(prepared.chunks) for prepared in self._prepared)
        return (
            len(self._prepared) >= self._microbatch_max_files
            or chunk_total >= self._microbatch_max_chunks
        )

    # ------------------------------------------------------------------
    # Status helpers
    # ------------------------------------------------------------------

    def _display_current_locked(self) -> Path | None:
        if self._current is not None:
            return self._current
        if self._active_batch:
            return self._active_batch[0].path
        if self._prepared:
            return self._prepared[0].path
        if self._prepare_futures:
            return next(iter(self._prepare_futures.values()))
        if self._queue:
            return self._queue[0]
        return None

    def _pending_paths_locked(self) -> list[Path]:
        active_paths = {prepared.path for prepared in self._active_batch}
        pending: list[Path] = []
        seen: set[Path] = set()
        for path in list(self._prepare_futures.values()) + [prepared.path for prepared in self._prepared] + list(self._queue):
            if path in active_paths or path in seen:
                continue
            seen.add(path)
            pending.append(path)
        return pending

    def _make_progress_locked(
        self,
        stage: str,
        current_path: Path | None,
        processed_chunks: int,
        total_chunks: int,
        batch_files: list[str] | None = None,
    ) -> dict:
        return {
            "stage": stage,
            "current_file": current_path.name if current_path else None,
            "current_path": str(current_path) if current_path else None,
            "processed_chunks": processed_chunks,
            "total_chunks": total_chunks,
            "batch_files": batch_files or ([current_path.name] if current_path else []),
            "batch_size": len(batch_files or ([current_path.name] if current_path else [])),
            "cache_hits": 0,
            "cache_misses": max(0, total_chunks),
            "adaptive_batch_size": None,
            "updated_at": time.time(),
        }

    def _refresh_progress_locked(self):
        if self._active_batch:
            return
        current = self._display_current_locked()
        if current is None:
            self._progress = None
            return
        stage = "preparing" if (self._prepare_futures or self._prepared) else "queued"
        self._progress = self._make_progress_locked(
            stage=stage,
            current_path=current,
            processed_chunks=0,
            total_chunks=0,
        )
