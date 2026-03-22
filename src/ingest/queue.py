"""
IngestQueue — 后台异步 ingest 任务队列。

将 PDF ingest 从 HTTP 请求路径中解耦：API 调用立即返回，
实际处理由后台 worker 线程按序完成。
"""

from __future__ import annotations

import logging
import threading
from pathlib import Path

logger = logging.getLogger(__name__)


class IngestQueue:
    """
    线程安全的 ingest 任务队列。

    用法：
        q = IngestQueue(pipeline)
        q.start()
        q.submit(Path("some.pdf"))   # 立即返回
        q.stop()                     # 等待当前任务完成后退出
    """

    def __init__(self, pipeline, on_done=None, ml_executor=None):
        from src.ingest.pipeline import IngestPipeline
        self.pipeline: IngestPipeline = pipeline
        self._on_done = on_done       # callable() invoked after each file is indexed
        self._ml_executor = ml_executor  # if set, embed_chunks runs in this executor
        self._queue: list[Path] = []
        self._lock = threading.Lock()
        self._event = threading.Event()
        self._current: str | None = None   # 正在处理的文件名
        self._thread = threading.Thread(target=self._worker, daemon=True, name="ingest-worker")
        self._running = False

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
        logger.info("[queue] Ingest worker stopped")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def submit(self, pdf_path: Path) -> dict:
        """将单个 PDF 加入队列。立即返回，不等待处理完成。"""
        pdf_path = Path(pdf_path)
        with self._lock:
            # 避免重复入队同一文件
            if pdf_path not in self._queue and pdf_path.name != self._current:
                self._queue.append(pdf_path)
        self._event.set()
        logger.info(f"[queue] Queued: {pdf_path.name} (queue size: {self.queue_size})")
        return {"status": "queued", "file": pdf_path.name}

    def submit_many(self, pdf_paths: list[Path]) -> dict:
        """批量入队。返回入队数量。"""
        added = 0
        with self._lock:
            for p in pdf_paths:
                p = Path(p)
                if p not in self._queue and p.name != self._current:
                    self._queue.append(p)
                    added += 1
        if added:
            self._event.set()
        logger.info(f"[queue] Queued {added} files (queue size: {self.queue_size})")
        return {"queued": added}

    @property
    def queue_size(self) -> int:
        with self._lock:
            return len(self._queue)

    def status(self) -> dict:
        with self._lock:
            return {
                "queue_size": len(self._queue),
                "processing": self._current,
                "pending_files": [p.name for p in self._queue],
            }

    # ------------------------------------------------------------------
    # Worker
    # ------------------------------------------------------------------

    def _worker(self):
        while self._running:
            self._event.wait(timeout=10)   # 定时唤醒，防止事件丢失
            self._event.clear()
            self._drain()

    def _drain(self):
        while True:
            with self._lock:
                if not self._queue:
                    self._current = None
                    break
                path = self._queue.pop(0)
                self._current = path.name

            try:
                logger.info(f"[queue] Processing: {path.name}")
                if self._ml_executor:
                    # Run ingest in the dedicated ML thread so that embedding shares
                    # the same Metal command queue as the retriever's encode() calls.
                    future = self._ml_executor.submit(self.pipeline.ingest, path)
                    result = future.result()
                else:
                    result = self.pipeline.ingest(path)
                logger.info(f"[queue] Done: {result}")
                if self._on_done:
                    self._on_done()
            except Exception:
                logger.exception(f"[queue] Failed: {path.name}")
            finally:
                with self._lock:
                    self._current = None
