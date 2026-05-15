"""Prepared-mode helpers for ingest queue."""

from __future__ import annotations

import logging
import time
from concurrent.futures import FIRST_COMPLETED, wait

from src.ingest.pipeline import PreparedIngestFile

logger = logging.getLogger(__name__)


def schedule_prepare_tasks(queue) -> bool:
    if queue._prepare_executor is None:
        return False

    scheduled = False
    while True:
        with queue._lock:
            if len(queue._prepare_futures) >= queue._parse_workers or not queue._queue:
                queue._refresh_progress_locked()
                return scheduled
            path = queue._queue.popleft()
            future = queue._prepare_executor.submit(queue.pipeline.prepare_file, path)
            queue._prepare_futures[future] = path
            if queue._prepared:
                queue._prepared_ready_at = time.monotonic()
            queue._refresh_progress_locked()
            scheduled = True


def collect_prepare_results(queue, timeout: float) -> bool:
    with queue._lock:
        futures = list(queue._prepare_futures.keys())
    if not futures:
        return False

    done, _ = wait(futures, timeout=timeout, return_when=FIRST_COMPLETED)
    if not done:
        return False

    with queue._lock:
        completed = [future for future in list(queue._prepare_futures.keys()) if future.done()]

    handled_any = False
    for future in completed:
        with queue._lock:
            path = queue._prepare_futures.pop(future, None)
        if path is None:
            continue

        handled_any = True
        try:
            result = future.result()
            if isinstance(result, PreparedIngestFile):
                with queue._lock:
                    queue._prepared.append(result)
                    if queue._prepared_ready_at is None:
                        queue._prepared_ready_at = time.monotonic()
                    queue._refresh_progress_locked()
            else:
                logger.info(f"[queue] Done: {result}")
                if queue._on_done:
                    queue._on_done()
                with queue._lock:
                    queue._last_completed = dict(result)
                    queue._tracked_paths.discard(path)
                    queue._refresh_progress_locked()
        except Exception as e:
            logger.exception(f"[queue] Failed during prepare: {path.name}")
            with queue._lock:
                queue._tracked_paths.discard(path)
                queue._last_completed = {
                    "status": "error",
                    "file": path.name,
                    "error": str(e),
                }
                queue._refresh_progress_locked()

    return handled_any


def process_prepared_batch(queue, batch: list[PreparedIngestFile]):
    total_chunks = sum(len(prepared.chunks) for prepared in batch)
    batch_files = [prepared.path.name for prepared in batch]
    logger.info(
        "[queue] Embedding batch: %d file(s), %d chunk(s): %s",
        len(batch),
        total_chunks,
        ", ".join(batch_files),
    )

    with queue._lock:
        queue._active_batch = batch
        queue._progress = queue._make_progress_locked(
            stage="embedding",
            current_path=batch[0].path if batch else None,
            processed_chunks=0,
            total_chunks=total_chunks,
            batch_files=batch_files,
        )

    def _on_progress(update: dict):
        with queue._lock:
            if not queue._active_batch:
                return
            progress = dict(queue._progress or {})
            progress.update(update)
            progress.setdefault("batch_files", batch_files)
            progress.setdefault("batch_size", len(batch_files))
            if "current_file" not in progress and batch:
                progress["current_file"] = batch[0].path.name
            if "current_path" not in progress and batch:
                progress["current_path"] = str(batch[0].path)
            progress["updated_at"] = time.time()
            queue._progress = progress

    try:
        results = queue.pipeline.ingest_prepared_batch(batch, progress_callback=_on_progress)
    except Exception as e:
        logger.exception("[queue] Failed batch")
        results = [
            {"status": "error", "file": prepared.path.name, "error": str(e)}
            for prepared in batch
        ]
    if len(results) < len(batch):
        results = list(results) + [
            {"status": "error", "file": prepared.path.name, "error": "Missing batch result"}
            for prepared in batch[len(results) :]
        ]

    for prepared, result in zip(batch, results, strict=False):
        logger.info(f"[queue] Done: {result}")
        if queue._on_done:
            queue._on_done()
        with queue._lock:
            queue._last_completed = dict(result)
            queue._tracked_paths.discard(prepared.path)

    with queue._lock:
        queue._active_batch = []
        queue._refresh_progress_locked()


def should_wait_for_more_prepared(queue) -> bool:
    with queue._lock:
        if not queue._prepared:
            return False
        if reached_microbatch_limit_locked(queue):
            return False
        ready_at = queue._prepared_ready_at
    if ready_at is None:
        return False
    return time.monotonic() - ready_at < queue._microbatch_linger_s


def should_process_batch(queue) -> bool:
    with queue._lock:
        if not queue._prepared:
            return False
        if reached_microbatch_limit_locked(queue):
            return True
        if queue._prepared_ready_at is None:
            return False
        if not (queue._prepare_futures or queue._queue):
            return time.monotonic() - queue._prepared_ready_at >= queue._microbatch_linger_s
        return time.monotonic() - queue._prepared_ready_at >= queue._microbatch_linger_s


def pop_prepared_batch(queue) -> list[PreparedIngestFile]:
    with queue._lock:
        batch: list[PreparedIngestFile] = []
        chunk_total = 0
        while queue._prepared:
            candidate = queue._prepared[0]
            next_chunk_total = chunk_total + len(candidate.chunks)
            if batch and (
                len(batch) >= queue._microbatch_max_files
                or next_chunk_total > queue._microbatch_max_chunks
            ):
                break
            batch.append(queue._prepared.popleft())
            chunk_total = next_chunk_total
            if (
                len(batch) >= queue._microbatch_max_files
                or chunk_total >= queue._microbatch_max_chunks
            ):
                break
        queue._prepared_ready_at = time.monotonic() if queue._prepared else None
        return batch


def has_outstanding_preparation(queue) -> bool:
    with queue._lock:
        return bool(queue._prepare_futures)


def reached_microbatch_limit_locked(queue) -> bool:
    if not queue._prepared:
        return False
    chunk_total = sum(len(prepared.chunks) for prepared in queue._prepared)
    return (
        len(queue._prepared) >= queue._microbatch_max_files
        or chunk_total >= queue._microbatch_max_chunks
    )
