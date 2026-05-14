"""Status and pause helpers for ingest queue."""

from __future__ import annotations

import logging
import time
from pathlib import Path

logger = logging.getLogger(__name__)


def status(queue) -> dict:
    with queue._lock:
        processing_files = [prepared.path.name for prepared in queue._active_batch]
        if not processing_files:
            current = display_current_locked(queue)
            processing_files = [current.name] if current else []
        return {
            "queue_size": len(pending_paths_locked(queue)),
            "processing": processing_files[0] if processing_files else None,
            "processing_files": processing_files,
            "pending_files": [p.name for p in pending_paths_locked(queue)],
            "progress": dict(queue._progress) if queue._progress else None,
            "last_completed": dict(queue._last_completed) if queue._last_completed else None,
            "paused": queue._paused_reason is not None,
            "pause_reason": queue._paused_reason,
            "paused_since": queue._paused_since,
        }


def is_background_paused(queue) -> bool:
    if queue._should_pause_background is None:
        return False
    try:
        return bool(queue._should_pause_background())
    except Exception as exc:
        logger.exception("[queue] Foreground pause check failed: %s", exc)
        return False


def mark_paused(queue, reason: str):
    with queue._lock:
        if queue._paused_reason is None:
            queue._paused_since = time.time()
            logger.info("[queue] Paused background ingest: %s", reason)
        queue._paused_reason = reason
        queue._refresh_progress_locked()
        if queue._progress is not None:
            queue._progress["stage"] = "paused"
            queue._progress["pause_reason"] = reason
            queue._progress["paused_since"] = queue._paused_since


def clear_pause(queue):
    with queue._lock:
        if queue._paused_reason is not None:
            logger.info("[queue] Resumed background ingest")
        queue._paused_reason = None
        queue._paused_since = None


def is_marked_paused(queue) -> bool:
    with queue._lock:
        return queue._paused_reason is not None


def display_current_locked(queue) -> Path | None:
    if queue._current is not None:
        return queue._current
    if queue._active_batch:
        return queue._active_batch[0].path
    if queue._prepared:
        return queue._prepared[0].path
    if queue._prepare_futures:
        return next(iter(queue._prepare_futures.values()))
    if queue._queue:
        return queue._queue[0]
    return None


def pending_paths_locked(queue) -> list[Path]:
    active_paths = {prepared.path for prepared in queue._active_batch}
    pending: list[Path] = []
    seen: set[Path] = set()
    for path in (
        list(queue._prepare_futures.values())
        + [prepared.path for prepared in queue._prepared]
        + list(queue._queue)
    ):
        if path in active_paths or path in seen:
            continue
        seen.add(path)
        pending.append(path)
    return pending


def make_progress_locked(
    queue,
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


def refresh_progress_locked(queue):
    if queue._active_batch:
        return
    current = display_current_locked(queue)
    if current is None:
        queue._progress = None
        return
    stage = "preparing" if (queue._prepare_futures or queue._prepared) else "queued"
    queue._progress = make_progress_locked(
        queue,
        stage=stage,
        current_path=current,
        processed_chunks=0,
        total_chunks=0,
    )
