import threading
import time
from pathlib import Path

from src.ingest.chunker import Chunk
from src.ingest.pdf_analyzer import ParsedDocument
from src.ingest.pipeline import IngestMetrics, PreparedIngestFile
from src.ingest.queue import IngestQueue


def wait_until(predicate, timeout: float = 2.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.02)
    return False


class BlockingPipeline:
    def __init__(self):
        self.started = threading.Event()
        self.release = threading.Event()

    def ingest(self, path):
        self.started.set()
        self.release.wait(timeout=2)
        return {"status": "done", "file": Path(path).name}


def test_same_basename_different_paths_can_queue_while_processing():
    pipeline = BlockingPipeline()
    queue = IngestQueue(pipeline)

    first = Path("/tmp/source-a/report.pdf")
    second = Path("/tmp/source-b/report.pdf")

    queue.start()
    try:
        queue.submit(first)
        assert pipeline.started.wait(timeout=1)

        result = queue.submit(second)

        assert result["status"] == "queued"
        assert queue.queue_size == 1
    finally:
        pipeline.release.set()
        queue.stop()


class FlakyLegacyPipeline:
    def __init__(self):
        self.calls = 0

    def ingest(self, path):
        self.calls += 1
        if self.calls == 1:
            raise RuntimeError("temporary failure")
        return {"status": "done", "file": Path(path).name}


def test_legacy_worker_records_failure_and_allows_retry():
    pipeline = FlakyLegacyPipeline()
    queue = IngestQueue(pipeline)
    path = Path("/tmp/source-a/retry-note.md")

    queue.start()
    try:
        assert queue.submit(path)["status"] == "queued"
        assert wait_until(lambda: (queue.status()["last_completed"] or {}).get("status") == "error")
        failed = queue.status()["last_completed"]
        assert failed["file"] == path.name
        assert "temporary failure" in failed["error"]

        assert queue.submit(path)["status"] == "queued"
        assert wait_until(lambda: (queue.status()["last_completed"] or {}).get("status") == "done")
        assert queue.status()["last_completed"] == {"status": "done", "file": path.name}
    finally:
        queue.stop()


class BatchingPipeline:
    def __init__(self):
        self.started = threading.Event()
        self.release = threading.Event()
        self.batch_sizes: list[int] = []

    def prepare_file(self, path):
        path = Path(path)
        return PreparedIngestFile(
            path=path,
            file_id=1,
            file_hash="hash",
            mtime_ns=0,
            doc=ParsedDocument(
                file_path=path,
                file_name=path.name,
                total_pages=1,
                is_scanned=False,
                pages=[],
            ),
            tags_json="[]",
            chunks=[
                Chunk(
                    text=f"{path.name}-1",
                    chunk_type="text",
                    file_name=path.name,
                    file_path=str(path),
                    page_num=1,
                ),
                Chunk(
                    text=f"{path.name}-2",
                    chunk_type="text",
                    file_name=path.name,
                    file_path=str(path),
                    page_num=1,
                ),
            ],
            is_cjk=False,
            old_qdrant_ids=[],
            metrics=IngestMetrics(chunk_count=2),
        )

    def ingest_prepared_batch(self, prepared_files, progress_callback=None):
        self.batch_sizes.append(len(prepared_files))
        if progress_callback:
            progress_callback(
                {
                    "stage": "embedding",
                    "processed_chunks": 2,
                    "total_chunks": 4,
                    "adaptive_batch_size": 4,
                }
            )
        self.started.set()
        self.release.wait(timeout=2)
        if progress_callback:
            progress_callback(
                {
                    "stage": "storing",
                    "processed_chunks": 4,
                    "total_chunks": 4,
                    "current_file": prepared_files[0].path.name,
                    "current_path": str(prepared_files[0].path),
                }
            )
        return [
            {
                "status": "done",
                "file": prepared.path.name,
                "chunks": len(prepared.chunks),
                "metrics": {"total_s": 0.1},
            }
            for prepared in prepared_files
        ]


def test_queue_batches_files_and_exposes_chunk_progress():
    pipeline = BatchingPipeline()
    queue = IngestQueue(
        pipeline,
        parse_workers=2,
        microbatch_max_files=8,
        microbatch_max_chunks=128,
        microbatch_linger_ms=100,
    )

    first = Path("/tmp/source-a/report-a.pdf")
    second = Path("/tmp/source-b/report-b.pdf")

    queue.start()
    try:
        queue.submit(first)
        queue.submit(second)

        assert pipeline.started.wait(timeout=1)
        status = queue.status()

        assert status["processing"] in {first.name, second.name}
        assert status["progress"]["stage"] == "embedding"
        assert status["progress"]["processed_chunks"] == 2
        assert status["progress"]["total_chunks"] == 4
        assert status["progress"]["batch_size"] == 2
        assert len(status["progress"]["batch_files"]) == 2

        pipeline.release.set()
    finally:
        queue.stop()

    assert pipeline.batch_sizes == [2]


def test_queue_respects_microbatch_file_boundary():
    pipeline = BatchingPipeline()
    queue = IngestQueue(
        pipeline,
        parse_workers=3,
        microbatch_max_files=2,
        microbatch_max_chunks=128,
        microbatch_linger_ms=100,
    )

    files = [
        Path("/tmp/source-a/report-a.pdf"),
        Path("/tmp/source-b/report-b.pdf"),
        Path("/tmp/source-c/report-c.pdf"),
    ]

    queue.start()
    try:
        for path in files:
            queue.submit(path)
        assert pipeline.started.wait(timeout=1)
        pipeline.release.set()
        assert wait_until(lambda: pipeline.batch_sizes == [2, 1])
    finally:
        queue.stop()


def test_queue_pauses_prepared_batch_while_foreground_active():
    pipeline = BatchingPipeline()
    foreground_active = True
    queue = IngestQueue(
        pipeline,
        parse_workers=1,
        microbatch_max_files=8,
        microbatch_max_chunks=128,
        microbatch_linger_ms=0,
        should_pause_background=lambda: foreground_active,
        pause_check_interval_ms=50,
    )

    path = Path("/tmp/source-a/report-a.pdf")

    queue.start()
    try:
        queue.submit(path)

        deadline = threading.Event()
        paused = False
        for _ in range(20):
            status = queue.status()
            if status["paused"]:
                paused = True
                break
            deadline.wait(timeout=0.05)

        assert paused is True
        assert pipeline.started.is_set() is False
        status = queue.status()
        assert status["pause_reason"] == "foreground_active"
        assert status["progress"]["stage"] == "paused"

        foreground_active = False
        assert pipeline.started.wait(timeout=1)
        pipeline.release.set()
        assert wait_until(lambda: (queue.status()["last_completed"] or {}).get("status") == "done")
    finally:
        queue.stop()
