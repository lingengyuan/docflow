from __future__ import annotations

import asyncio
import time

import pytest

from src.api.model_tasks import ModelTaskController, ModelTaskTimeout


def test_model_task_timeout_retires_worker_and_allows_next_task():
    controller = ModelTaskController(thread_name_prefix="test-model-task")
    try:
        with pytest.raises(ModelTaskTimeout):
            asyncio.run(controller.run("slow", lambda: time.sleep(0.2), timeout_s=0.01))

        result = asyncio.run(controller.run("fast", lambda: "ok", timeout_s=0.1))

        assert result == "ok"
    finally:
        controller.shutdown()


def test_cancel_and_retire_allows_later_task_to_run():
    controller = ModelTaskController(thread_name_prefix="test-model-task")
    try:
        task = controller.submit("slow-stream", lambda: time.sleep(0.2))

        controller.cancel_and_retire(task, reason="test cancellation")
        result = asyncio.run(controller.run("fast-stream", lambda: "ok", timeout_s=0.1))

        assert result == "ok"
    finally:
        controller.shutdown()
