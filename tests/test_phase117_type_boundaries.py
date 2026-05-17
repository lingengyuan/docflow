from __future__ import annotations

import tomllib
from pathlib import Path

from src.api.state import LLMSwitchState

ROOT = Path(__file__).resolve().parents[1]


def read(path: str) -> str:
    return (ROOT / path).read_text(encoding="utf-8")


def test_phase117_mypy_covers_api_ingest_and_maintenance_boundaries():
    with (ROOT / "pyproject.toml").open("rb") as handle:
        pyproject = tomllib.load(handle)

    files = set(pyproject["tool"]["mypy"]["files"])
    expected = {
        "src/api/state.py",
        "src/api/runtime_helpers.py",
        "src/ingest/embedder.py",
        "src/ingest/parsers",
        "src/ingest/pipeline.py",
        "src/ingest/pipeline_batch.py",
        "src/ingest/pipeline_types.py",
        "src/maintenance/offline_doctor.py",
        "src/maintenance/platform.py",
    }

    assert expected <= files


def test_phase117_ingest_registry_and_embedder_boundaries_are_typed():
    parsers = read("src/ingest/parsers/__init__.py")
    pipeline_types = read("src/ingest/pipeline_types.py")

    assert "dict[str, FileParser]" in parsers
    assert "def resolve(self, file_path: Path) -> FileParser" in parsers
    assert "class IngestEmbedder(Protocol)" in pipeline_types
    assert "class ProgressPayload(TypedDict" in pipeline_types


def test_phase117_llm_switch_snapshot_is_typed_and_mapping_compatible():
    state = LLMSwitchState()

    state.set("switching", model="qwen-local")
    snapshot = state.snapshot()

    assert snapshot["state"] == "switching"
    assert snapshot["model"] == "qwen-local"
    assert dict(state)["state"] == "switching"
