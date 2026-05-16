from __future__ import annotations

from pathlib import Path

import yaml

from src.api.runtime_helpers import _parse_watch_dirs
from src.config import ConfigError, DocFlowSettings
from src.maintenance import startup
from src.query.engine import QueryEngine, QuerySettings
from src.query.generator import Answer


def _load_yaml(path: str) -> dict:
    with Path(path).open(encoding="utf-8") as fh:
        return yaml.safe_load(fh) or {}


def test_example_and_docker_configs_share_runtime_quality_keys():
    example = _load_yaml("config.example.yaml")
    docker = _load_yaml("config.docker.yaml")

    assert set(example) == set(docker)
    assert set(example["query"]) == set(docker["query"])
    assert set(example["ingest"]) == set(docker["ingest"])
    assert example["query"]["fallback_mode"] == "visible_snippet_fallback"
    assert docker["query"]["fallback_mode"] == "visible_snippet_fallback"


def test_typed_settings_resolve_local_paths_from_config_location(tmp_path):
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "paths": {
                    "watch_dirs": [{"path": "data/watch", "recursive": True}],
                    "db_path": "data/docflow.db",
                    "id_counter": "data/qdrant_id_counter.txt",
                    "supported_extensions": [".md"],
                },
                "qdrant": {"host": "localhost", "port": 6333, "collection": "custom"},
                "embedding": {"model": "Qwen/Qwen3-Embedding-0.6B", "batch_size": 16},
                "chunking": {"chunk_size": 256, "chunk_overlap": 32},
                "ingest": {
                    "parse_workers": 4,
                    "microbatch_max_files": 3,
                    "microbatch_max_chunks": 99,
                    "microbatch_linger_ms": 12,
                    "pause_check_interval_ms": 34,
                },
                "query": {"min_rerank_score": 0.2, "fallback_mode": "visible_snippet_fallback"},
            }
        ),
        encoding="utf-8",
    )

    settings = DocFlowSettings.from_file(config_path)

    assert settings.paths.db_path == tmp_path / "data" / "docflow.db"
    assert settings.paths.id_counter == tmp_path / "data" / "qdrant_id_counter.txt"
    assert settings.paths.watch_dirs[0].path == tmp_path / "data" / "watch"
    assert settings.qdrant.collection == "custom"
    assert settings.embedding.batch_size == 16
    assert settings.query.min_rerank_score == 0.2
    assert settings.ingest.parse_workers == 4
    assert settings.ingest.microbatch_max_files == 3
    assert settings.ingest.microbatch_max_chunks == 99
    assert settings.ingest.microbatch_linger_ms == 12
    assert settings.ingest.pause_check_interval_ms == 34

    runtime_watch_dirs = _parse_watch_dirs(_load_yaml(str(config_path)), config_path=config_path)
    assert runtime_watch_dirs[0].path == tmp_path / "data" / "watch"


def test_missing_watch_dirs_are_reported_not_silently_defaulted(tmp_path):
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "paths": {
                    "db_path": "data/docflow.db",
                    "id_counter": "data/qdrant_id_counter.txt",
                    "supported_extensions": [".md"],
                },
                "qdrant": {"host": "localhost", "port": 6333},
                "embedding": {"model": "Qwen/Qwen3-Embedding-0.6B"},
                "chunking": {"chunk_size": 512, "chunk_overlap": 51},
            }
        ),
        encoding="utf-8",
    )

    settings = DocFlowSettings.from_file(config_path)
    assert settings.paths.watch_dirs == ()

    try:
        _parse_watch_dirs(_load_yaml(str(config_path)), config_path=config_path)
    except ConfigError as exc:
        assert "paths.watch_dirs" in str(exc)
    else:
        raise AssertionError("Expected runtime watch-dir parsing to raise ConfigError")


def test_invalid_config_is_reported_as_startup_failure(tmp_path):
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "paths": {"db_path": "data/docflow.db"},
                "qdrant": {"port": 6333},
                "embedding": {"model": "Qwen/Qwen3-Embedding-0.6B"},
                "chunking": {"chunk_size": 512, "chunk_overlap": 51},
            }
        ),
        encoding="utf-8",
    )

    with config_path.open(encoding="utf-8") as fh:
        assert yaml.safe_load(fh)

    try:
        DocFlowSettings.from_file(config_path)
    except ConfigError as exc:
        assert "qdrant.host" in str(exc)
    else:
        raise AssertionError("Expected invalid config to raise ConfigError")

    report = startup.check_config(config_path)
    assert report["status"] == "unavailable"
    assert "qdrant.host" in report["error"]


def test_configured_insufficient_evidence_message_is_visible_to_users():
    class LowEvidenceRetriever:
        def retrieve(self, *args, **kwargs):
            return [
                {
                    "text": "weak match",
                    "file_name": "weak.md",
                    "rerank_score": 0.01,
                }
            ]

    class Generator:
        calls = 0

        def generate(self, *args, **kwargs):
            self.calls += 1
            return Answer(text="should not be generated", citations=[])

    generator = Generator()
    engine = QueryEngine(
        LowEvidenceRetriever(),
        generator,
        settings=QuerySettings(
            min_rerank_score=0.5,
            insufficient_evidence_message="资料不足：请先导入更多来源。",
        ),
    )

    answer = engine.query("weak question")

    assert answer.text == "资料不足：请先导入更多来源。"
    assert answer.quality["status"] == "insufficient_evidence"
    assert generator.calls == 0
