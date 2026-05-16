from __future__ import annotations

from pathlib import Path

import yaml


def test_scan_passes_config_path_to_watch_dir_parser(monkeypatch, tmp_path):
    import main
    import src.api.app as api_app
    from src.ingest.pipeline import IngestPipeline

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
                "qdrant": {"host": "localhost", "port": 6333, "collection": "docflow"},
                "embedding": {"model": "Qwen/Qwen3-Embedding-0.6B"},
                "chunking": {"chunk_size": 512, "chunk_overlap": 51},
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setenv("DOCFLOW_CONFIG", str(config_path))

    calls: list[Path | None] = []

    def fake_parse_watch_dirs(cfg: dict, config_path: str | Path | None = None):
        calls.append(Path(config_path) if config_path else None)
        return []

    class Pipeline:
        class Registry:
            supported_extensions: list[str] = []

        registry = Registry()

    monkeypatch.setattr(api_app, "_parse_watch_dirs", fake_parse_watch_dirs)
    monkeypatch.setattr(IngestPipeline, "from_config", lambda path: Pipeline())

    main.scan()

    assert calls == [config_path]
