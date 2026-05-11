from __future__ import annotations

import socket
import subprocess

from src.maintenance import startup


def test_aggregate_status_uses_worst_check():
    checks = {
        "python": {"status": "ok"},
        "sqlite": {"status": "degraded"},
        "qdrant": {"status": "ok"},
    }

    assert startup.aggregate_status(checks) == "degraded"


def test_sqlite_missing_database_is_degraded(tmp_path):
    cfg = {"paths": {"db_path": str(tmp_path / "docflow.db")}}

    result = startup.check_sqlite(cfg)

    assert result["status"] == "degraded"
    assert "does not exist yet" in result["quick_check"]


def test_ensure_config_file_copies_example_and_creates_local_dirs(tmp_path):
    example_path = tmp_path / "config.example.yaml"
    config_path = tmp_path / "config.yaml"
    example_path.write_text(
        """
paths:
  watch_dirs:
    - path: "data/watch"
      recursive: true
  db_path: "data/docflow.db"
  id_counter: "data/qdrant_id_counter.txt"
qdrant:
  host: "localhost"
  port: 6333
""",
        encoding="utf-8",
    )

    generated = startup.ensure_config_file(config_path, example_path=example_path)

    assert generated == config_path
    assert config_path.exists()
    assert (tmp_path / "data").is_dir()
    assert (tmp_path / "data" / "watch").is_dir()


def test_app_port_reports_in_use():
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.bind(("127.0.0.1", 0))
    sock.listen(1)
    try:
        port = sock.getsockname()[1]
        result = startup.check_app_port(port)
    finally:
        sock.close()

    assert result["status"] == "unavailable"
    assert str(port + 1) in result["actions"][0]
    assert "docflow start" in result["actions"][0]


def test_build_startup_report_marks_qdrant_as_blocker(monkeypatch, tmp_path):
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        """
paths:
  db_path: "{db_path}"
qdrant:
  host: "localhost"
  port: 6333
  collection: "docflow"
ollama:
  base_url: "http://localhost:11434"
""".format(db_path=tmp_path / "docflow.db"),
        encoding="utf-8",
    )
    monkeypatch.setattr(
        startup, "check_python_dependencies", lambda: {"status": "ok", "actions": []}
    )
    monkeypatch.setattr(
        startup, "check_qdrant", lambda cfg: {"status": "unavailable", "actions": ["start qdrant"]}
    )
    monkeypatch.setattr(
        startup, "check_ollama", lambda cfg: {"status": "degraded", "actions": [], "optional": True}
    )
    monkeypatch.setattr(
        startup, "check_app_port", lambda port, host="127.0.0.1": {"status": "ok", "actions": []}
    )

    report = startup.build_startup_report(config_path)

    assert report["can_start"] is False
    assert report["startup_blockers"] == ["qdrant"]
    assert "start qdrant" in report["actions"]


def test_offline_report_records_zero_unexpected_hosts(monkeypatch):
    monkeypatch.setattr(
        startup,
        "build_startup_report",
        lambda config_path="config.yaml", app_port=8000: {"status": "ok", "checks": {}},
    )
    monkeypatch.setattr(
        startup,
        "_run_offline_runtime_checks",
        lambda cfg: [{"name": "startup", "status": "ok"}],
    )

    report = startup.build_offline_report("config.yaml")

    assert report["status"] == "ok"
    assert report["unexpected_outbound_connections"] == 0
    assert report["runtime_checks"][0]["name"] == "startup"
    assert report["network_registry"]


def test_offline_report_uses_privacy_allowed_hosts(monkeypatch, tmp_path):
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        """
privacy:
  allowed_hosts: ["example.com"]
""",
        encoding="utf-8",
    )
    monkeypatch.setattr(
        startup,
        "build_startup_report",
        lambda config_path="config.yaml", app_port=8000: {"status": "ok", "checks": {}},
    )
    monkeypatch.setattr(
        startup,
        "_run_offline_runtime_checks",
        lambda cfg: [{"name": "startup", "status": "ok"}],
    )

    report = startup.build_offline_report(config_path)

    assert report["allowed_hosts"] == ["example.com"]


def test_ensure_qdrant_suggests_run_command_when_container_is_missing(monkeypatch):
    cfg = {"qdrant": {"host": "localhost", "port": 6333, "collection": "docflow"}}
    monkeypatch.setattr(
        startup, "check_qdrant", lambda cfg: {"status": "unavailable", "actions": ["run qdrant"]}
    )
    monkeypatch.setattr(startup.shutil, "which", lambda name: "/usr/local/bin/docker")

    def runner(args: list[str], timeout: float):
        assert args == ["docker", "inspect", "qdrant"]
        return subprocess.CompletedProcess(args, 1, "", "not found")

    result = startup.ensure_qdrant(cfg, runner=runner)

    assert result["attempted"] is False
    assert result["actions"] == ["Run: docker compose up -d qdrant"]


def test_format_report_lists_blockers():
    report = {
        "status": "unavailable",
        "url": "http://localhost:8000",
        "startup_blockers": ["qdrant"],
        "actions": ["start qdrant"],
        "checks": {
            "python": {"status": "ok"},
            "qdrant": {"status": "unavailable", "error": "connection refused"},
        },
    }

    text = startup.format_report(report)

    assert "Startup blockers: qdrant" in text
    assert "connection refused" in text


def test_format_offline_report_lists_zero_unexpected_connections():
    text = startup.format_offline_report(
        {
            "unexpected_outbound_connections": 0,
            "unexpected_hosts": [],
            "error": "",
            "runtime_checks": [
                {"name": "startup", "status": "ok"},
                {"name": "ingest", "status": "ok"},
            ],
            "network_registry": [{"id": "local_services"}],
        }
    )

    assert "0 unexpected outbound connections" in text
    assert "Covered local paths: startup, ingest" in text
    assert "Registered network cases: local_services" in text
