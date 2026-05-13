from __future__ import annotations

from pathlib import Path

from src.api import app as api_app
from src.api.state import AppState, LLMSwitchState


def test_phase34_api_routes_are_registered_from_route_modules():
    source = Path("src/api/app_impl.py").read_text(encoding="utf-8")

    assert "query_routes.create_router" in source
    assert "library_routes.create_router" in source
    assert "imports_routes.create_router" in source
    assert "settings_routes.create_router" in source
    assert "maintenance_routes.create_router" in source
    assert '@app.get("/api/' not in source
    assert '@app.post("/api/' not in source
    assert '@app.patch("/api/' not in source
    assert '@app.delete("/api/' not in source

    for path in [
        "src/api/routes/query.py",
        "src/api/routes/library.py",
        "src/api/routes/imports.py",
        "src/api/routes/settings.py",
        "src/api/routes/maintenance.py",
        "src/api/services/query_service.py",
        "src/api/services/import_service.py",
        "src/api/services/health_service.py",
    ]:
        assert Path(path).exists()


def test_phase52_public_entrypoints_are_small_facades():
    limits = {
        "src/api/app.py": 500,
        "src/ingest/store.py": 500,
        "src/query/retriever.py": 500,
    }

    for path, limit in limits.items():
        line_count = len(Path(path).read_text(encoding="utf-8").splitlines())
        assert line_count < limit


def test_phase60_god_modules_are_split_into_focused_files():
    limits = {
        "src/api/app_impl.py": 1700,
        "src/api/health_checks.py": 550,
        "src/ingest/store_impl.py": 80,
        "src/ingest/store_db.py": 260,
        "src/ingest/store_files.py": 450,
        "src/ingest/store_vectors.py": 350,
        "src/ingest/store_history.py": 220,
        "src/ingest/store_library.py": 180,
        "src/query/retriever_impl.py": 750,
        "src/query/router.py": 140,
        "src/query/reranker.py": 140,
    }

    for path, limit in limits.items():
        assert Path(path).exists()
        line_count = len(Path(path).read_text(encoding="utf-8").splitlines())
        assert line_count < limit, f"{path} has {line_count} lines"


def test_phase34_app_state_holds_runtime_dependencies():
    assert isinstance(api_app.app_state, AppState)
    assert api_app.app_state.model_tasks is api_app.model_tasks
    assert api_app.app_state.llm_switch_state is api_app.llm_switch_state


def test_phase34_llm_switch_state_is_thread_safe_mapping():
    state = LLMSwitchState()

    state.set("switching", model="model-a")
    switching = dict(state)
    state.set("idle", model="model-a", message="done")
    idle = dict(state)

    assert switching["state"] == "switching"
    assert switching["finished_at"] is None
    assert idle["state"] == "idle"
    assert idle["model"] == "model-a"
    assert idle["message"] == "done"
    assert idle["started_at"] == switching["started_at"]
    assert idle["finished_at"] is not None
