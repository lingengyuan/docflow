from __future__ import annotations

from pathlib import Path

from src.api import app as api_app
from src.api.state import AppState, LLMSwitchState


def test_phase34_api_routes_are_registered_from_route_modules():
    source = Path("src/api/app.py").read_text(encoding="utf-8")

    assert "app.include_router(query_routes.create_router" in source
    assert "app.include_router(library_routes.create_router" in source
    assert "app.include_router(imports_routes.create_router" in source
    assert "app.include_router(settings_routes.create_router" in source
    assert "app.include_router(maintenance_routes.create_router" in source
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
