from __future__ import annotations

from pathlib import Path

from src.api import app as api_app
from src.api.runtime import get_api_runtime
from src.api.state import AppContext, AppState, LLMSwitchState


def test_phase34_api_routes_are_registered_from_route_modules():
    app_source = Path("src/api/app_impl.py").read_text(encoding="utf-8")
    route_source = Path("src/api/app_routes.py").read_text(encoding="utf-8")

    assert "query_routes.create_router" in route_source
    assert "library_routes.create_router" in route_source
    assert "imports_routes.create_router" in route_source
    assert "settings_routes.create_router" in route_source
    assert "maintenance_routes.create_router" in route_source
    assert '@app.get("/api/' not in app_source
    assert '@app.post("/api/' not in app_source
    assert '@app.patch("/api/' not in app_source
    assert '@app.delete("/api/' not in app_source

    for path in [
        "src/api/routes/query.py",
        "src/api/routes/library.py",
        "src/api/routes/imports.py",
        "src/api/routes/settings.py",
        "src/api/routes/maintenance.py",
        "src/api/services/query_service.py",
        "src/api/services/import_service.py",
        "src/api/services/health_service.py",
        "src/api/app_routes.py",
        "src/api/app_static.py",
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
        "src/api/app_impl.py": 350,
        "src/api/app_routes.py": 180,
        "src/api/app_static.py": 80,
        "src/api/lifecycle.py": 250,
        "src/api/runtime_helpers.py": 250,
        "src/api/handlers/import_handlers.py": 250,
        "src/api/handlers/library_handlers.py": 350,
        "src/api/handlers/maintenance_handlers.py": 80,
        "src/api/handlers/query_handlers.py": 350,
        "src/api/handlers/query_stream_handlers.py": 240,
        "src/api/handlers/settings_handlers.py": 150,
        "src/api/health_checks.py": 250,
        "src/api/health_status.py": 350,
        "src/ingest/store_impl.py": 80,
        "src/ingest/store_db.py": 260,
        "src/ingest/store_files.py": 450,
        "src/ingest/store_vectors.py": 350,
        "src/ingest/store_history.py": 220,
        "src/ingest/store_library.py": 180,
        "src/ingest/pipeline.py": 400,
        "src/ingest/pipeline_batch.py": 260,
        "src/ingest/pipeline_context.py": 180,
        "src/ingest/pipeline_types.py": 80,
        "src/ingest/pipeline_vectors.py": 150,
        "src/ingest/queue.py": 350,
        "src/ingest/queue_prepared.py": 240,
        "src/ingest/queue_status.py": 170,
        "src/query/retriever_impl.py": 350,
        "src/query/vector_search.py": 120,
        "src/query/keyword_search.py": 140,
        "src/query/fusion.py": 140,
        "src/query/debug.py": 180,
        "src/query/router.py": 140,
        "src/query/reranker.py": 140,
        "src/query/engine.py": 350,
        "src/query/engine_helpers.py": 180,
        "src/query/settings.py": 90,
        "src/query/generator.py": 350,
        "src/query/generator_backends.py": 220,
        "src/query/citations.py": 150,
        "src/query/generator_context.py": 80,
        "src/query/generator_prompts.py": 60,
        "src/config.py": 350,
        "src/config_defaults.py": 80,
        "src/config_ingest.py": 80,
        "src/config_query.py": 90,
        "src/maintenance/startup.py": 350,
        "src/maintenance/startup_checks.py": 350,
        "src/maintenance/offline_doctor.py": 350,
        "src/quality/browser_acceptance.py": 180,
        "src/quality/browser_acceptance_plan.py": 160,
        "src/quality/browser_acceptance_checks.py": 350,
        "src/quality/browser_acceptance_a11y.py": 220,
        "src/quality/browser_acceptance_mutation.py": 350,
        "src/quality/browser_acceptance_report.py": 130,
    }

    for path, limit in limits.items():
        assert Path(path).exists()
        line_count = len(Path(path).read_text(encoding="utf-8").splitlines())
        assert line_count < limit, f"{path} has {line_count} lines"


def test_phase34_app_state_holds_runtime_dependencies():
    assert api_app.app_context is api_app.app_state
    assert isinstance(api_app.app_context, AppContext)
    assert isinstance(api_app.app_state, AppState)
    assert api_app.app_state.model_tasks is api_app.model_tasks
    assert api_app.app_state.llm_switch_state is api_app.llm_switch_state


def test_phase68_runtime_state_has_single_context_source():
    for name in [
        "pipeline",
        "ingest_queue",
        "query_engine",
        "store",
        "watcher",
        "watch_dirs",
        "llm_options",
        "model_tasks",
        "llm_switch_state",
    ]:
        assert name not in vars(api_app)
        assert getattr(api_app, name) is getattr(api_app.app_context, name)


def test_phase98_api_handlers_use_runtime_context_not_app_impl_module_lookup():
    checked_paths = [
        *sorted(Path("src/api/handlers").glob("*_handlers.py")),
        Path("src/api/lifecycle.py"),
        Path("src/api/runtime_helpers.py"),
    ]

    for path in checked_paths:
        source = path.read_text(encoding="utf-8")
        assert 'sys.modules["src.api.app_impl"]' not in source
        assert "import sys" not in source
        assert "get_api_runtime" in source

    pyproject = Path("pyproject.toml").read_text(encoding="utf-8")
    assert '"src/api/runtime.py"' in pyproject
    assert '"src/api/services"' in pyproject

    runtime = get_api_runtime()
    assert runtime.FOREGROUND_PAUSE_GRACE_S == api_app.FOREGROUND_PAUSE_GRACE_S
    assert runtime.INGEST_PAUSE_CHECK_INTERVAL_MS == api_app.INGEST_PAUSE_CHECK_INTERVAL_MS


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
