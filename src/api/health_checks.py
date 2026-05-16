"""Health check helpers for the DocFlow API."""

from __future__ import annotations

import sqlite3
from collections.abc import Callable
from pathlib import Path
from time import perf_counter
from typing import Any

from src import net
from src.api.health_status import (
    _aggregate_health_status as _aggregate_health_status,
)
from src.api.health_status import (
    _health_actions as _health_actions,
)
from src.api.health_status import (
    _health_capabilities as _health_capabilities,
)
from src.api.health_status import (
    _health_groups as _health_groups,
)
from src.model_cache import (
    configured_hf_model_status,
)

COLLECTION_NAME = "docflow"


def _default_store_getter() -> Any | None:
    return None


_store_getter: Callable[[], Any | None] = _default_store_getter


def configure_health_checks(*, store_getter: Callable[[], Any | None]) -> None:
    global _store_getter
    _store_getter = store_getter


def _timed_check(fn) -> dict:
    start = perf_counter()
    try:
        result = fn()
        if not isinstance(result, dict):
            result = {"status": "ok"}
    except Exception as exc:
        result = {"status": "unavailable", "error": str(exc)}
    result["latency_ms"] = round((perf_counter() - start) * 1000, 2)
    return result
def _check_sqlite(cfg: dict) -> dict:
    active_store = _store_getter()
    if active_store is not None:
        with active_store._conn() as conn:
            conn.execute("SELECT 1").fetchone()
            conn.execute("CREATE TEMP TABLE IF NOT EXISTS health_check(value INTEGER)")
            conn.execute("DELETE FROM health_check")
            conn.execute("INSERT INTO health_check(value) VALUES (1)")
            fts_tables = [
                row["name"]
                for row in conn.execute(
                    """
                    SELECT name FROM sqlite_master
                    WHERE type = 'table'
                      AND name IN ('chunks_fts', 'chunks_fts_trigram', 'history_fts')
                    ORDER BY name
                    """
                ).fetchall()
            ]
        required_fts_tables = {"chunks_fts", "chunks_fts_trigram", "history_fts"}
        missing_fts_tables = sorted(required_fts_tables - set(fts_tables))
        return {
            "status": "ok" if not missing_fts_tables else "unavailable",
            "mode": "runtime",
            "write_check": "ok",
            "fts_tables": fts_tables,
            "missing_fts_tables": missing_fts_tables,
            "quick_check": "skipped during app runtime",
            "note": "Use `docflow doctor --strict` for a full SQLite integrity check.",
        }
    else:
        db_path = Path(cfg["paths"]["db_path"]).expanduser()
        conn = sqlite3.connect(db_path)
        try:
            conn.execute("SELECT 1").fetchone()
            conn.execute("CREATE TEMP TABLE IF NOT EXISTS health_check(value INTEGER)")
            conn.execute("INSERT INTO health_check(value) VALUES (1)")
            quick_check = conn.execute("PRAGMA quick_check").fetchone()[0]
        finally:
            conn.close()

    status = "ok" if quick_check == "ok" else "unavailable"
    return {"status": status, "mode": "offline", "quick_check": quick_check}
def _check_qdrant(cfg: dict) -> dict:
    from qdrant_client import QdrantClient

    client = QdrantClient(host=cfg["qdrant"]["host"], port=cfg["qdrant"]["port"], timeout=2)
    try:
        collection = cfg.get("qdrant", {}).get("collection", COLLECTION_NAME)
        info = client.get_collection(collection)
        return {
            "status": "ok",
            "collection": collection,
            "points_count": getattr(info, "points_count", 0),
            "vectors_count": getattr(info, "vectors_count", None),
        }
    finally:
        close = getattr(client, "close", None)
        if callable(close):
            close()
def _check_ollama(cfg: dict) -> dict:
    ollama_cfg = cfg.get("ollama", {})
    ingest_cfg = cfg.get("ingest", {})
    llm_cfg = cfg.get("llm", {})
    base_url = ollama_cfg.get("base_url", "http://localhost:11434").rstrip("/")
    required = {
        "ocr": ollama_cfg.get("ocr_model", ""),
        "contextual_prefix": (
            ingest_cfg.get("contextual_prefix_model", "")
            if ingest_cfg.get("contextual_prefix_mode") == "ollama"
            else ""
        ),
        "llm": llm_cfg.get("ollama_model", ollama_cfg.get("llm_model", ""))
        if llm_cfg.get("backend", "local") == "local"
        else "",
    }
    try:
        response = net.get(
            f"{base_url}/api/tags",
            timeout=net.Timeout(2.0, connect=1.0),
        )
        response.raise_for_status()
        data = response.json()
    except net.ConnectTimeout as exc:
        error = f"connection timeout: {exc}"
    except net.ReadTimeout as exc:
        error = f"read timeout: {exc}"
    except Exception as exc:
        error = str(exc)
    else:
        installed = set()
        for item in data.get("models", []):
            name = item.get("name", "")
            if not name:
                continue
            installed.add(name)
            installed.add(name.split(":", 1)[0])

        models = {}
        missing = []
        for purpose, model in required.items():
            if not model:
                continue
            available = model in installed or model.split(":", 1)[0] in installed
            models[purpose] = {"model": model, "available": available}
            if not available:
                missing.append(model)

        status = "ok" if not missing else "degraded"
        return {
            "status": status,
            "base_url": base_url,
            "models": models,
            "missing_models": missing,
        }

    models = {
        purpose: {"model": model, "available": False}
        for purpose, model in required.items()
        if model
    }
    return {
        "status": "degraded",
        "base_url": base_url,
        "models": models,
        "missing_models": [model for model in required.values() if model],
        "error": error,
        "actions": ["打开 Ollama；只有 OCR 或 Ollama 后端功能需要它。"],
    }
def _check_models(cfg: dict) -> dict:
    local_models = configured_hf_model_status(cfg)
    missing = [item["model"] for item in local_models.values() if not item["cached"]]
    return {
        "status": "ok" if not missing else "degraded",
        "local_cache": local_models,
        "missing_local_cache": missing,
        "note": "Local model check only inspects cache folders and does not download models.",
    }
