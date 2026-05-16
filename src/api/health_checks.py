"""Health check helpers for the DocFlow API."""

from __future__ import annotations

import sqlite3
from collections.abc import Callable
from pathlib import Path
from time import perf_counter
from typing import Any

from src import net
from src.domain_types import HealthAction
from src.model_cache import (
    configured_hf_model_status,
)
from src.model_cache import (
    configured_model_names as _configured_model_names,
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
def _health_capabilities(cfg: dict, checks: dict) -> dict:
    ollama_models = checks["ollama"].get("models", {})
    local_models = checks["models"].get("local_cache", {})
    ingest_cfg = cfg.get("ingest", {})
    vlm_cfg = cfg.get("vlm", {})
    contextual_prefix_enabled = ingest_cfg.get("contextual_prefix", False)
    contextual_prefix_available = contextual_prefix_enabled and (
        ingest_cfg.get("contextual_prefix_mode") != "ollama"
        or ollama_models.get("contextual_prefix", {}).get("available", False)
    )
    vlm_enabled = vlm_cfg.get("enabled", True)
    return {
        "query": checks["sqlite"]["status"] == "ok" and checks["qdrant"]["status"] == "ok",
        "ingest": checks["sqlite"]["status"] == "ok" and checks["qdrant"]["status"] == "ok",
        "ocr": ollama_models.get("ocr", {}).get("available", False),
        "enhanced_llm": local_models.get("llm_enhanced", {}).get("cached", False),
        "vlm": vlm_enabled and local_models.get("vlm", {}).get("cached", False),
        "vlm_enabled": vlm_enabled,
        "contextual_prefix": contextual_prefix_available,
        "contextual_prefix_enabled": contextual_prefix_enabled,
        "contextual_prefix_mode": ingest_cfg.get("contextual_prefix_mode", "metadata"),
    }
def _health_groups(cfg: dict, checks: dict, capabilities: dict) -> dict:
    model_names = _configured_model_names(cfg)
    local_models = checks["models"].get("local_cache", {})
    missing_local_cache = set(checks["models"].get("missing_local_cache", []))
    vlm_cfg = cfg.get("vlm", {})
    cfg.get("ingest", {})

    def core_item(
        key: str,
        label: str,
        ok: bool,
        detail_ok: str,
        detail_bad: str,
        actions: list[str] | None = None,
    ) -> dict:
        return {
            "key": key,
            "label": label,
            "status": "ok" if ok else "unavailable",
            "detail": detail_ok if ok else detail_bad,
            "actions": [] if ok else (actions or []),
        }

    def check_item(key: str, label: str, check: dict, fallback_detail: str) -> dict:
        detail = (
            check.get("error") or check.get("note") or check.get("collection") or fallback_detail
        )
        actions = []
        if check.get("status") != "ok":
            if key == "sqlite":
                actions = [
                    "运行 docflow doctor --strict",
                    "必要时先备份，再运行 docflow admin rebuild --dry-run",
                ]
            elif key == "qdrant":
                actions = ["确认 Docker/Qdrant 已启动", "运行 docflow admin check --json"]
        return {
            "key": key,
            "label": label,
            "status": check.get("status", "unknown"),
            "detail": str(detail),
            "actions": actions,
        }

    def optional_item(
        key: str,
        label: str,
        enabled: bool,
        available: bool,
        detail_ok: str,
        detail_bad: str,
        actions: list[str] | None = None,
    ) -> dict:
        if not enabled:
            return {
                "key": key,
                "label": label,
                "status": "off",
                "detail": "未启用，不影响问答和入库核心流程。",
                "actions": [],
            }
        return {
            "key": key,
            "label": label,
            "status": "ok" if available else "optional_unavailable",
            "detail": detail_ok if available else detail_bad,
            "actions": [] if available else (actions or []),
        }

    def model_cache_item(key: str, label: str, model: str, critical: bool = False) -> dict:
        model_status = local_models.get(key, {})
        if model_status.get("model"):
            model = str(model_status["model"])
        cached = True
        if model and "/" in model:
            cached = bool(model_status.get("cached", False))
        if not model:
            return {
                "key": key,
                "label": label,
                "status": "off",
                "detail": "未配置。",
                "actions": [],
            }
        if cached:
            return {
                "key": key,
                "label": label,
                "status": "ok",
                "detail": f"{model} 本地可用。",
                "actions": [],
            }
        return {
            "key": key,
            "label": label,
            "status": "degraded" if critical else "optional_unavailable",
            "detail": f"{model} 本地缓存缺失。",
            "actions": [f"联网后准备模型缓存：{model}"],
        }

    enhanced_model = local_models.get("llm_enhanced", {}).get("model", "")
    vlm_model = local_models.get("vlm", {}).get("model") or vlm_cfg.get("model", "")
    contextual_prefix_enabled = capabilities.get("contextual_prefix_enabled", False)
    contextual_prefix_mode = capabilities.get("contextual_prefix_mode", "metadata")
    ocr_missing = ", ".join(checks["ollama"].get("missing_models", []))
    missing_model_text = ", ".join(sorted(missing_local_cache))

    return {
        "core": {
            "label": "核心功能",
            "items": [
                core_item(
                    "query",
                    "问答",
                    capabilities.get("query", False),
                    "可以检索文档并回答问题。",
                    "SQLite 或 Qdrant 不可用，问答不可用。",
                    ["运行 docflow doctor --strict", "确认 Qdrant 正在运行"],
                ),
                core_item(
                    "ingest",
                    "入库",
                    capabilities.get("ingest", False),
                    "可以解析文件并写入索引。",
                    "SQLite 或 Qdrant 不可用，入库不可用。",
                    [
                        "运行 docflow admin check --json",
                        "必要时运行 docflow admin rebuild --dry-run",
                    ],
                ),
                check_item("sqlite", "SQLite", checks["sqlite"], "本地记录库可用。"),
                check_item("qdrant", "Qdrant", checks["qdrant"], "向量库可用。"),
            ],
        },
        "runtime": {
            "label": "模型运行时",
            "items": [
                model_cache_item(
                    "embedding", "向量模型", model_names.get("embedding", ""), critical=True
                ),
                model_cache_item(
                    "reranker", "精排模型", model_names.get("reranker", ""), critical=True
                ),
                model_cache_item("llm", "回答模型", model_names.get("llm", ""), critical=True),
                model_cache_item("llm_enhanced", "增强回答模型", enhanced_model, critical=False),
                optional_item(
                    "ocr_runtime",
                    "OCR 模型",
                    bool(checks["ollama"].get("models", {}).get("ocr")),
                    capabilities.get("ocr", False),
                    checks["ollama"].get("models", {}).get("ocr", {}).get("model", "OCR 模型")
                    + " 可用。",
                    f"缺失：{ocr_missing or 'OCR 模型或 Ollama'}。",
                    ["打开 Ollama", f"运行 ollama pull {model_names.get('ocr', 'glm-ocr')}"],
                ),
                model_cache_item("vlm", "图片理解模型", vlm_model, critical=False),
            ],
        },
        "optional": {
            "label": "可选能力",
            "items": [
                optional_item(
                    "ocr",
                    "OCR",
                    bool(checks["ollama"].get("models", {}).get("ocr")),
                    capabilities.get("ocr", False),
                    "扫描 PDF 识别可用。",
                    f"只影响扫描 PDF 识别；缺失：{ocr_missing or 'OCR 模型或 Ollama'}。",
                    ["打开 Ollama", f"运行 ollama pull {model_names.get('ocr', 'glm-ocr')}"],
                ),
                optional_item(
                    "enhanced_llm",
                    "增强模型",
                    bool(enhanced_model),
                    capabilities.get("enhanced_llm", False),
                    "增强问答模型已缓存。",
                    "只影响增强模型切换；缺失："
                    f"{enhanced_model or missing_model_text or '增强模型缓存'}。",
                    [f"联网后准备模型缓存：{enhanced_model}"] if enhanced_model else [],
                ),
                optional_item(
                    "vlm",
                    "图片理解",
                    bool(vlm_cfg.get("enabled", True)),
                    capabilities.get("vlm", False),
                    "图片入库解析可用。",
                    f"只影响图片入库；缺失：{vlm_model or missing_model_text or 'VLM 模型缓存'}。",
                    [f"联网后准备模型缓存：{vlm_model}"] if vlm_model else [],
                ),
                optional_item(
                    "contextual_prefix",
                    "上下文前缀",
                    contextual_prefix_enabled,
                    capabilities.get("contextual_prefix", False),
                    f"{contextual_prefix_mode} 模式可用。",
                    "只影响检索上下文增强，不影响基础问答。",
                    ["如需启用，先确认 config.yaml 中的 contextual_prefix 配置。"],
                ),
            ],
        },
    }
def _health_actions(checks: dict, capabilities: dict) -> list[HealthAction]:
    actions: list[HealthAction] = []

    def add(label: str, detail: str, command: str = "", kind: str = "repair") -> None:
        actions.append(
            {
                "label": label,
                "detail": detail,
                "command": command,
                "kind": kind,
            }
        )

    if checks["sqlite"].get("status") != "ok":
        add(
            "检查本地记录库",
            "确认 SQLite 是否可读写；严格检查可能更慢，但能发现索引损坏。",
            "docflow doctor --strict",
        )
    if checks["qdrant"].get("status") != "ok":
        add(
            "恢复向量库",
            "确认 Docker Desktop 和 qdrant 容器已运行，然后再检查索引一致性。",
            "docker start qdrant && docflow admin check --json",
        )
    if checks["ollama"].get("status") != "ok":
        add(
            "打开 Ollama",
            "只有 OCR、Ollama 后端或 Ollama 上下文前缀需要；核心问答仍可使用 MLX。",
            "",
            kind="optional",
        )
    if not capabilities.get("ocr", False) and checks["ollama"].get("models", {}).get("ocr"):
        model = checks["ollama"]["models"]["ocr"].get("model", "glm-ocr")
        add(
            "准备扫描 PDF OCR",
            "只有扫描版 PDF 需要；普通 PDF、Markdown 和代码文件不受影响。",
            f"ollama pull {model}",
            kind="optional",
        )
    for model in checks["models"].get("missing_local_cache", []):
        add(
            f"准备本地模型：{model}",
            "模型未缓存时不要切换到它；联网后先准备缓存。",
            "",
            kind="optional",
        )

    if not actions:
        add(
            "检查索引一致性",
            "只读检查 SQLite 和 Qdrant 是否一致。",
            "docflow admin check --json",
            kind="safe",
        )
        add(
            "预览备份计划",
            "只查看将要备份的内容，不创建新备份。",
            "docflow admin backup --dry-run",
            kind="safe",
        )

    unique: dict[tuple[str, str], dict] = {}
    for action in actions:
        unique[(action["label"], action["command"])] = action
    return list(unique.values())
def _aggregate_health_status(checks: dict) -> str:
    critical = [checks["sqlite"]["status"], checks["qdrant"]["status"]]
    if any(status != "ok" for status in critical):
        return "unavailable"
    optional = [checks["ollama"]["status"], checks["models"]["status"]]
    if any(status != "ok" for status in optional):
        return "degraded"
    return "ok"
