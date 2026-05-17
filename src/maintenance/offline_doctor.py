"""Offline network doctor checks for local-only DocFlow workflows."""

from __future__ import annotations

import json
import logging
import tempfile
from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import yaml

from src import net
from src.model_cache import configured_hf_model_status

if TYPE_CHECKING:
    from src.ingest.chunker import Chunk


def _load_config_for_offline_guard(config_path: str | Path) -> dict:
    try:
        from src.maintenance.startup import load_config

        cfg, _ = load_config(config_path)
    except (OSError, ValueError, yaml.YAMLError):
        return {}
    return cfg


def _run_offline_runtime_checks(cfg: dict) -> list[dict]:
    return [
        _runtime_check("startup", lambda: {"covered": True}),
        _runtime_check("ingest", _offline_ingest_probe),
        _runtime_check("query", _offline_query_probe),
        _runtime_check("model status", lambda: _offline_model_status_probe(cfg)),
        _runtime_check("source preview", _offline_source_preview_probe),
    ]


def _runtime_check(name: str, check: Callable[[], dict]) -> dict:
    try:
        result = check()
    except Exception as exc:
        return {"name": name, "status": "unavailable", "error": str(exc)}
    status = str(result.pop("status", "ok"))
    return {"name": name, "status": status, **result}


class _OfflineFakeEmbedder:
    embedding_cache_key = "offline-check"

    def encode_texts(self, texts: list[str], progress_callback=None) -> np.ndarray:
        if progress_callback:
            progress_callback(
                {
                    "encoded_texts": len(texts),
                    "batch_size": len(texts),
                }
            )
        return np.ones((len(texts), 3), dtype=np.float32)

    def upsert_embeddings(
        self,
        chunks: list[Chunk],
        dense_vecs: np.ndarray,
        min_next_id: int | None = 1,
    ) -> list[int]:
        del dense_vecs
        start = int(min_next_id or 1)
        return list(range(start, start + len(chunks)))

    def delete_file_vectors(self, qdrant_ids: list[int]) -> None:
        del qdrant_ids

    def max_point_id(self) -> int:
        return 0

    def close(self) -> None:
        return None


def _offline_ingest_probe() -> dict:
    from src.ingest.chunker import StructuredChunker
    from src.ingest.parsers import ParserRegistry
    from src.ingest.pipeline import IngestPipeline
    from src.ingest.store import DocStore

    pipeline_logger = logging.getLogger("src.ingest.pipeline")
    previous_disabled = pipeline_logger.disabled
    pipeline_logger.disabled = True
    pipeline = None
    store = None
    try:
        with tempfile.TemporaryDirectory(prefix="docflow-offline-ingest-") as tmp:
            root = Path(tmp)
            sample = root / "offline-check.md"
            sample.write_text(
                "# Offline Check\n\nDocFlow keeps this check local.",
                encoding="utf-8",
            )
            store = DocStore(root / "docflow.db")
            registry = ParserRegistry.from_config(
                {
                    "ollama": {"base_url": "http://localhost:11434", "ocr_model": "glm-ocr"},
                    "paths": {"supported_extensions": [".md"]},
                    "vlm": {"enabled": False},
                }
            )
            pipeline = IngestPipeline(
                registry=registry,
                chunker=StructuredChunker(chunk_size=256, chunk_overlap=20),
                embedder=_OfflineFakeEmbedder(),
                store=store,
                use_embedding_cache=False,
            )
            result = pipeline.ingest(sample)
            if result.get("status") != "done":
                raise RuntimeError(str(result))
            return {"file": result.get("file", ""), "chunks": int(result.get("chunks", 0))}
    finally:
        if pipeline is not None:
            pipeline.close()
        if store is not None:
            store.close()
        pipeline_logger.disabled = previous_disabled


def _offline_query_probe() -> dict:
    from src.query.generator import AnswerGenerator

    answer = AnswerGenerator().generate("What can DocFlow answer offline?", [])
    return {"answer_chars": len(answer.text), "citations": len(answer.citations)}


def _offline_model_status_probe(cfg: dict) -> dict:
    statuses = configured_hf_model_status(cfg)
    missing = [item["model"] for item in statuses.values() if not item["cached"]]
    allow_download = bool(cfg.get("privacy", {}).get("allow_model_download", False))
    return {
        "status": "degraded" if missing else "ok",
        "checked_models": len(statuses),
        "missing_cache": missing,
        "allow_model_download": allow_download,
    }


def _offline_source_preview_probe() -> dict:
    from src.domain_types import FileStatus
    from src.ingest.store import DocStore

    with tempfile.TemporaryDirectory(prefix="docflow-offline-preview-") as tmp:
        root = Path(tmp)
        sample = root / "source-preview.md"
        text = "DocFlow source preview stays on this machine."
        sample.write_text(text, encoding="utf-8")
        store = DocStore(root / "docflow.db")
        try:
            file_id = store.upsert_file(
                file_path=sample,
                file_name=sample.name,
                file_hash=DocStore.compute_hash(sample),
                status=FileStatus.DONE,
                total_pages=1,
                mtime_ns=sample.stat().st_mtime_ns,
            )
            store.add_chunks(
                file_id,
                [
                    {
                        "qdrant_id": 1,
                        "chunk_type": "text",
                        "page_num": 1,
                        "section": "",
                        "char_count": len(text),
                        "raw_text": text,
                        "embedding_text": text,
                        "tokenized_text": text.lower(),
                    }
                ],
            )
            chunks = store.list_file_chunks(file_id)
            if not chunks or chunks[0].get("raw_text") != text:
                raise RuntimeError("source preview chunk was not readable")
            return {"chunks": len(chunks), "file": sample.name}
        finally:
            store.close()


def build_offline_report(
    config_path: str | Path = "config.yaml",
    app_port: int = 8000,
) -> dict:
    from src.maintenance.startup import build_startup_report

    cfg = _load_config_for_offline_guard(config_path)
    allowed_hosts = net.configured_allowed_hosts(cfg)
    with net.NetworkGuard(allowed_hosts=allowed_hosts) as guard:
        try:
            startup_report = build_startup_report(config_path=config_path, app_port=app_port)
            from src.maintenance import startup

            runtime_checks = startup._run_offline_runtime_checks(cfg)
            error = ""
        except Exception as exc:
            startup_report = {}
            runtime_checks = []
            error = str(exc)
    unexpected_hosts = sorted(guard.unexpected_hosts)
    failed_runtime_checks = [
        check["name"] for check in runtime_checks if check.get("status") == "unavailable"
    ]
    return {
        "status": "ok" if not unexpected_hosts and not failed_runtime_checks else "unavailable",
        "unexpected_outbound_connections": len(unexpected_hosts),
        "unexpected_hosts": unexpected_hosts,
        "allowed_hosts": sorted(allowed_hosts),
        "error": error,
        "runtime_checks": runtime_checks,
        "failed_runtime_checks": failed_runtime_checks,
        "network_registry": net.network_access_registry(),
        "startup_report": startup_report,
    }


def format_offline_report(report: dict) -> str:
    if report["unexpected_outbound_connections"] == 0:
        lines = ["DocFlow offline network check: ok", "0 unexpected outbound connections"]
    else:
        lines = [
            "DocFlow offline network check: unavailable",
            f"{report['unexpected_outbound_connections']} unexpected hosts: "
            + ", ".join(report["unexpected_hosts"]),
        ]
    if report.get("error"):
        lines.append(f"Error: {report['error']}")
    runtime_checks = report.get("runtime_checks") or []
    if runtime_checks:
        covered = ", ".join(str(check.get("name", "")) for check in runtime_checks)
        lines.append(f"Covered local paths: {covered}")
    failed = report.get("failed_runtime_checks") or []
    if failed:
        lines.append(f"Runtime checks failed: {', '.join(failed)}")
    registry = report.get("network_registry") or []
    if registry:
        cases = ", ".join(str(item.get("id", "")) for item in registry)
        lines.append(f"Registered network cases: {cases}")
    return "\n".join(lines)


def offline_doctor_command(
    config_path: str | Path = "config.yaml",
    app_port: int = 8000,
    as_json: bool = False,
) -> int:
    report = build_offline_report(config_path=config_path, app_port=app_port)
    print(
        json.dumps(report, ensure_ascii=False, indent=2)
        if as_json
        else format_offline_report(report)
    )
    return 0 if report["status"] == "ok" else 1
