"""Model cache checks used before loading local Hugging Face backed models."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any


def hf_cache_dir() -> Path:
    hub_cache = os.getenv("HUGGINGFACE_HUB_CACHE")
    if hub_cache:
        return Path(hub_cache).expanduser()
    hf_home = Path(os.getenv("HF_HOME", str(Path.home() / ".cache" / "huggingface"))).expanduser()
    return hf_home / "hub"


def is_remote_model_reference(model_name: str | Path | None) -> bool:
    if not model_name:
        return False
    text = str(model_name).strip()
    if not text or Path(text).expanduser().exists():
        return False
    return "/" in text and not text.startswith(("./", "../"))


def hf_model_cache_path(model_name: str) -> Path:
    return hf_cache_dir() / f"models--{model_name.replace('/', '--')}"


def is_hf_model_cached(model_name: str | Path | None) -> bool:
    if not is_remote_model_reference(model_name):
        return True
    model_dir = hf_model_cache_path(str(model_name))
    snapshots_dir = model_dir / "snapshots"
    if not snapshots_dir.exists():
        return False
    return any(snap.is_dir() for snap in snapshots_dir.iterdir())


def assert_model_download_allowed(
    model_name: str | Path | None,
    allow_model_download: bool,
    *,
    purpose: str,
) -> None:
    if not is_remote_model_reference(model_name):
        return
    if is_hf_model_cached(model_name):
        return
    if allow_model_download:
        return
    raise RuntimeError(
        f"{purpose} model cache is missing: {model_name}. "
        "Model downloads are disabled by privacy.allow_model_download."
    )


def configured_model_names(cfg: dict[str, Any]) -> dict[str, str]:
    ollama_cfg = cfg.get("ollama", {})
    llm_cfg = cfg.get("llm", {})
    embedding_cfg = cfg.get("embedding", {})
    reranker_cfg = cfg.get("reranker", {})
    vlm_cfg = cfg.get("vlm", {})
    ingest_cfg = cfg.get("ingest", {})
    backend = str(llm_cfg.get("backend", "local")).strip().lower()
    llm_model = (
        llm_cfg.get("mlx_model", "")
        if backend == "mlx"
        else llm_cfg.get("ollama_model") or ollama_cfg.get("llm_model", "")
    )
    llm_enhanced = (
        llm_cfg.get("mlx_model_enhanced", "")
        if backend == "mlx"
        else llm_cfg.get("ollama_model_enhanced") or ollama_cfg.get("llm_model_enhanced", "")
    )
    return {
        "embedding": embedding_cfg.get("model", ""),
        "reranker": reranker_cfg.get("model", ""),
        "llm": llm_model,
        "llm_enhanced": llm_enhanced,
        "ocr": ollama_cfg.get("ocr_model", ""),
        "contextual_prefix": ingest_cfg.get("contextual_prefix_model", ""),
        "vlm": vlm_cfg.get("model", "") if vlm_cfg.get("enabled", True) else "",
    }


def configured_hf_model_status(cfg: dict[str, Any]) -> dict[str, dict[str, Any]]:
    statuses: dict[str, dict[str, Any]] = {}
    for purpose, model in configured_model_names(cfg).items():
        if not is_remote_model_reference(model):
            continue
        cached = is_hf_model_cached(model)
        statuses[purpose] = {
            "model": model,
            "cached": cached,
            "cache_path": str(hf_model_cache_path(model)),
        }
    return statuses
