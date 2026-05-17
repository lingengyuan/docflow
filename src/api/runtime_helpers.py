"""Runtime helper functions for the DocFlow API."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import yaml

from src.api.runtime import get_api_runtime
from src.api.state import LLMSwitchStatus
from src.config import ConfigError, DocFlowSettings
from src.config import WatchDirSettings as RuntimeWatchDirSettings
from src.ingest.watcher import WatchDir
from src.model_cache import (
    assert_model_download_allowed,
    configured_model_names,
    hf_cache_dir,
    hf_model_cache_path,
    is_hf_model_cached,
    is_remote_model_reference,
)

if TYPE_CHECKING:
    from src.ingest.store import DocStore


def _api():
    return get_api_runtime()


def _parse_watch_dirs(
    cfg: dict,
    config_path: str | Path | None = None,
) -> list[WatchDir]:
    """Parse configured watch folders through the typed runtime settings."""
    resolved_config_path = Path(config_path).expanduser() if config_path else _api().CONFIG_PATH
    settings = DocFlowSettings.from_mapping(cfg, resolved_config_path)
    if not settings.paths.watch_dirs:
        raise ConfigError("Missing required config value: paths.watch_dirs")
    return [_watch_dir_from_settings(item) for item in settings.paths.watch_dirs]


def _watch_dir_from_settings(item: RuntimeWatchDirSettings) -> WatchDir:
    return WatchDir(
        path=item.path,
        recursive=item.recursive,
        extensions=list(item.extensions),
    )


def _configured_model_names(cfg: dict) -> dict[str, str]:
    return configured_model_names(cfg)


def _hf_cache_dir() -> Path:
    return hf_cache_dir()


def _safe_path_size(path: Path, *, max_entries: int = 100_000) -> int:
    return _api().health_service.safe_path_size(path, max_entries=max_entries)


def _unique_existing_paths(paths: list[Path]) -> list[Path]:
    result: list[Path] = []
    seen: set[str] = set()
    for path in paths:
        expanded = path.expanduser()
        try:
            key = str(expanded.resolve())
        except OSError:
            key = str(expanded)
        if key in seen or not expanded.exists():
            continue
        seen.add(key)
        result.append(expanded)
    return result


def _config_path(raw: str | Path) -> Path:
    path = Path(raw).expanduser()
    if path.is_absolute():
        return path
    return _api().CONFIG_PATH.parent / path


def _configured_model_cache_paths(cfg: dict) -> list[Path]:
    paths: list[Path] = []
    model_names = {name for name in _configured_model_names(cfg).values() if name}
    for model_name in model_names:
        if is_remote_model_reference(model_name):
            paths.append(hf_model_cache_path(model_name))

    onnx_cache_dir = cfg.get("embedding", {}).get("onnx_cache_dir")
    if onnx_cache_dir:
        paths.append(_config_path(onnx_cache_dir))

    if any("/" not in name for name in model_names):
        paths.append(Path.home() / ".ollama" / "models")

    return _unique_existing_paths(paths)


def _source_file_usage(files: list[dict]) -> dict:
    return _api().health_service.source_file_usage(files)


def _app_data_paths(cfg: dict) -> list[Path]:
    paths_cfg = cfg.get("paths", {})
    candidates = [
        _config_path(paths_cfg.get("db_path", "docflow.db")),
        _config_path(paths_cfg.get("id_counter", "qdrant_id_counter.txt")),
        _config_path("qdrant_storage"),
    ]
    db_path = _config_path(paths_cfg.get("db_path", "docflow.db"))
    candidates.extend(
        [
            Path(f"{db_path}-wal"),
            Path(f"{db_path}-shm"),
        ]
    )
    return _unique_existing_paths(candidates)


def _collect_storage_usage(cfg: dict, doc_store: DocStore) -> dict:
    return _api().health_service.collect_storage_usage(
        cfg,
        doc_store,
        configured_model_cache_paths=_api()._configured_model_cache_paths,
        app_data_paths=_api()._app_data_paths,
        disk_usage=_api().shutil.disk_usage,
    )


def _is_hf_model_cached(model_name: str) -> bool:
    return is_hf_model_cached(model_name)


def _llm_model_status(model_name: str) -> dict:
    cached = (
        _api()._is_hf_model_cached(model_name)
        if is_remote_model_reference(model_name)
        else True
    )
    return {
        "model": model_name,
        "cached": cached,
        "available": cached,
        "current": bool(
            _api().query_engine and _api().query_engine.generator.current_model == model_name
        ),
        "detail": "本地缓存可用。" if cached else "本地缓存缺失，切换前需要先准备模型。",
        "actions": [] if cached else [f"联网后准备模型缓存：{model_name}"],
    }


def _set_llm_switch_state(
    state: LLMSwitchStatus,
    *,
    model: str | None = None,
    message: str = "",
) -> None:
    _api().llm_switch_state.set(state, model=model, message=message)


def _load_mlx_model_candidate(model_name: str):
    from mlx_lm import load

    with open(_api().CONFIG_PATH, encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}
    allow_model_download = bool(cfg.get("privacy", {}).get("allow_model_download", False))
    assert_model_download_allowed(
        model_name,
        allow_model_download,
        purpose="answer",
    )
    return load(model_name)
