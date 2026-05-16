"""Typed configuration loading for DocFlow runtime paths and quality gates."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from src.config_defaults import (
    DEFAULT_CONFIG_SECTIONS,
)
from src.config_ingest import IngestSettings
from src.config_query import QuerySettings


class ConfigError(ValueError): ...


@dataclass(frozen=True)
class WatchDirSettings:
    path: Path
    recursive: bool = True
    extensions: tuple[str, ...] = ()


@dataclass(frozen=True)
class PathsSettings:
    db_path: Path
    id_counter: Path
    watch_dirs: tuple[WatchDirSettings, ...]
    supported_extensions: tuple[str, ...]

    @classmethod
    def from_mapping(cls, cfg: dict[str, Any], config_dir: Path) -> PathsSettings:
        paths_cfg = _section(cfg, "paths")
        raw_watch_dirs = paths_cfg.get("watch_dirs", paths_cfg.get("watch_dir", []))
        if isinstance(raw_watch_dirs, (str, Path)):
            raw_watch_dirs = [{"path": str(raw_watch_dirs)}]

        watch_dirs = []
        for entry in raw_watch_dirs or []:
            if isinstance(entry, dict):
                raw_path = _required(entry, "path", "paths.watch_dirs[].path")
                recursive = bool(entry.get("recursive", True))
                extensions = tuple(str(value) for value in entry.get("extensions", []))
            else:
                raw_path = entry
                recursive = True
                extensions = ()
            watch_dirs.append(
                WatchDirSettings(
                    path=_resolve_path(raw_path, config_dir),
                    recursive=recursive,
                    extensions=extensions,
                )
            )

        return cls(
            db_path=_resolve_path(_required(paths_cfg, "db_path", "paths.db_path"), config_dir),
            id_counter=_resolve_path(
                paths_cfg.get("id_counter", "data/qdrant_id_counter.txt"),
                config_dir,
            ),
            watch_dirs=tuple(watch_dirs),
            supported_extensions=tuple(
                str(value) for value in paths_cfg.get("supported_extensions", [])
            ),
        )


@dataclass(frozen=True)
class QdrantSettings:
    host: str
    port: int
    collection: str = "docflow"

    @classmethod
    def from_mapping(cls, cfg: dict[str, Any]) -> QdrantSettings:
        qdrant_cfg = _section(cfg, "qdrant")
        return cls(
            host=str(_required(qdrant_cfg, "host", "qdrant.host")),
            port=int(_required(qdrant_cfg, "port", "qdrant.port")),
            collection=str(qdrant_cfg.get("collection", "docflow")),
        )


@dataclass(frozen=True)
class OllamaSettings:
    base_url: str = "http://localhost:11434"
    ocr_model: str = "glm-ocr"
    llm_model: str = "qwen2.5:7b"
    llm_model_enhanced: str = "qwen3:8b"

    @classmethod
    def from_mapping(cls, cfg: dict[str, Any]) -> OllamaSettings:
        ollama_cfg = cfg.get("ollama", {})
        return cls(
            base_url=str(ollama_cfg.get("base_url", "http://localhost:11434")),
            ocr_model=str(ollama_cfg.get("ocr_model", "glm-ocr")),
            llm_model=str(ollama_cfg.get("llm_model", "qwen2.5:7b")),
            llm_model_enhanced=str(ollama_cfg.get("llm_model_enhanced", "qwen3:8b")),
        )


@dataclass(frozen=True)
class EmbeddingSettings:
    model: str
    backend: str = "torch"
    batch_size: int = 32

    @classmethod
    def from_mapping(cls, cfg: dict[str, Any]) -> EmbeddingSettings:
        embedding_cfg = _section(cfg, "embedding")
        return cls(
            model=str(_required(embedding_cfg, "model", "embedding.model")),
            backend=str(embedding_cfg.get("backend", "torch")),
            batch_size=int(embedding_cfg.get("batch_size", 32)),
        )


@dataclass(frozen=True)
class ChunkingSettings:
    chunk_size: int = 512
    chunk_overlap: int = 51
    ocr_text_threshold: float = 0.10

    @classmethod
    def from_mapping(cls, cfg: dict[str, Any]) -> ChunkingSettings:
        chunking_cfg = _section(cfg, "chunking")
        return cls(
            chunk_size=int(chunking_cfg.get("chunk_size", 512)),
            chunk_overlap=int(chunking_cfg.get("chunk_overlap", 51)),
            ocr_text_threshold=float(chunking_cfg.get("ocr_text_threshold", 0.10)),
        )


@dataclass(frozen=True)
class RerankerSettings:
    model: str = "Qwen/Qwen3-Reranker-0.6B"
    instruction: str = ""

    @classmethod
    def from_mapping(cls, cfg: dict[str, Any]) -> RerankerSettings:
        reranker_cfg = cfg.get("reranker", {})
        return cls(
            model=str(reranker_cfg.get("model", "Qwen/Qwen3-Reranker-0.6B")),
            instruction=str(reranker_cfg.get("instruction", "")),
        )


@dataclass(frozen=True)
class LLMSettings:
    backend: str = "local"
    mlx_model: str = "mlx-community/Qwen3-4B-4bit"
    mlx_model_enhanced: str = "mlx-community/Qwen3-8B-4bit"
    ollama_model: str = "qwen2.5:7b"
    ollama_model_enhanced: str = "qwen3:8b"
    claude_model: str = "claude-sonnet-4-6"
    claude_api_key: str = ""

    @classmethod
    def from_mapping(cls, cfg: dict[str, Any], ollama: OllamaSettings) -> LLMSettings:
        llm_cfg = cfg.get("llm", {})
        return cls(
            backend=str(llm_cfg.get("backend", cfg.get("llm_backend", "local"))),
            mlx_model=str(llm_cfg.get("mlx_model", "mlx-community/Qwen3-4B-4bit")),
            mlx_model_enhanced=str(
                llm_cfg.get("mlx_model_enhanced", "mlx-community/Qwen3-8B-4bit")
            ),
            ollama_model=str(llm_cfg.get("ollama_model", ollama.llm_model)),
            ollama_model_enhanced=str(
                llm_cfg.get("ollama_model_enhanced", ollama.llm_model_enhanced)
            ),
            claude_model=str(llm_cfg.get("claude_model", "claude-sonnet-4-6")),
            claude_api_key=str(llm_cfg.get("claude_api_key", "")),
        )


@dataclass(frozen=True)
class PrivacySettings:
    allow_model_download: bool = False
    allowed_hosts: tuple[str, ...] = ()

    @classmethod
    def from_mapping(cls, cfg: dict[str, Any]) -> PrivacySettings:
        privacy_cfg = cfg.get("privacy", {})
        return cls(
            allow_model_download=bool(privacy_cfg.get("allow_model_download", False)),
            allowed_hosts=tuple(str(value) for value in privacy_cfg.get("allowed_hosts", [])),
        )


@dataclass(frozen=True)
class DocFlowSettings:
    config_path: Path
    config_dir: Path
    raw: dict[str, Any]
    paths: PathsSettings
    qdrant: QdrantSettings
    ollama: OllamaSettings
    embedding: EmbeddingSettings
    chunking: ChunkingSettings
    ingest: IngestSettings
    reranker: RerankerSettings
    llm: LLMSettings
    query: QuerySettings
    privacy: PrivacySettings

    @classmethod
    def from_file(cls, config_path: str | Path) -> DocFlowSettings:
        path = Path(config_path).expanduser()
        with path.open(encoding="utf-8") as f:
            raw = yaml.safe_load(f) or {}
        if not isinstance(raw, dict):
            raise ConfigError("config.yaml must contain a mapping at the top level")
        return cls.from_mapping(raw, path)

    @classmethod
    def from_mapping(
        cls,
        cfg: dict[str, Any],
        config_path: str | Path = "config.yaml",
    ) -> DocFlowSettings:
        cfg = _with_defaults(cfg)
        path = Path(config_path).expanduser()
        config_dir = path.resolve().parent
        ollama = OllamaSettings.from_mapping(cfg)
        return cls(
            config_path=path,
            config_dir=config_dir,
            raw=cfg,
            paths=PathsSettings.from_mapping(cfg, config_dir),
            qdrant=QdrantSettings.from_mapping(cfg),
            ollama=ollama,
            embedding=EmbeddingSettings.from_mapping(cfg),
            chunking=ChunkingSettings.from_mapping(cfg),
            ingest=IngestSettings.from_mapping(cfg, ollama.llm_model),
            reranker=RerankerSettings.from_mapping(cfg),
            llm=LLMSettings.from_mapping(cfg, ollama),
            query=QuerySettings.from_mapping(cfg),
            privacy=PrivacySettings.from_mapping(cfg),
        )


def _section(cfg: dict[str, Any], key: str) -> dict[str, Any]:
    value = cfg.get(key)
    if not isinstance(value, dict):
        raise ConfigError(f"Missing or invalid config section: {key}")
    return value


def _with_defaults(cfg: dict[str, Any]) -> dict[str, Any]:
    merged = {key: dict(value) if isinstance(value, dict) else value for key, value in cfg.items()}
    for key, defaults in DEFAULT_CONFIG_SECTIONS.items():
        _merge_section(merged, key, defaults)
    return merged


def _merge_section(merged: dict[str, Any], key: str, defaults: dict[str, Any]) -> None:
    value = merged.get(key)
    if value is None:
        merged[key] = dict(defaults)
        return
    if not isinstance(value, dict):
        raise ConfigError(f"Missing or invalid config section: {key}")
    merged[key] = {**defaults, **value}


def _required(section: dict[str, Any], key: str, label: str) -> Any:
    value = section.get(key)
    if value is None or value == "":
        raise ConfigError(f"Missing required config value: {label}")
    return value


def _resolve_path(value: str | Path, config_dir: Path) -> Path:
    path = Path(value).expanduser()
    return path if path.is_absolute() else config_dir / path
