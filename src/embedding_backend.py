from __future__ import annotations

import logging
import os
import re
import types
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger(__name__)

SUPPORTED_EMBEDDING_BACKENDS = {"torch", "onnx"}


@dataclass(frozen=True)
class EmbeddingBackendConfig:
    model_name: str
    device: str = "cpu"
    backend: str = "torch"
    onnx_provider: str = "CPUExecutionProvider"
    onnx_optimization: str | None = None
    onnx_quantization: str | None = None
    onnx_cache_dir: Path | None = None

    def normalized_backend(self) -> str:
        backend = self.backend.strip().lower()
        if backend == "pytorch":
            backend = "torch"
        if backend not in SUPPORTED_EMBEDDING_BACKENDS:
            raise ValueError(
                f"Unsupported embedding backend: {self.backend}. "
                f"Supported: {sorted(SUPPORTED_EMBEDDING_BACKENDS)}"
            )
        return backend

    def cache_key(self) -> str:
        backend = self.normalized_backend()
        parts = [backend, self.model_name]
        if backend == "onnx":
            parts.append(self.onnx_provider)
            if self.onnx_optimization:
                parts.append(f"opt={self.onnx_optimization}")
            if self.onnx_quantization:
                parts.append(f"quant={self.onnx_quantization}")
        return "::".join(parts)


def embedding_backend_config_from_dict(
    cfg: dict,
    config_path: str | Path,
) -> EmbeddingBackendConfig:
    embedding_cfg = cfg.get("embedding", {})
    config_dir = Path(config_path).expanduser().resolve().parent

    onnx_cache_dir_raw = embedding_cfg.get("onnx_cache_dir", "data/embedding_onnx")
    onnx_cache_dir = Path(onnx_cache_dir_raw).expanduser()
    if not onnx_cache_dir.is_absolute():
        onnx_cache_dir = config_dir / onnx_cache_dir

    return EmbeddingBackendConfig(
        model_name=embedding_cfg["model"],
        device=embedding_cfg.get("device", "cpu"),
        backend=embedding_cfg.get("backend", "torch"),
        onnx_provider=embedding_cfg.get("onnx_provider", "CPUExecutionProvider"),
        onnx_optimization=_clean_optional(embedding_cfg.get("onnx_optimization")),
        onnx_quantization=_clean_optional(embedding_cfg.get("onnx_quantization")),
        onnx_cache_dir=onnx_cache_dir,
    )


def load_embedding_model(config: EmbeddingBackendConfig):
    backend = config.normalized_backend()
    if backend == "onnx":
        return _load_onnx_model(config)
    return _load_torch_model(config)


def _load_torch_model(config: EmbeddingBackendConfig):
    import torch
    from sentence_transformers import SentenceTransformer

    n_threads = os.cpu_count() or 4
    torch.set_num_threads(n_threads)
    logger.info(f"[embedding] CPU threads: {n_threads}")
    logger.info(f"[embedding] Loading torch model: {config.model_name}")
    return SentenceTransformer(
        config.model_name,
        device=config.device,
        trust_remote_code=True,
    )


def _load_onnx_model(config: EmbeddingBackendConfig):
    from sentence_transformers import SentenceTransformer

    model_dir = _onnx_model_dir(config)
    file_name = _existing_requested_onnx_file(model_dir, config)
    if file_name is None and not _requested_onnx_variants(config):
        file_name = _preferred_onnx_file(model_dir, config)

    if file_name is None:
        logger.info(f"[embedding] Exporting ONNX model: {config.model_name}")
        try:
            model = SentenceTransformer(
                config.model_name,
                backend="onnx",
                trust_remote_code=True,
                model_kwargs={"provider": config.onnx_provider},
            )
        except Exception as exc:
            raise RuntimeError(
                f"Failed to load ONNX embedding model {config.model_name}. "
                "Check onnxruntime/optimum dependencies and model compatibility."
            ) from exc

        model_dir.mkdir(parents=True, exist_ok=True)
        model.save_pretrained(str(model_dir))
        file_name = _export_onnx_variants(model, model_dir, config)

    logger.info(
        "[embedding] Loading ONNX model: %s (%s)",
        config.model_name,
        file_name,
    )
    model = SentenceTransformer(
        str(model_dir),
        backend="onnx",
        trust_remote_code=True,
        model_kwargs={
            "provider": config.onnx_provider,
            "file_name": file_name,
        },
    )
    return _patch_required_onnx_inputs(model)


def _export_onnx_variants(model, model_dir: Path, config: EmbeddingBackendConfig) -> str:
    preferred_file = _existing_requested_onnx_file(model_dir, config)
    if preferred_file is not None:
        return preferred_file

    if config.onnx_optimization:
        try:
            from sentence_transformers import export_optimized_onnx_model

            export_optimized_onnx_model(
                model=model,
                optimization_config=config.onnx_optimization,
                model_name_or_path=str(model_dir),
            )
        except Exception as exc:
            logger.warning(
                "[embedding] ONNX optimization %s failed for %s: %s",
                config.onnx_optimization,
                config.model_name,
                exc,
            )

    if config.onnx_quantization:
        try:
            from sentence_transformers import export_dynamic_quantized_onnx_model

            export_dynamic_quantized_onnx_model(
                model=model,
                quantization_config=config.onnx_quantization,
                model_name_or_path=str(model_dir),
            )
        except Exception as exc:
            logger.warning(
                "[embedding] ONNX quantization %s failed for %s: %s",
                config.onnx_quantization,
                config.model_name,
                exc,
            )

    preferred_file = _preferred_onnx_file(model_dir, config)
    if preferred_file is None:
        raise RuntimeError(
            f"Failed to materialize an ONNX model for {config.model_name} at {model_dir}"
        )
    return preferred_file


def _preferred_onnx_file(model_dir: Path, config: EmbeddingBackendConfig) -> str | None:
    candidates = _requested_onnx_variants(config)
    candidates.extend(["onnx/model.onnx", "model.onnx"])

    for candidate in candidates:
        if (model_dir / candidate).exists():
            return candidate
    return None


def _existing_requested_onnx_file(model_dir: Path, config: EmbeddingBackendConfig) -> str | None:
    for candidate in _requested_onnx_variants(config):
        if (model_dir / candidate).exists():
            return candidate
    return None


def _onnx_model_dir(config: EmbeddingBackendConfig) -> Path:
    cache_dir = Path(config.onnx_cache_dir or "data/embedding_onnx")
    return cache_dir / _safe_model_dir_name(config.model_name)


def _patch_required_onnx_inputs(model):
    input_names = _onnx_input_names(model)
    if "position_ids" not in input_names:
        return model

    original_tokenize = model.tokenize

    def tokenize_with_position_ids(self, texts, **kwargs):
        features = original_tokenize(texts, **kwargs)
        if "position_ids" in features or "attention_mask" not in features:
            return features

        import torch

        attention_mask = features["attention_mask"]
        if not torch.is_tensor(attention_mask):
            return features

        position_ids = attention_mask.long().cumsum(-1) - 1
        position_ids = position_ids.masked_fill(attention_mask == 0, 0)
        features["position_ids"] = position_ids
        return features

    logger.info("[embedding] Injecting position_ids for ONNX model inputs")
    model.tokenize = types.MethodType(tokenize_with_position_ids, model)
    return model


def _onnx_input_names(model) -> set[str]:
    try:
        transformer = model[0]
        auto_model = transformer.auto_model
        input_names = getattr(auto_model, "input_names", None)
        if isinstance(input_names, dict):
            return set(input_names.keys())
        if isinstance(input_names, (list, tuple, set)):
            return set(input_names)
    except Exception:
        return set()
    return set()


def _safe_model_dir_name(model_name: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "-", model_name).strip("-")


def _clean_optional(value) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _requested_onnx_variants(config: EmbeddingBackendConfig) -> list[str]:
    candidates: list[str] = []
    if config.onnx_quantization:
        candidates.append(f"onnx/model_qint8_{config.onnx_quantization}.onnx")
    if config.onnx_optimization:
        candidates.append(f"onnx/model_{config.onnx_optimization}.onnx")
    return candidates
