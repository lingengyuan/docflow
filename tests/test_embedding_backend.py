from pathlib import Path

import pytest
import torch

from src.embedding_backend import (
    EmbeddingBackendConfig,
    _patch_required_onnx_inputs,
    _preferred_onnx_file,
    embedding_backend_config_from_dict,
    load_embedding_model,
)


def test_config_from_dict_resolves_relative_onnx_cache_dir(tmp_path):
    config_path = tmp_path / "config.yaml"
    config_path.write_text("embedding: {}\n", encoding="utf-8")
    cfg = {
        "embedding": {
            "model": "Qwen/Qwen3-Embedding-0.6B",
            "backend": "onnx",
            "device": "cpu",
            "onnx_cache_dir": "data/embedding_onnx",
        }
    }

    config = embedding_backend_config_from_dict(cfg, config_path)

    assert config.onnx_cache_dir == tmp_path / "data/embedding_onnx"


def test_preferred_onnx_file_prefers_quantized_then_optimized(tmp_path):
    model_dir = tmp_path / "model"
    onnx_dir = model_dir / "onnx"
    onnx_dir.mkdir(parents=True)
    (onnx_dir / "model_O3.onnx").write_text("", encoding="utf-8")
    (onnx_dir / "model_qint8_arm64.onnx").write_text("", encoding="utf-8")

    config = EmbeddingBackendConfig(
        model_name="Qwen/Qwen3-Embedding-0.6B",
        backend="onnx",
        onnx_optimization="O3",
        onnx_quantization="arm64",
        onnx_cache_dir=tmp_path,
    )

    assert _preferred_onnx_file(model_dir, config) == "onnx/model_qint8_arm64.onnx"


def test_load_embedding_model_reloads_local_onnx_export(monkeypatch, tmp_path):
    calls = []

    class FakeSentenceTransformer:
        def __init__(self, model_name_or_path, **kwargs):
            calls.append((model_name_or_path, kwargs))

        def save_pretrained(self, path):
            export_dir = Path(path) / "onnx"
            export_dir.mkdir(parents=True, exist_ok=True)
            (export_dir / "model.onnx").write_text("base", encoding="utf-8")

    def fake_export_optimized_onnx_model(model, optimization_config, model_name_or_path):
        export_dir = Path(model_name_or_path) / "onnx"
        export_dir.mkdir(parents=True, exist_ok=True)
        (export_dir / f"model_{optimization_config}.onnx").write_text("optimized", encoding="utf-8")

    import sentence_transformers
    monkeypatch.setattr(sentence_transformers, "SentenceTransformer", FakeSentenceTransformer)
    monkeypatch.setattr(
        sentence_transformers,
        "export_optimized_onnx_model",
        fake_export_optimized_onnx_model,
    )

    config = EmbeddingBackendConfig(
        model_name="Qwen/Qwen3-Embedding-0.6B",
        backend="onnx",
        onnx_provider="CPUExecutionProvider",
        onnx_optimization="O3",
        onnx_cache_dir=tmp_path,
    )

    load_embedding_model(config)

    assert calls[0][0] == "Qwen/Qwen3-Embedding-0.6B"
    assert calls[0][1]["backend"] == "onnx"
    assert calls[1][0].startswith(str(tmp_path))
    assert calls[1][1]["model_kwargs"]["file_name"] == "onnx/model_O3.onnx"


def test_embedding_cache_key_distinguishes_onnx_and_torch():
    torch_config = EmbeddingBackendConfig(model_name="Qwen/Qwen3-Embedding-0.6B", backend="torch")
    onnx_config = EmbeddingBackendConfig(
        model_name="Qwen/Qwen3-Embedding-0.6B",
        backend="onnx",
        onnx_provider="CPUExecutionProvider",
        onnx_optimization="O3",
    )

    assert torch_config.cache_key() != onnx_config.cache_key()
    assert onnx_config.cache_key().startswith("onnx::")


def test_invalid_backend_rejected():
    config = EmbeddingBackendConfig(model_name="Qwen/Qwen3-Embedding-0.6B", backend="bad")
    with pytest.raises(ValueError):
        config.normalized_backend()


def test_patch_required_onnx_inputs_injects_position_ids():
    class FakeAutoModel:
        input_names = {"input_ids": 0, "attention_mask": 1, "position_ids": 2}

    class FakeTransformer:
        auto_model = FakeAutoModel()

    class FakeModel:
        def __getitem__(self, index):
            assert index == 0
            return FakeTransformer()

        def tokenize(self, texts, **kwargs):
            assert texts == ["hello"]
            return {
                "input_ids": torch.tensor([[1, 2, 3]]),
                "attention_mask": torch.tensor([[1, 1, 0]]),
            }

    model = _patch_required_onnx_inputs(FakeModel())
    features = model.tokenize(["hello"])

    assert "position_ids" in features
    assert features["position_ids"].tolist() == [[0, 1, 0]]
