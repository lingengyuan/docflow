from __future__ import annotations

import os
from pathlib import Path

import pytest

from src import model_cache


def test_remote_model_download_is_blocked_when_cache_missing(monkeypatch, tmp_path):
    monkeypatch.setenv("HUGGINGFACE_HUB_CACHE", str(tmp_path / "hub"))

    with pytest.raises(RuntimeError, match="privacy.allow_model_download"):
        model_cache.assert_model_download_allowed(
            "org/model",
            False,
            purpose="embedding",
        )


def test_remote_model_download_is_allowed_by_explicit_privacy_flag(monkeypatch, tmp_path):
    monkeypatch.setenv("HUGGINGFACE_HUB_CACHE", str(tmp_path / "hub"))

    model_cache.assert_model_download_allowed(
        "org/model",
        True,
        purpose="embedding",
    )


def test_cached_remote_model_does_not_require_download_flag(monkeypatch, tmp_path):
    cache = tmp_path / "hub"
    monkeypatch.setenv("HUGGINGFACE_HUB_CACHE", str(cache))
    (cache / "models--org--model" / "snapshots" / "abc123").mkdir(parents=True)

    model_cache.assert_model_download_allowed(
        "org/model",
        False,
        purpose="embedding",
    )


def test_resolve_model_load_reference_uses_cached_snapshot_when_downloads_blocked(
    monkeypatch, tmp_path
):
    cache = tmp_path / "hub"
    monkeypatch.setenv("HUGGINGFACE_HUB_CACHE", str(cache))
    older = cache / "models--org--model" / "snapshots" / "abc123"
    newer = cache / "models--org--model" / "snapshots" / "def456"
    older.mkdir(parents=True)
    newer.mkdir(parents=True)
    os.utime(older, (1, 1))
    os.utime(newer, (2, 2))

    resolved = model_cache.resolve_model_load_reference(
        "org/model",
        False,
        purpose="embedding",
    )

    assert resolved == str(newer)


def test_resolve_model_load_reference_keeps_remote_id_when_downloads_allowed(
    monkeypatch, tmp_path
):
    monkeypatch.setenv("HUGGINGFACE_HUB_CACHE", str(tmp_path / "hub"))

    resolved = model_cache.resolve_model_load_reference(
        "org/model",
        True,
        purpose="embedding",
    )

    assert resolved == "org/model"


def test_local_model_references_do_not_use_huggingface_cache(tmp_path):
    local_dir = tmp_path / "local-model"
    local_dir.mkdir()

    assert model_cache.is_remote_model_reference(local_dir) is False
    assert model_cache.is_remote_model_reference("qwen2.5:7b") is False


def test_configured_hf_model_status_reports_only_remote_models(monkeypatch, tmp_path):
    monkeypatch.setenv("HUGGINGFACE_HUB_CACHE", str(tmp_path / "hub"))
    cfg = {
        "embedding": {"model": "Qwen/Qwen3-Embedding-0.6B"},
        "reranker": {"model": "Qwen/Qwen3-Reranker-0.6B"},
        "llm": {"backend": "local", "ollama_model": "qwen2.5:7b"},
        "ollama": {"llm_model": "qwen2.5:7b"},
        "vlm": {"enabled": False, "model": "mlx-community/Qwen3-VL-8B-Instruct-4bit"},
    }

    statuses = model_cache.configured_hf_model_status(cfg)

    assert sorted(statuses) == ["embedding", "reranker"]
    assert statuses["embedding"]["cached"] is False
    assert Path(statuses["embedding"]["cache_path"]).name == "models--Qwen--Qwen3-Embedding-0.6B"
