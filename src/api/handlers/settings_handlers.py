from __future__ import annotations

import sys

import yaml
from fastapi import HTTPException

from src.api.model_tasks import ModelTaskTimeout
from src.api.schemas import LLMSwitchRequest


def _api():
    return sys.modules["src.api.app_impl"]


async def get_llm():
    if _api().query_engine is None:
        raise HTTPException(503, "Not ready")
    backend = _api().query_engine.generator.backend
    return {
        "current": _api().query_engine.generator.current_model,
        "options": _api().llm_options,
        "models": [_api()._llm_model_status(model) for model in _api().llm_options],
        "backend": backend,
        "network_mode": "cloud" if backend == "claude" else "local",
        "privacy_notice": (
            "云端回答已启用。提问内容会发送到你配置的外部模型服务。"
            if backend == "claude"
            else "本地回答已启用。默认不会把提问发送到外部模型服务。"
        ),
        "switch": dict(_api().llm_switch_state),
    }


async def set_llm(req: LLMSwitchRequest):
    if _api().query_engine is None:
        raise HTTPException(503, "Not ready")
    if req.model not in _api().llm_options:
        raise HTTPException(400, f"Unknown model: {req.model}. Available: {_api().llm_options}")
    gen = _api().query_engine.generator
    if req.model == gen.current_model:
        _api()._set_llm_switch_state("idle", model=req.model, message="Already using this model")
        return {"ok": True, "model": req.model, "unchanged": True}
    model_status = _api()._llm_model_status(req.model)
    if gen.backend == "mlx" and not model_status["available"]:
        message = f"Model is not cached locally: {req.model}"
        _api()._set_llm_switch_state("error", model=req.model, message=message)
        raise HTTPException(409, message)
    _api()._set_llm_switch_state("switching", model=req.model)
    if gen.backend == "mlx":
        try:
            loaded_model, loaded_tokenizer = await _api().model_tasks.run(
                "llm_switch",
                lambda: _api()._load_mlx_model_candidate(req.model),
                timeout_s=_api().MODEL_TASK_TIMEOUT_S,
            )
            gen._mlx_model = loaded_model
            gen._mlx_tokenizer = loaded_tokenizer
            gen.mlx_model_name = req.model
        except ModelTaskTimeout as exc:
            _api().logger.warning("[api/llm] switch timeout id=%s model=%s", exc.task_id, req.model)
            _api()._set_llm_switch_state(
                "error",
                model=req.model,
                message=_api().MODEL_TIMEOUT_MESSAGE,
            )
            raise HTTPException(504, _api().MODEL_TIMEOUT_MESSAGE) from exc
        except Exception as exc:
            message = str(exc) or "Model switch failed"
            _api().logger.exception("[api/llm] switch failed model=%s", req.model)
            _api()._set_llm_switch_state("error", model=req.model, message=message)
            raise HTTPException(500, message) from exc
    elif gen.backend == "claude":
        gen.claude_model = req.model
    else:
        gen.ollama_model = req.model
    _api()._set_llm_switch_state("idle", model=req.model, message="Switched")
    _api().logger.info(f"[llm] Switched to {req.model}")
    return {"ok": True, "model": req.model}


async def list_sources():
    """返回所有监控目录配置。"""
    return [
        {
            "path": str(wd.path),
            "recursive": wd.recursive,
            "extensions": wd.extensions or _api().pipeline.registry.supported_extensions,
        }
        for wd in _api().watch_dirs
    ]


async def health():
    with open(_api().CONFIG_PATH, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    return _api().health_service.build_health(
        cfg,
        timed_check=_api()._timed_check,
        check_sqlite=_api()._check_sqlite,
        check_qdrant=_api()._check_qdrant,
        check_ollama=_api()._check_ollama,
        check_models=_api()._check_models,
        health_capabilities=_api()._health_capabilities,
        aggregate_health_status=_api()._aggregate_health_status,
        health_groups=_api()._health_groups,
        health_actions=_api()._health_actions,
    )
