"""Platform capability report for local DocFlow installs."""

from __future__ import annotations

import importlib.util
import json
import platform
import sys
from pathlib import Path
from typing import Any

import yaml

from src.maintenance.startup import ensure_config_file


def _package_available(module_name: str) -> bool:
    return importlib.util.find_spec(module_name) is not None


def _load_config(config_path: str | Path = "config.yaml") -> dict[str, Any]:
    path = ensure_config_file(config_path)
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def build_platform_report(config_path: str | Path = "config.yaml") -> dict[str, Any]:
    cfg = _load_config(config_path)
    machine = platform.machine().lower()
    system = platform.system().lower()
    is_apple_silicon = system == "darwin" and machine in {"arm64", "aarch64"}
    llm_backend = str(cfg.get("llm", {}).get("backend", "local")).strip().lower()
    optional_packages = {
        "mlx": _package_available("mlx"),
        "mlx_lm": _package_available("mlx_lm"),
        "onnxruntime": _package_available("onnxruntime"),
        "torch": _package_available("torch"),
        "playwright": _package_available("playwright"),
    }
    mlx_ready = is_apple_silicon and optional_packages["mlx"] and optional_packages["mlx_lm"]
    warnings: list[str] = []
    if llm_backend == "mlx" and not mlx_ready:
        warnings.append("MLX backend is configured but this platform or install is not MLX-ready.")

    return {
        "python": {
            "version": platform.python_version(),
            "implementation": platform.python_implementation(),
            "executable": sys.executable,
        },
        "os": {
            "system": platform.system(),
            "release": platform.release(),
            "machine": platform.machine(),
            "apple_silicon": is_apple_silicon,
        },
        "configured": {
            "llm_backend": llm_backend,
            "embedding_backend": cfg.get("embedding", {}).get("backend", ""),
            "vlm_enabled": bool(cfg.get("vlm", {}).get("enabled", False)),
        },
        "capabilities": {
            "base_runtime": True,
            "ollama_compatible_answers": llm_backend in {"local", "ollama"},
            "mlx_answers": mlx_ready,
            "mlx_reranker": mlx_ready,
            "onnx_embeddings": optional_packages["onnxruntime"],
            "torch_embeddings": optional_packages["torch"],
        },
        "optional_packages": optional_packages,
        "warnings": warnings,
    }


def platform_command(args: list[str]) -> int:
    as_json = "--json" in args
    report = build_platform_report()
    if as_json:
        print(json.dumps(report, ensure_ascii=False, indent=2))
    else:
        print(f"Python: {report['python']['version']} ({report['python']['implementation']})")
        print(
            "OS: "
            f"{report['os']['system']} {report['os']['release']} {report['os']['machine']}"
        )
        print(f"Configured answer backend: {report['configured']['llm_backend']}")
        print(f"Base runtime: {'ok' if report['capabilities']['base_runtime'] else 'unavailable'}")
        print(f"MLX ready: {'yes' if report['capabilities']['mlx_answers'] else 'no'}")
        for warning in report["warnings"]:
            print(f"Warning: {warning}")
    return 0 if not report["warnings"] else 1
