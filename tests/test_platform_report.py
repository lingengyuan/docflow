from __future__ import annotations

from src.maintenance.platform import build_platform_report


def test_platform_report_is_machine_readable():
    report = build_platform_report("config.example.yaml")

    assert report["python"]["version"]
    assert report["os"]["system"]
    assert report["configured"]["llm_backend"] == "local"
    assert report["capabilities"]["base_runtime"] is True
    assert "mlx_answers" in report["capabilities"]
    assert "mlx_lm" in report["optional_packages"]


def test_default_config_is_not_mlx_only():
    report = build_platform_report("config.example.yaml")

    assert report["configured"]["llm_backend"] in {"local", "ollama"}
    assert report["capabilities"]["ollama_compatible_answers"] is True
    assert report["warnings"] == []
