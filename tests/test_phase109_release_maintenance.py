from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def read(path: str) -> str:
    return (ROOT / path).read_text(encoding="utf-8")


def test_no_build_docker_image_path_is_public_default():
    compose = read("docker-compose.image.yml")
    workflow = read(".github/workflows/docker-image.yml")
    readme = read("README.md")

    assert "ghcr.io/lingengyuan/docflow:edge" in compose
    assert "branches: [main]" in workflow
    assert "type=raw,value=edge" in workflow
    assert "docker compose -f docker-compose.image.yml up" in readme


def test_openssf_scorecard_is_part_of_release_surface():
    workflow = read(".github/workflows/scorecard.yml")
    release = read("docs/release.md")
    status = read("docs/status.md")

    assert "ossf/scorecard-action" in workflow
    assert "scorecard-results.sarif" in workflow
    assert "publish_results: true" in workflow
    assert "OpenSSF Scorecard" in release
    assert "OpenSSF Scorecard" in status


def test_dependency_layers_are_documented():
    development = read("docs/development.md")

    for path in (
        "requirements-core.txt",
        "requirements-local-model.txt",
        "requirements.txt",
        "requirements-vision.txt",
        "requirements-mac-mlx.txt",
        "requirements-dev.txt",
    ):
        assert (ROOT / path).exists()
        assert path in development


def test_threat_model_and_model_license_boundaries_are_public():
    threat_model = read("docs/threat-model.md")
    model_licenses = read("docs/model-licenses.md")
    security = read("SECURITY.md")

    assert "Browser to DocFlow API" in threat_model
    assert "DocFlow to the internet" in threat_model
    assert "OpenSSF Scorecard" in threat_model
    assert "Model weights are separate artifacts" in model_licenses
    assert "Qwen/Qwen3-Embedding-0.6B" in model_licenses
    assert "glm-ocr" in model_licenses
    assert "docs/threat-model.md" in security
