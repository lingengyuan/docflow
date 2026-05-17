from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[1]


def read(path: str) -> str:
    return (ROOT / path).read_text(encoding="utf-8")


def workflow_uses_entries(node: Any) -> list[str]:
    entries: list[str] = []
    if isinstance(node, dict):
        uses = node.get("uses")
        if isinstance(uses, str):
            entries.append(uses)
        for value in node.values():
            entries.extend(workflow_uses_entries(value))
    elif isinstance(node, list):
        for value in node:
            entries.extend(workflow_uses_entries(value))
    return entries


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

    assert "permissions: read-all" not in workflow
    assert "contents: read" in workflow
    assert "security-events: write" in workflow
    assert "id-token: write" in workflow
    assert "actions: read" in workflow
    assert "persist-credentials: false" in workflow
    assert "ossf/scorecard-action@05b42c624433fc40578a4040d5cf5e36ddca8cde" in workflow
    assert "scorecard-results.sarif" in workflow
    assert "publish_results: true" in workflow
    assert "OpenSSF Scorecard" in release
    assert "OpenSSF Scorecard" in status
    assert "Release artifacts are not signed yet" in release


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
    assert "Release artifact tampering" in threat_model
    assert "Model weights are separate artifacts" in model_licenses
    assert "Qwen/Qwen3-Embedding-0.6B" in model_licenses
    assert "glm-ocr" in model_licenses
    assert "docs/threat-model.md" in security


def test_phase115_release_security_posture_is_pinned_and_documented():
    dockerfile = read("Dockerfile")
    compose = read("docker-compose.yml")
    image_compose = read("docker-compose.image.yml")
    docker_workflow = read(".github/workflows/docker-image.yml")
    python_package = read(".github/workflows/python-package.yml")
    release = read("docs/release.md")
    security = read("SECURITY.md")

    assert "python:3.12-slim@sha256:" in dockerfile
    assert "qdrant/qdrant:latest@sha256:" in compose
    assert "qdrant/qdrant:latest@sha256:" in image_compose
    assert "docker/build-push-action@10e90e3645eae34f1e60eeb005ba3a3d33f178e8" in docker_workflow
    assert "sbom: true" in docker_workflow
    assert "provenance: mode=max" in docker_workflow
    assert (
        "scripts/build_release_candidate.py --dist-dir dist --skip-build --json"
        in python_package
    )
    assert "SHA256SUMS" in read("scripts/build_release_candidate.py")
    assert "workflow actions, Docker bases, and Qdrant service images are still pinned" in release
    assert "Release artifacts are not signed yet" in release
    assert "Do not describe a release as signed" in release
    assert (
        "Keep GitHub Actions, Docker base images, and service container images pinned"
        in security
    )


def test_phase115_security_invariants_cover_future_workflows():
    workflow_dir = ROOT / ".github" / "workflows"
    for path in sorted([*workflow_dir.glob("*.yml"), *workflow_dir.glob("*.yaml")]):
        text = path.read_text(encoding="utf-8")
        workflow = yaml.safe_load(text)
        assert "permissions:" in text
        assert "permissions: read-all" not in text
        for uses in workflow_uses_entries(workflow):
            assert re.search(r"@[0-9a-f]{40}$", uses), f"{path}: {uses}"

    for path in ("docker-compose.yml", "docker-compose.image.yml"):
        text = read(path)
        assert "qdrant/qdrant:latest@sha256:" in text

    assert "python:3.12-slim@sha256:" in read("Dockerfile")
