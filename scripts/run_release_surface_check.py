#!/usr/bin/env python3

from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[1]
ACTION_PIN_PATTERN = re.compile(r"@[0-9a-f]{40}$")

REQUIRED_ROOT_FILES = [
    ".github/ISSUE_TEMPLATE/bug.md",
    ".github/ISSUE_TEMPLATE/feature.md",
    ".github/ISSUE_TEMPLATE/question.md",
    ".github/PULL_REQUEST_TEMPLATE.md",
    ".github/dependabot.yml",
    "CHANGELOG.md",
    "CODE_OF_CONDUCT.md",
    "CONTRIBUTING.md",
    "Dockerfile",
    "LICENSE",
    "README.md",
    "README.zh-CN.md",
    "ROADMAP.md",
    "SECURITY.md",
    "config.docker.yaml",
    "config.example.yaml",
    "docker-compose.image.yml",
    "docker-compose.yml",
    "pyproject.toml",
    "requirements-core.txt",
    "requirements-dev.txt",
    "requirements-local-model.txt",
    "requirements-mac-mlx.txt",
    "requirements-vision.txt",
    "requirements.txt",
    "scripts/build_release_candidate.py",
]

PUBLIC_DOCS = [
    "architecture.md",
    "cli.md",
    "development.md",
    "evaluation.md",
    "features.md",
    "model-licenses.md",
    "privacy.md",
    "release.md",
    "status.md",
    "threat-model.md",
]

PUBLIC_ADRS = [
    "README.md",
    "0001-module-boundaries.md",
    "0002-local-first-no-telemetry.md",
    "0003-third-party-integration-scope.md",
]

DISALLOWED_PUBLIC_DOC_REFS = [
    "docs/critique-2026-05.md",
    "docs/improvement-roadmap.md",
    "docs/scoring-2026-05.md",
    "critique-2026-05.md",
    "improvement-roadmap.md",
    "scoring-2026-05.md",
]

DISALLOWED_LEGACY_SCORE_FILES = [
    "eval/phase11_maturity_dimensions.json",
    "eval/phase11_questions.jsonl",
]

DISALLOWED_SUBJECTIVE_SCORE_PHRASES = [
    "rolling 9-point maturity scorecard",
    "9-point maturity scorecard",
    "9-point maturity",
    "maturity scorecard",
]

DISALLOWED_TRACKED_PATHS = [
    "obsidian-plugin/",
    "frontend/js/pwa.js",
    "frontend/sw.js",
    "frontend/manifest.webmanifest",
    "src/api/routes/obsidian.py",
    "src/api/handlers/obsidian_handlers.py",
    "tests/test_obsidian_api.py",
    "tests/test_obsidian_plugin.py",
]

DISALLOWED_SCOPE_TERMS = [
    "obsidian-plugin",
    "/api/obsidian",
    "serviceWorker",
    "manifest.webmanifest",
    "pwa.js",
]

REQUIRED_WORKFLOWS = [
    ".github/workflows/ci.yml",
    ".github/workflows/codeql.yml",
    ".github/workflows/dependency-audit.yml",
    ".github/workflows/docker-image.yml",
    ".github/workflows/evaluation.yml",
    ".github/workflows/python-package.yml",
    ".github/workflows/scorecard.yml",
]


def read(path: str | Path) -> str:
    return (ROOT / path).read_text(encoding="utf-8")


def fail(message: str) -> None:
    print(f"Release surface check failed: {message}", file=sys.stderr)
    raise SystemExit(1)


def require_files(paths: list[str]) -> None:
    missing = [path for path in paths if not (ROOT / path).exists()]
    if missing:
        fail(f"missing required files: {', '.join(missing)}")


def require_snippets(path: str, snippets: list[str]) -> None:
    text = read(path)
    missing = [snippet for snippet in snippets if snippet not in text]
    if missing:
        fail(f"{path} is missing required text: {', '.join(missing)}")


def workflow_documents() -> list[tuple[Path, Any]]:
    documents: list[tuple[Path, Any]] = []
    workflow_dir = ROOT / ".github" / "workflows"
    for path in sorted([*workflow_dir.glob("*.yml"), *workflow_dir.glob("*.yaml")]):
        documents.append((path, yaml.safe_load(path.read_text(encoding="utf-8"))))
    return documents


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


def git_ls_files() -> list[str]:
    try:
        result = subprocess.run(
            ["git", "-C", str(ROOT), "ls-files"],
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError) as exc:
        fail(f"could not inspect tracked files with git: {exc}")
    return [line.strip() for line in result.stdout.splitlines() if line.strip()]


def check_public_docs() -> None:
    docs = sorted(path.name for path in (ROOT / "docs").glob("*.md"))
    if docs != PUBLIC_DOCS:
        fail(f"docs root must stay public and focused: expected {PUBLIC_DOCS}, got {docs}")

    tracked = git_ls_files()
    internal_prefixes = ("docs/history/", "plans/", "drafts/")
    leaked = [path for path in tracked if path.startswith(internal_prefixes)]
    if leaked:
        fail(f"internal planning files are tracked: {', '.join(leaked[:10])}")
    legacy_score_files = [path for path in tracked if path in DISALLOWED_LEGACY_SCORE_FILES]
    if legacy_score_files:
        fail(
            "legacy subjective-score files are tracked: "
            + ", ".join(legacy_score_files)
        )
    out_of_scope_paths = [
        path
        for path in tracked
        if any(path == blocked or path.startswith(blocked) for blocked in DISALLOWED_TRACKED_PATHS)
    ]
    if out_of_scope_paths:
        fail(
            "out-of-scope integration or PWA files are tracked: "
            + ", ".join(out_of_scope_paths[:10])
        )

    gitignore = read(".gitignore")
    for ignored in ("docs/history/", "output/", ".cache/", "config.yaml", "qdrant_storage/"):
        if ignored not in gitignore:
            fail(f".gitignore no longer protects {ignored}")

    public_files = [
        ROOT / "README.md",
        ROOT / "README.zh-CN.md",
        ROOT / "CHANGELOG.md",
        ROOT / "ROADMAP.md",
        *sorted((ROOT / "docs").glob("*.md")),
        *sorted((ROOT / "docs" / "adr").glob("*.md")),
    ]
    for public_file in public_files:
        text = public_file.read_text(encoding="utf-8")
        leaked_refs = [ref for ref in DISALLOWED_PUBLIC_DOC_REFS if ref in text]
        if leaked_refs:
            fail(
                f"{public_file.relative_to(ROOT)} still references internal docs: "
                + ", ".join(leaked_refs)
            )
        leaked_score_phrases = [
            phrase for phrase in DISALLOWED_SUBJECTIVE_SCORE_PHRASES if phrase in text
        ]
        if leaked_score_phrases:
            fail(
                f"{public_file.relative_to(ROOT)} still uses subjective score phrasing: "
                + ", ".join(leaked_score_phrases)
            )

    scanned_files = [
        ROOT / path
        for path in tracked
        if not path.startswith(("docs/history/", "build/", "dist/", "tests/"))
        and path
        not in {
            "scripts/run_release_surface_check.py",
            "scripts/run_dead_code_audit.py",
            "scripts/package_smoke.py",
        }
        and Path(path).suffix in {".py", ".js", ".html", ".md", ".toml", ".json", ".yml", ".yaml"}
    ]
    for scanned_file in scanned_files:
        text = scanned_file.read_text(encoding="utf-8")
        leaked_terms = [term for term in DISALLOWED_SCOPE_TERMS if term in text]
        if leaked_terms:
            fail(
                f"{scanned_file.relative_to(ROOT)} still references out-of-scope terms: "
                + ", ".join(leaked_terms)
            )


def validation_count(text: str, pattern: str) -> str:
    match = re.search(pattern, text)
    if not match:
        fail(f"could not find validation count for pattern: {pattern}")
    return match.group(1)


def check_status_alignment() -> None:
    status = read("docs/status.md")
    readme = read("README.md")
    readme_zh = read("README.zh-CN.md")

    status_tests = validation_count(status, r"Unit/integration tests: (\d+) passed")
    readme_tests = validation_count(readme, r"(\d+) tests")
    zh_tests = validation_count(readme_zh, r"`(\d+)` 个测试通过")
    if len({status_tests, readme_tests, zh_tests}) != 1:
        fail(
            "README and docs/status.md disagree on test count: "
            f"{readme_tests}, {zh_tests}, {status_tests}"
        )

    require_snippets(
        "README.md",
        [
            "No telemetry, analytics, or document upload.",
            "docker compose -f docker-compose.image.yml up",
            "docflow doctor --offline",
            "not a personal vault",
        ],
    )
    require_snippets(
        "docs/status.md",
        [
            "not a broad public benchmark",
            "BEIR SciFact-lite",
            "BEIR NFCorpus-lite",
            "Archived subset only",
            "Offline doctor: 0 unexpected outbound connections",
            "DocFlow is not published to PyPI yet",
            "release candidate manifest",
            "OpenSSF Scorecard",
        ],
    )


def check_pypi_claims_are_bounded() -> None:
    public_files = [
        ROOT / "README.md",
        ROOT / "README.zh-CN.md",
        ROOT / "CHANGELOG.md",
        ROOT / "ROADMAP.md",
        ROOT / "CONTRIBUTING.md",
        ROOT / "SECURITY.md",
        ROOT / ".github" / "PULL_REQUEST_TEMPLATE.md",
        *sorted((ROOT / ".github" / "ISSUE_TEMPLATE").glob("*.md")),
        *sorted((ROOT / "docs").glob("*.md")),
        *sorted((ROOT / "docs" / "adr").glob("*.md")),
    ]
    disallowed = [
        re.compile(r"\bpip install -U docflow\b"),
        re.compile(r"\bpython -m pip install docflow\b"),
        re.compile(r"\bpython -m pip install -U docflow\b"),
        re.compile(r"\bpip install docflow\b"),
        re.compile(r"\bpip install \"docflow"),
        re.compile(r"\bpip install 'docflow"),
        re.compile(r"\bDocFlow is (?:published|available) on PyPI\b", re.IGNORECASE),
        re.compile(r"\bPyPI package is available\b", re.IGNORECASE),
        re.compile(r"\binstall from PyPI\b", re.IGNORECASE),
        re.compile(r"\binstall DocFlow from PyPI\b", re.IGNORECASE),
    ]
    allowed_context = (
        "not published to PyPI",
        "not on PyPI yet",
        "PyPI publishing is enabled later",
    )
    findings: list[str] = []
    for public_file in public_files:
        text = public_file.read_text(encoding="utf-8")
        for pattern in disallowed:
            for match in pattern.finditer(text):
                line = text.count("\n", 0, match.start()) + 1
                line_text = text.splitlines()[line - 1]
                if any(context in line_text for context in allowed_context):
                    continue
                findings.append(f"{public_file.relative_to(ROOT)}:{line}: {match.group(0)}")
    if findings:
        fail("public docs claim PyPI availability before publishing: " + "; ".join(findings))


def check_docker_and_package_surface() -> None:
    require_snippets(
        "docker-compose.yml",
        [
            "docflow:",
            "qdrant:",
            "./config.docker.yaml:/app/config.yaml:ro",
            '"8000:8000"',
        ],
    )
    require_snippets(
        "docker-compose.image.yml",
        [
            "ghcr.io/lingengyuan/docflow:edge",
            "qdrant:",
            "qdrant/qdrant:latest@sha256:",
            "./config.docker.yaml:/app/config.yaml:ro",
        ],
    )
    require_snippets(
        "docker-compose.yml",
        [
            "qdrant/qdrant:latest@sha256:",
        ],
    )
    require_snippets(
        "Dockerfile",
        [
            "python:3.12-slim@sha256:",
            'CMD ["docflow", "serve"]',
        ],
    )

    pyproject = read("pyproject.toml")
    if '"ROADMAP.md"' not in pyproject:
        fail("pyproject package data is missing ROADMAP.md")
    for doc in PUBLIC_DOCS:
        if f"docs/{doc}" not in pyproject:
            fail(f"pyproject package data is missing docs/{doc}")
    for adr in PUBLIC_ADRS:
        if f"docs/adr/{adr}" not in pyproject:
            fail(f"pyproject package data is missing docs/adr/{adr}")
    if '"eval/external_benchmarks.json"' not in pyproject:
        fail("pyproject package data is missing eval/external_benchmarks.json")
    if '"eval/answer_faithfulness_v1.jsonl"' not in pyproject:
        fail("pyproject package data is missing eval/answer_faithfulness_v1.jsonl")
    for artifact in (
        "eval/results/external/beir-scifact-lite-20e459e.json",
        "eval/results/external/beir-scifact-lite-latest.json",
        "eval/results/external/beir-nfcorpus-lite-08f3965.json",
        "eval/results/external/beir-nfcorpus-lite-latest.json",
        "eval/results/large-library/large-library-20e459e.json",
        "eval/results/large-library/large-library-5b57389.json",
        "eval/results/large-library/large-library-latest.json",
    ):
        if f'"{artifact}"' not in pyproject:
            fail(f"pyproject package data is missing {artifact}")
    for asset in ("chat.png", "library.png", "notes.png", "settings.png"):
        if f"docs/assets/{asset}" not in pyproject:
            fail(f"pyproject package data is missing docs/assets/{asset}")

    release_script = read("scripts/build_release_candidate.py")
    for snippet in (
        "docflow.release_candidate.v1",
        "RELEASE_MANIFEST.json",
        "RELEASE_NOTES.md",
        "SHA256SUMS",
        "\"build\":",
        "\"python\":",
        "\"platform\":",
        "\"github_run_id\":",
        "published\": False",
        "signed\": False",
    ):
        if snippet not in release_script:
            fail(f"release-candidate script is missing {snippet}")


def check_workflow_security_invariants() -> None:
    for path, document in workflow_documents():
        relative = path.relative_to(ROOT)
        text = path.read_text(encoding="utf-8")
        if "permissions:" not in text:
            fail(f"{relative} must declare workflow or job token permissions explicitly")
        if "permissions: read-all" in text:
            fail(f"{relative} uses broad read-all token permissions")
        unpinned = [
            uses for uses in workflow_uses_entries(document) if not ACTION_PIN_PATTERN.search(uses)
        ]
        if unpinned:
            fail(
                f"{relative} has unpinned workflow actions: "
                + ", ".join(sorted(set(unpinned)))
            )

    for path in ("docker-compose.yml", "docker-compose.image.yml"):
        text = read(path)
        if "qdrant/qdrant:latest" in text and "qdrant/qdrant:latest@sha256:" not in text:
            fail(f"{path} must pin the Qdrant service image by digest")

    dockerfile = read("Dockerfile")
    if "python:3.12-slim" in dockerfile and "python:3.12-slim@sha256:" not in dockerfile:
        fail("Dockerfile must pin the Python base image by digest")


def check_workflows() -> None:
    require_files(REQUIRED_WORKFLOWS)
    check_workflow_security_invariants()
    require_snippets(
        ".github/workflows/ci.yml",
        [
            "ruff check",
            "mypy",
            "pytest",
            "scripts/run_performance_smoke.py --json",
            "main.py dev eval faithfulness --json",
            "main.py dev eval large-library --documents 200 --queries 5",
            "--max-retrieval-p95-ms 1500",
            "--max-answer-p95-ms 2000",
            "--min-top-file-accuracy 1.0 --json",
            "scripts/run_external_benchmark_status.py --json",
            "scripts/run_dead_code_audit.py --json",
            "main.py dev eval parsing --json",
            "scripts/run_release_surface_check.py",
            "scripts/run_phase110_readiness_check.py",
            "scripts/package_smoke.py",
            "doctor --offline",
        ],
    )
    require_snippets(
        ".github/workflows/evaluation.yml",
        [
            "workflow_dispatch",
            "schedule:",
            "qdrant/qdrant",
            "allow_model_download: true",
            "main.py dev eval parsing",
            "main.py dev eval public",
            "main.py dev eval large-library --documents 10000 --queries 20",
            "--write-results --json",
            "--no-rerank",
            "eval/results/large-library/*",
        ],
    )
    require_snippets(
        ".github/workflows/docker-image.yml",
        [
            "branches: [main]",
            "type=raw,value=edge",
            "ghcr.io/",
            "github.repository",
            "docker/build-push-action@10e90e3645eae34f1e60eeb005ba3a3d33f178e8",
            "sbom: true",
            "provenance: mode=max",
        ],
    )
    require_snippets(
        ".github/workflows/scorecard.yml",
        [
            "contents: read",
            "security-events: write",
            "id-token: write",
            "actions: read",
            "persist-credentials: false",
            "ossf/scorecard-action@05b42c624433fc40578a4040d5cf5e36ddca8cde",
            "scorecard-results.sarif",
            "publish_results: true",
        ],
    )
    require_snippets(
        ".github/workflows/python-package.yml",
        [
            "python scripts/package_smoke.py",
            "scripts/build_release_candidate.py --dist-dir dist --skip-build --json",
            "docflow-python-package",
        ],
    )
    require_snippets(
        "docs/development.md",
        ["requirements-core.txt", "requirements-local-model.txt", "requirements-vision.txt"],
    )
    require_snippets(
        "docs/threat-model.md",
        [
            "Browser to DocFlow API",
            "DocFlow to the internet",
            "OpenSSF Scorecard",
            "Release artifact tampering",
        ],
    )
    require_snippets(
        "docs/release.md",
        [
            "workflow actions, Docker bases, and Qdrant service images are still pinned",
            "SHA256SUMS",
            "RELEASE_MANIFEST.json",
            "RELEASE_NOTES.md",
            "Docker image SBOM and provenance attestations",
            "Release artifacts are not signed yet",
            "Trusted Publishing",
        ],
    )
    require_snippets(
        "docs/model-licenses.md",
        ["Model weights are separate artifacts", "Qwen/Qwen3-Embedding-0.6B", "glm-ocr"],
    )


def main() -> int:
    require_files(REQUIRED_ROOT_FILES)
    check_public_docs()
    check_pypi_claims_are_bounded()
    check_status_alignment()
    check_docker_and_package_surface()
    check_workflows()
    print("Release surface check passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
