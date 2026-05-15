#!/usr/bin/env python3

from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

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
]

PUBLIC_DOCS = [
    "architecture.md",
    "cli.md",
    "development.md",
    "evaluation.md",
    "features.md",
    "privacy.md",
    "release.md",
    "status.md",
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
    for ignored in ("docs/history/", "output/", "config.yaml", "qdrant_storage/"):
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

    scanned_files = [
        ROOT / path
        for path in tracked
        if not path.startswith(("docs/history/", "build/", "dist/", "tests/"))
        and path not in {"scripts/run_release_surface_check.py", "scripts/package_smoke.py"}
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
            "docker compose up --build",
            "docflow doctor --offline",
            "not a personal vault",
        ],
    )
    require_snippets(
        "docs/status.md",
        [
            "not a broad public benchmark",
            "No external benchmark score has been archived yet",
            "Offline doctor: 0 unexpected outbound connections",
            "DocFlow is not published to PyPI yet",
        ],
    )


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
            "ghcr.io/lingengyuan/docflow",
            "qdrant:",
            "./config.docker.yaml:/app/config.yaml:ro",
        ],
    )
    require_snippets("Dockerfile", ['CMD ["docflow", "serve"]'])

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
    for asset in ("chat.png", "library.png", "notes.png", "settings.png"):
        if f"docs/assets/{asset}" not in pyproject:
            fail(f"pyproject package data is missing docs/assets/{asset}")


def check_workflows() -> None:
    require_files(REQUIRED_WORKFLOWS)
    require_snippets(
        ".github/workflows/ci.yml",
        [
            "ruff check",
            "mypy",
            "pytest",
            "scripts/run_performance_smoke.py --json",
            "scripts/run_external_benchmark_status.py --json",
            "main.py eval parsing --json",
            "scripts/run_release_surface_check.py",
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
            "main.py eval parsing",
            "main.py eval public",
            "--no-rerank",
        ],
    )
    require_snippets(
        ".github/workflows/docker-image.yml",
        ["ghcr.io/", "github.repository", "docker/build-push-action"],
    )
    require_snippets(
        ".github/workflows/python-package.yml",
        ["python scripts/package_smoke.py", "docflow-python-package"],
    )


def main() -> int:
    require_files(REQUIRED_ROOT_FILES)
    check_public_docs()
    check_status_alignment()
    check_docker_and_package_surface()
    check_workflows()
    print("Release surface check passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
