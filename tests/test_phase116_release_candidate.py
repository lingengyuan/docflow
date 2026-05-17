from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "build_release_candidate.py"


def test_release_candidate_script_generates_manifest_checksums_and_notes(tmp_path):
    version = "1.2.3"
    dist = tmp_path / "dist"
    dist.mkdir()
    wheel = dist / f"docflow-{version}-py3-none-any.whl"
    source = dist / f"docflow-{version}.tar.gz"
    wheel.write_bytes(b"wheel")
    source.write_bytes(b"source")

    result = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--dist-dir",
            str(dist),
            "--version",
            version,
            "--commit",
            "abc123def456",
            "--skip-build",
            "--json",
        ],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )

    manifest = json.loads(result.stdout)
    assert manifest["schema"] == "docflow.release_candidate.v1"
    assert manifest["version"] == version
    assert manifest["commit"] == "abc123def456"
    assert manifest["build"]["command"].startswith(
        "python scripts/build_release_candidate.py --dist-dir "
    )
    assert "--skip-build" in manifest["build"]["command"]
    assert manifest["build"]["python"]
    assert manifest["build"]["platform"]
    assert manifest["pypi"]["published"] is False
    assert manifest["signing"]["signed"] is False
    assert manifest["checksum_file"] == "SHA256SUMS"
    assert manifest["release_notes_template"] == "RELEASE_NOTES.md"
    assert {item["file"] for item in manifest["artifacts"]} == {wheel.name, source.name}
    assert f"ghcr.io/lingengyuan/docflow:{version}" in manifest["container_images"][
        "release_tags"
    ]
    assert wheel.name in (dist / "SHA256SUMS").read_text(encoding="utf-8")
    assert source.name in (dist / "SHA256SUMS").read_text(encoding="utf-8")
    notes = (dist / "RELEASE_NOTES.md").read_text(encoding="utf-8")
    assert "DocFlow is not published to PyPI yet" in notes
    assert "Release artifacts are not signed yet" in notes
    assert (dist / "RELEASE_MANIFEST.json").is_file()


def test_release_candidate_policy_is_wired_into_release_surface():
    release = Path("docs/release.md").read_text(encoding="utf-8")
    development = Path("docs/development.md").read_text(encoding="utf-8")
    workflow = Path(".github/workflows/python-package.yml").read_text(encoding="utf-8")
    release_check = Path("scripts/run_release_surface_check.py").read_text(encoding="utf-8")

    assert "scripts/build_release_candidate.py --dist-dir dist --skip-build --json" in workflow
    assert "scripts/build_release_candidate.py --dist-dir dist --clean-dist --json" in release
    assert "RELEASE_MANIFEST.json" in release
    assert "RELEASE_NOTES.md" in release
    assert "PyPI publishing is intentionally disabled" in release
    assert "Trusted Publishing is configured" in release
    assert "Python version, platform" in release
    assert "scripts/build_release_candidate.py" in development
    assert "check_pypi_claims_are_bounded" in release_check
