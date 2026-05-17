from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def run_docflow(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(ROOT / "main.py"), *args],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )


def tracked_files() -> set[str]:
    result = subprocess.run(
        ["git", "ls-files"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    return {line.strip() for line in result.stdout.splitlines() if line.strip()}


def test_phase118_legacy_score_files_are_no_longer_tracked():
    tracked = tracked_files()

    assert "eval/internal_quality_baseline_dimensions.json" in tracked
    assert "eval/phase11_maturity_dimensions.json" not in tracked
    assert "eval/phase11_questions.jsonl" not in tracked


def test_phase118_dev_help_does_not_promote_legacy_score_command():
    result = run_docflow("dev", "--help")

    assert result.returncode == 0
    assert "maturity-eval" not in result.stdout
    assert "scorecard" not in result.stdout
    assert "docflow dev eval" in result.stdout


def test_phase118_internal_baseline_warns_before_output():
    result = run_docflow("dev", "maturity-eval", "--skip-retrieval", "--skip-parsing")

    assert result.returncode == 0
    assert "Internal-only planning baseline" in result.stderr
    assert "not release readiness" in result.stderr
    assert "DocFlow internal planning baseline" in result.stdout
    assert "not a public quality claim" in result.stdout


def test_phase118_public_docs_do_not_use_legacy_score_phrasing():
    public_paths = [
        "README.md",
        "README.zh-CN.md",
        "CHANGELOG.md",
        "ROADMAP.md",
        *[str(path.relative_to(ROOT)) for path in sorted((ROOT / "docs").glob("*.md"))],
        *[str(path.relative_to(ROOT)) for path in sorted((ROOT / "docs" / "adr").glob("*.md"))],
    ]
    public_text = "\n".join((ROOT / path).read_text(encoding="utf-8") for path in public_paths)

    for forbidden in [
        "rolling 9-point maturity scorecard",
        "9-point maturity scorecard",
        "9-point maturity",
        "maturity scorecard",
    ]:
        assert forbidden not in public_text
