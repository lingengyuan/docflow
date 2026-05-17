#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]

PUBLIC_CLAIM_PATHS = [
    ROOT / "README.md",
    ROOT / "README.zh-CN.md",
    ROOT / "ROADMAP.md",
    *sorted((ROOT / "docs").glob("*.md")),
    *sorted((ROOT / "docs" / "adr").glob("*.md")),
]

DISALLOWED_SCORE_CLAIMS = [
    (re.compile(r"\b9[05]\+\b", re.IGNORECASE), "optimistic public score target"),
    (re.compile(r"\b9[05]\s*/\s*100\b", re.IGNORECASE), "subjective public score"),
    (re.compile(r"\b9[05][ -]?point\b", re.IGNORECASE), "score-framed quality push"),
    (re.compile(r"\b9-point maturity\b", re.IGNORECASE), "legacy subjective baseline"),
    (re.compile(r"\bmaturity scorecard\b", re.IGNORECASE), "legacy subjective baseline"),
    (re.compile(r"(?:达到|超过|冲到|冲刺|目标)\s*9[05]\s*(?:分|\+)?"), "主观公开评分目标"),
]

REQUIRED_STATUS_SNIPPETS = [
    "OpenSSF Scorecard: latest reviewed pre-Phase115 baseline was 4.5/10",
    "not a mature open-source security score",
    (
        "Phase115 addressed workflow token permissions, GitHub Action pins, "
        "Docker base/Qdrant service image pins"
    ),
    "branch protection, enforced code review, CI-on-PR history",
    "signed releases, PyPI publishing, fuzzing",
    "hash-pinned Python installation commands",
    "not a broad public benchmark",
    "A full BEIR, MTEB, C-MTEB, or domain-specific benchmark is still needed",
    "DocFlow is not published to PyPI yet",
]


class ReadinessError(RuntimeError):
    pass


def _line_number(text: str, index: int) -> int:
    return text.count("\n", 0, index) + 1


def _run(command: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(command, cwd=ROOT, check=True, capture_output=True, text=True)


def check_public_score_claims() -> list[str]:
    findings: list[str] = []
    for path in PUBLIC_CLAIM_PATHS:
        text = path.read_text(encoding="utf-8")
        for pattern, description in DISALLOWED_SCORE_CLAIMS:
            for match in pattern.finditer(text):
                line = _line_number(text, match.start())
                relative = path.relative_to(ROOT)
                findings.append(
                    f"{relative}:{line} contains {description}: {match.group(0)!r}"
                )
    if findings:
        raise ReadinessError("public score claims found:\n" + "\n".join(findings))
    return ["public score claims: none"]


def check_status_scorecard_alignment() -> list[str]:
    status = (ROOT / "docs" / "status.md").read_text(encoding="utf-8")
    missing = [snippet for snippet in REQUIRED_STATUS_SNIPPETS if snippet not in status]
    if missing:
        raise ReadinessError(
            "docs/status.md does not record the current readiness gaps: "
            + ", ".join(missing)
        )
    return ["status scorecard and benchmark caveats: recorded"]


def check_release_surface() -> list[str]:
    _run([sys.executable, "scripts/run_release_surface_check.py"])
    return ["release surface: passed"]


def check_dead_code_audit() -> list[str]:
    result = _run([sys.executable, "scripts/run_dead_code_audit.py", "--json"])
    payload = json.loads(result.stdout)
    if payload.get("status") != "ok" or payload.get("issues"):
        raise ReadinessError("dead-code audit is not clean")
    return ["dead-code audit: passed"]


def run_checks() -> list[str]:
    checks: list[str] = []
    checks.extend(check_public_score_claims())
    checks.extend(check_status_scorecard_alignment())
    checks.extend(check_release_surface())
    checks.extend(check_dead_code_audit())
    return checks


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run the Phase110 release-readiness guard.")
    parser.add_argument("--json", action="store_true", help="Print machine-readable output.")
    args = parser.parse_args(argv)

    try:
        checks = run_checks()
    except (OSError, subprocess.CalledProcessError, ReadinessError, json.JSONDecodeError) as exc:
        if args.json:
            payload: dict[str, Any] = {"status": "failed", "error": str(exc)}
            print(json.dumps(payload, indent=2, ensure_ascii=False))
        else:
            print(f"Phase110 readiness check failed: {exc}", file=sys.stderr)
        return 1

    if args.json:
        print(json.dumps({"status": "ok", "checks": checks}, indent=2, ensure_ascii=False))
    else:
        print("Phase110 readiness check passed")
        for check in checks:
            print(f"- {check}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
