#!/usr/bin/env python3
"""Audit stale command surface and known deferred feature remnants."""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import main as docflow_main  # noqa: E402

REMOVED_PATHS = [
    "obsidian-plugin/",
    "frontend/js/pwa.js",
    "frontend/sw.js",
    "frontend/manifest.webmanifest",
    "src/api/routes/obsidian.py",
    "src/api/handlers/obsidian_handlers.py",
]

COMMAND_SURFACE_FILES = [
    "AGENTS.md",
    "main.py",
    "README.md",
    "README.zh-CN.md",
    "CONTRIBUTING.md",
    ".github/PULL_REQUEST_TEMPLATE.md",
    "docs/cli.md",
    "docs/evaluation.md",
    "docs/release.md",
    "eval/qa_v1.jsonl",
    "scripts/install_local.sh",
    "scripts/service.sh",
    "src/maintenance/backup.py",
]

LEGACY_COMMAND_PATTERNS = [
    r"\bdocflow\s+eval\b",
    r"\bdocflow\s+platform\b",
    r"\bdocflow\s+browser-acceptance\b",
    r"\bdocflow\s+restore-drill\b",
    r"\bdocflow\s+repair-ids\b",
    r"\bdocflow\s+rebuild\b",
    r"\bdocflow\s+check\b",
    r"\bdocflow\s+backup\b",
    r"\bdocflow\s+sample-suite\b",
    r"\bdocflow\s+maturity-eval\b",
    r"\bdocflow\s+service\b",
    r"\bdocflow\s+install-local\b",
    r"\bdocflow\s+export-chunks\b",
    r"\bmain\.py\s+eval\b",
    r"\bmain\.py\s+platform\b",
    r"\bmain\.py\s+browser-acceptance\b",
    r"\bmain\.py\s+restore-drill\b",
    r"\bmain\.py\s+repair-ids\b",
    r"\bmain\.py\s+rebuild\b",
    r"\bmain\.py\s+check\b",
    r"\bmain\.py\s+backup\b",
    r"\bmain\.py\s+sample-suite\b",
    r"\bmain\.py\s+maturity-eval\b",
    r"\bmain\.py\s+service\b",
    r"\bmain\.py\s+install-local\b",
    r"\bmain\.py\s+export-chunks\b",
    r"\bpython\s+main\.py\s+eval\b",
    r"\bpython\s+main\.py\s+platform\b",
    r"\bpython\s+main\.py\s+browser-acceptance\b",
    r"\bpython\s+main\.py\s+restore-drill\b",
    r"\bpython\s+main\.py\s+repair-ids\b",
    r"\bpython\s+main\.py\s+rebuild\b",
]

PUBLIC_HELP_FORBIDDEN_TERMS = [
    " eval ",
    "platform",
    "browser-acceptance",
    "restore-drill",
    "repair-ids",
    "sample-suite",
    "maturity-eval",
    "export-chunks",
    "install-local",
]

MAINTENANCE_MODULES = {
    "__init__.py": "package marker",
    "backup.py": "backup, restore-plan, restore-drill",
    "consistency.py": "index consistency, rebuild, repair ids",
    "demo.py": "demo library creation",
    "launchd.py": "optional macOS local service",
    "local_install.py": "source checkout install plan",
    "platform.py": "runtime capability status",
    "startup.py": "startup, doctor, offline doctor",
}


def git_ls_files() -> list[str]:
    result = subprocess.run(
        ["git", "-C", str(PROJECT_ROOT), "ls-files"],
        check=True,
        capture_output=True,
        text=True,
    )
    return [line.strip() for line in result.stdout.splitlines() if line.strip()]


def command_names_from_main() -> list[str]:
    text = (PROJECT_ROOT / "main.py").read_text(encoding="utf-8")
    return sorted(set(re.findall(r'cmd == "([^"]+)"', text)))


def run_help(*args: str) -> str:
    result = subprocess.run(
        [sys.executable, str(PROJECT_ROOT / "main.py"), *args],
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout


def check_removed_paths(tracked: list[str]) -> list[str]:
    return [
        path
        for path in tracked
        if any(path == removed or path.startswith(removed) for removed in REMOVED_PATHS)
    ]


def check_public_readmes() -> dict[str, list[str]]:
    findings: dict[str, list[str]] = {}
    for path in COMMAND_SURFACE_FILES:
        text = (PROJECT_ROOT / path).read_text(encoding="utf-8")
        matches = [pattern for pattern in LEGACY_COMMAND_PATTERNS if re.search(pattern, text)]
        if matches:
            findings[path] = matches
    return findings


def check_public_help() -> list[str]:
    help_text = run_help("--help")
    return [term for term in PUBLIC_HELP_FORBIDDEN_TERMS if term in help_text]


def check_group_help() -> dict[str, bool]:
    admin_help = run_help("admin", "--help")
    dev_help = run_help("dev", "--help")
    return {
        "admin_has_platform": "platform" in admin_help,
        "admin_has_restore_drill": "restore-drill" in admin_help,
        "admin_has_repair_ids": "repair-ids" in admin_help,
        "dev_has_eval": "eval" in dev_help,
        "dev_has_browser_acceptance": "browser-acceptance" in dev_help,
        "dev_has_dead_code_audit": "dead-code-audit" in dev_help,
    }


def check_maintenance_modules() -> list[str]:
    files = sorted(path.name for path in (PROJECT_ROOT / "src" / "maintenance").glob("*.py"))
    return [path for path in files if path not in MAINTENANCE_MODULES]


def build_report() -> dict[str, Any]:
    tracked = git_ls_files()
    commands = command_names_from_main()
    allowed_top_level = (
        set(docflow_main.USER_COMMANDS)
        | {"admin", "dev"}
    )
    unknown_commands = [command for command in commands if command not in allowed_top_level]
    removed_paths = check_removed_paths(tracked)
    readme_findings = check_public_readmes()
    help_leaks = check_public_help()
    group_help = check_group_help()
    unknown_maintenance = check_maintenance_modules()

    issues: list[str] = []
    if unknown_commands:
        issues.append(f"unknown top-level CLI commands: {', '.join(unknown_commands)}")
    if removed_paths:
        issues.append(f"removed or deferred paths are still tracked: {', '.join(removed_paths)}")
    if readme_findings:
        issues.append(f"public command surface exposes retired commands: {readme_findings}")
    if help_leaks:
        issues.append(f"public help exposes internal commands: {', '.join(help_leaks)}")
    if not all(group_help.values()):
        issues.append(f"group help is incomplete: {group_help}")
    if unknown_maintenance:
        modules = ", ".join(unknown_maintenance)
        issues.append(f"maintenance modules need an allowlist reason: {modules}")

    return {
        "schema": "docflow.dead_code_audit.v1",
        "status": "ok" if not issues else "failed",
        "top_level_public_commands": sorted(docflow_main.USER_COMMANDS),
        "admin_commands": sorted(docflow_main.ADMIN_COMMANDS),
        "dev_commands": sorted(docflow_main.DEV_COMMANDS),
        "retired_top_level_commands": docflow_main.RETIRED_TOP_LEVEL_COMMANDS,
        "retired_command_replacements": docflow_main.RETIRED_TOP_LEVEL_COMMANDS,
        "tracked_removed_paths": removed_paths,
        "public_help_leaks": help_leaks,
        "command_surface_findings": readme_findings,
        "command_surface_files": COMMAND_SURFACE_FILES,
        "maintenance_modules": MAINTENANCE_MODULES,
        "unknown_maintenance_modules": unknown_maintenance,
        "issues": issues,
    }


def format_report(report: dict[str, Any]) -> str:
    lines = [f"DocFlow dead-code audit: {report['status']}"]
    lines.append(f"Public commands: {', '.join(report['top_level_public_commands'])}")
    lines.append(f"Admin commands: {', '.join(report['admin_commands'])}")
    lines.append(f"Dev commands: {', '.join(report['dev_commands'])}")
    lines.append(f"Retired top-level commands: {', '.join(report['retired_top_level_commands'])}")
    if report["issues"]:
        lines.append("Issues:")
        lines.extend(f"- {issue}" for issue in report["issues"])
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description="Audit DocFlow stale command and release surface.")
    parser.add_argument("--json", action="store_true", help="Emit JSON")
    args = parser.parse_args()

    report = build_report()
    if args.json:
        print(json.dumps(report, ensure_ascii=False, indent=2))
    else:
        print(format_report(report))
    return 0 if report["status"] == "ok" else 1


if __name__ == "__main__":
    raise SystemExit(main())
