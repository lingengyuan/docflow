from __future__ import annotations

import ast
from pathlib import Path


def _python_files_under_src() -> list[Path]:
    return [
        path
        for path in Path("src").rglob("*.py")
        if "__pycache__" not in path.parts and path.is_file()
    ]


def test_broad_exception_handlers_are_not_silent():
    offenders: list[str] = []
    for path in _python_files_under_src():
        lines = path.read_text(encoding="utf-8").splitlines()
        for index, line in enumerate(lines):
            if "except Exception:" not in line:
                continue
            window = "\n".join(lines[index : index + 8])
            if "logger." in window or "raise" in window:
                continue
            offenders.append(f"{path}:{index + 1}")

    assert offenders == []


def test_src_print_calls_are_limited_to_cli_maintenance_modules():
    offenders: list[str] = []
    for path in _python_files_under_src():
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            if not isinstance(node.func, ast.Name) or node.func.id != "print":
                continue
            if path.parts[:2] == ("src", "maintenance"):
                continue
            offenders.append(f"{path}:{node.lineno}")

    assert offenders == []
