from __future__ import annotations

from pathlib import Path


def _pinned_version(requirements_text: str, package: str) -> tuple[int, ...] | None:
    prefix = f"{package}=="
    for raw_line in requirements_text.splitlines():
        line = raw_line.strip().split(";", 1)[0].strip()
        if line.lower().startswith(prefix.lower()):
            return tuple(int(part) for part in line[len(prefix) :].split("."))
    return None


def test_runtime_requirements_exclude_experiment_only_packages():
    runtime = Path("requirements.txt").read_text(encoding="utf-8")
    blocked = {
        "accelerate",
        "cbor",
        "datasets",
        "ir_datasets",
        "pandas",
        "peft",
        "pyarrow",
        "trec-car-tools",
        "unlzw3",
        "warc3-wet",
        "warc3-wet-clueweb09",
        "zlib-state",
        "mlx-vlm",
        "pillow-heif",
    }

    for package in blocked:
        assert package not in runtime


def test_runtime_requirements_include_critical_packages():
    """Guard against regressions where runtime imports in ``src/`` rely on
    packages that were never declared in ``requirements.txt``.

    The list below is intentionally narrow: it tracks dependencies that have
    historically been forgotten (e.g. ``jieba`` for the retrieval tokenizer and
    ``python-multipart`` for FastAPI upload handling) and that the runtime cannot
    function without on a clean install.
    """

    runtime = Path("requirements.txt").read_text(encoding="utf-8")

    for package in ("jieba", "python-multipart"):
        assert package in runtime, f"{package} must be declared in requirements.txt"


def test_known_vulnerable_dependency_pins_do_not_regress():
    runtime = Path("requirements.txt").read_text(encoding="utf-8")
    dev = Path("requirements-dev.txt").read_text(encoding="utf-8")
    vision = Path("requirements-vision.txt").read_text(encoding="utf-8")

    floors = [
        (runtime, "python-multipart", (0, 0, 27)),
        (runtime, "pillow", (12, 2, 0)),
        (runtime, "onnx", (1, 21, 0)),
        (dev, "pytest", (9, 0, 3)),
        (vision, "pillow", (12, 2, 0)),
    ]
    for text, package, minimum in floors:
        version = _pinned_version(text, package)
        assert version is not None, f"{package} must be pinned"
        assert version >= minimum, f"{package} must be >= {'.'.join(map(str, minimum))}"


def test_dev_requirements_include_test_and_browser_tools():
    dev = Path("requirements-dev.txt").read_text(encoding="utf-8")

    assert "pytest" in dev
    assert "playwright" in dev
    assert "pip-audit" in dev


def test_optional_vision_requirements_are_separate():
    vision = Path("requirements-vision.txt").read_text(encoding="utf-8")
    mac_mlx = Path("requirements-mac-mlx.txt").read_text(encoding="utf-8")

    assert "mlx==" not in Path("requirements.txt").read_text(encoding="utf-8")
    assert "mlx-lm" not in Path("requirements.txt").read_text(encoding="utf-8")
    assert "mlx==" in mac_mlx
    assert "mlx-lm" in mac_mlx
    assert "requirements-mac-mlx.txt" in vision
    assert "mlx-vlm" in vision
    assert "pillow-heif" in vision
    assert "mlx==0.31.2" in mac_mlx
    assert "mlx-lm==0.31.3" in mac_mlx


def test_dependency_audit_workflow_and_frontend_scripts_exist():
    workflow = Path(".github/workflows/dependency-audit.yml").read_text(encoding="utf-8")
    package = Path("package.json").read_text(encoding="utf-8")

    assert "pip_audit" in workflow
    assert "npm run audit:frontend" in workflow
    assert "audit:frontend" in package
    assert "check:frontend" in package
