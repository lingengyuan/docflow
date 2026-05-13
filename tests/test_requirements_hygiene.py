from __future__ import annotations

from pathlib import Path


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


def test_dev_requirements_include_test_and_browser_tools():
    dev = Path("requirements-dev.txt").read_text(encoding="utf-8")

    assert "pytest" in dev
    assert "playwright" in dev


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
