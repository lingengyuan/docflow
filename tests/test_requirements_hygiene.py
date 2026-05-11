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


def test_dev_requirements_include_test_and_browser_tools():
    dev = Path("requirements-dev.txt").read_text(encoding="utf-8")

    assert "pytest" in dev
    assert "playwright" in dev


def test_optional_vision_requirements_are_separate():
    vision = Path("requirements-vision.txt").read_text(encoding="utf-8")

    assert "mlx-vlm" in vision
    assert "pillow-heif" in vision
