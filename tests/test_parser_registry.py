from pathlib import Path

from src.ingest.parsers import ParserRegistry
from src.ingest.parsers.image_parser import ImageParser
from src.ingest.parsers.txt_parser import TxtParser


def _base_config(*, vlm_enabled: bool = False) -> dict:
    return {
        "ollama": {
            "base_url": "http://localhost:11434",
            "ocr_model": "glm-ocr",
        },
        "paths": {
            "supported_extensions": [".md", ".py", ".ts", ".css", ".sh", ".png"],
        },
        "vlm": {
            "enabled": vlm_enabled,
            "model": "mlx-community/Qwen3-VL-8B-Instruct-4bit",
        },
    }


def test_configured_code_extensions_use_text_parser():
    registry = ParserRegistry.from_config(_base_config(vlm_enabled=False))

    for name in ("snippet.py", "component.ts", "styles.css", "script.sh"):
        parser = registry.resolve(Path(name))
        assert isinstance(parser, TxtParser)


def test_image_extensions_depend_on_vlm_flag():
    without_vlm = ParserRegistry.from_config(_base_config(vlm_enabled=False))
    with_vlm = ParserRegistry.from_config(_base_config(vlm_enabled=True))

    assert without_vlm.supports(Path("diagram.png")) is False
    assert isinstance(with_vlm.resolve(Path("diagram.png")), ImageParser)
