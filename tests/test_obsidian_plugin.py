import json
from pathlib import Path

PLUGIN_DIR = Path("obsidian-plugin/docflow-assistant")


def test_obsidian_plugin_manifest_is_installable():
    manifest = json.loads((PLUGIN_DIR / "manifest.json").read_text(encoding="utf-8"))

    assert manifest["id"] == "docflow-assistant"
    assert manifest["main"] == "main.js"
    assert manifest["isDesktopOnly"] is True
    assert (PLUGIN_DIR / "main.js").exists()
    assert (PLUGIN_DIR / "styles.css").exists()


def test_obsidian_plugin_exposes_phase38_workflows_without_developer_commands():
    source = (PLUGIN_DIR / "main.js").read_text(encoding="utf-8")

    assert "ask-docflow-selection" in source
    assert "find-docflow-related-notes" in source
    assert "insert-docflow-citations" in source
    assert "/api/query" in source
    assert "/api/obsidian/related" in source
    assert "MarkdownView" in source
    assert "replaceSelection" in source
    assert "doctor" not in source
    assert "repair" not in source
