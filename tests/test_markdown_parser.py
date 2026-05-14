from src.ingest.parsers.markdown_parser import MarkdownParser


def test_obsidian_cleanup_preserves_markdown_headings(tmp_path):
    note = tmp_path / "note.md"
    note.write_text(
        """---
tags:
  - project/docflow
review: weekly
---
# Top

[[Linked Page|Visible Link]]

> [!note] Callout title
> ## Nested Heading
> Body with #inline-tag ^block-id
""",
        encoding="utf-8",
    )

    doc = MarkdownParser().parse(note)
    text = doc.pages[0].text

    assert "# Top" in text
    assert "## Nested Heading" in text
    assert "Visible Link" in text
    assert "[!note]" not in text
    assert "^block-id" not in text
    assert "project/docflow" in doc.metadata["tags"]
    assert "inline-tag" in doc.metadata["tags"]
    assert doc.metadata["review"] == "weekly"
