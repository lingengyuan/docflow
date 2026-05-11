# DocFlow Assistant Obsidian Plugin

This is the Phase38 plugin skeleton for local DocFlow integration.

## Features

- Open a DocFlow side panel in Obsidian.
- Ask DocFlow directly from the side panel.
- Ask DocFlow with selected text.
- Find related notes for the current note through `/api/obsidian/related`.
- Insert citations from the latest answer into the active note.

## Local Install

Copy this folder to:

```text
<vault>/.obsidian/plugins/docflow-assistant/
```

Then enable `DocFlow Assistant` in Obsidian community plugins.

The default DocFlow URL is `http://127.0.0.1:8000` and can be changed in the plugin settings.
