# Features

DocFlow is designed as a local personal knowledge workspace.

## Core Workflow

- Scan watched folders.
- Parse supported files.
- Store file metadata and chunks in SQLite.
- Store vector data in Qdrant.
- Ask questions in the browser.
- Review citations and source previews.
- Save useful answers as Markdown notes.

## Supported Inputs

- PDF
- Markdown
- TXT
- DOCX
- Python, Rust, TypeScript, CSS, and shell-like text files
- Images when the optional image model is enabled

## Workspaces

- Chat: ask questions and inspect cited answers.
- Library: browse files, collections, tags, favorites, and source chunks.
- Notes: create quick notes, import webpages, and save generated knowledge outputs.
- Settings: review local model, dependency, watched folder, and storage status.
- Source Preview: inspect source material behind citations.

## Local Model Options

DocFlow currently supports local embedding, reranking, MLX-backed local LLMs, Ollama-based OCR, and optional image understanding.

The default product direction is local-first. Cloud model paths, when configured by the user, must remain explicit and visible.
