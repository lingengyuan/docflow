# DocFlow

DocFlow is a local-first personal knowledge assistant for PDFs, Markdown notes, DOCX files, text, code snippets, and images.

DocFlow 是一个本地优先的个人知识库助手，用来管理、检索和问答你的本地资料。

![DocFlow chat workspace](docs/assets/chat.png)

## English

### Project Description

DocFlow watches local folders, indexes supported files, and answers questions from your own library in a browser workspace.

### Features

- Local document Q&A with visible sources.
- Library, notes, settings, and source preview workspaces.
- PDF, Markdown, DOCX, TXT, code-like files, and optional image ingest.
- SQLite metadata, Qdrant vectors, and local model backends by default.

### Quick Start

```bash
git clone https://github.com/lingengyuan/docflow.git
cd docflow
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
docker run -d --name qdrant -p 6333:6333 qdrant/qdrant
python main.py serve
```

Open [http://localhost:8000](http://localhost:8000).

### Skill Catalog

Not applicable. DocFlow is an app, not a Codex skill package.

### Project Structure

- `main.py`: command entry point.
- `src/api/`: web app and API.
- `src/ingest/`: parsing, chunking, and indexing.
- `src/query/`: retrieval and answer generation.
- `frontend/`: browser workspace.
- `docs/`: public documentation.

### Configuration

DocFlow reads `config.yaml`. Configure watched folders, supported extensions, SQLite path, Qdrant connection, embedding model, and local LLM backend there.

### Development and Testing

```bash
.venv/bin/python -m pytest
python main.py browser-acceptance
```

### Contributing

Use focused changes, run tests before handoff, and keep user-facing UI free of command-line or developer-only wording.

### Maintenance Guide

See [CLI](docs/cli.md), [Development](docs/development.md), and [Evaluation](docs/evaluation.md).

### License

MIT.

## 简体中文

### 项目说明

DocFlow 会监听本地文件夹，把资料写入本地索引，然后通过浏览器页面进行提问、查来源、整理笔记和管理资料库。

### 功能特性

- 基于本地资料问答，并展示来源。
- 提供资料库、笔记、设置和来源预览工作区。
- 支持 PDF、Markdown、DOCX、TXT、代码文本和可选图片。
- 默认使用 SQLite、Qdrant 和本地模型后端。

### 快速开始

```bash
git clone https://github.com/lingengyuan/docflow.git
cd docflow
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
docker run -d --name qdrant -p 6333:6333 qdrant/qdrant
python main.py serve
```

打开 [http://localhost:8000](http://localhost:8000)。

### 技能目录

不适用。DocFlow 是应用，不是 Codex 技能包。

### 项目结构

- `main.py`：命令入口。
- `src/api/`：网页应用和接口。
- `src/ingest/`：解析、切块和索引。
- `src/query/`：检索和回答生成。
- `frontend/`：浏览器工作台。
- `docs/`：公开文档。

### 配置

DocFlow 读取 `config.yaml`。监听目录、支持文件类型、SQLite 路径、Qdrant 连接、Embedding 模型和本地 LLM 后端都在这里配置。

### 开发与测试

```bash
.venv/bin/python -m pytest
python main.py browser-acceptance
```

### 贡献指南

保持改动聚焦，交付前运行测试，普通用户界面不要出现命令行或开发者专用文案。

### 维护指南

见 [命令说明](docs/cli.md)、[开发说明](docs/development.md) 和 [评估说明](docs/evaluation.md)。

### 许可证

MIT。
