# DocFlow

Local-first personal knowledge Q&A for your documents, notes, code, and images.

DocFlow 是一个本地优先的个人知识库助手，用来管理、检索和问答你的本地资料。

![DocFlow chat workspace](docs/assets/chat.png)

## English

### Project Description

DocFlow watches folders on your machine, indexes supported files, and answers questions in a browser workspace with visible sources. It is built for private personal knowledge work: no telemetry, local storage, and local model backends by default.

### Features

- Ask questions across PDFs, Markdown, DOCX, TXT, code-like files, and optional image content.
- Review source snippets, save useful answers as notes, and inspect your local library.
- Keep metadata in SQLite, vectors in Qdrant, and model traffic local by default.
- Verify local behavior with tests, browser acceptance checks, retrieval evals, parsing evals, and an offline network check.

### Quick Start

```bash
git clone https://github.com/lingengyuan/docflow.git
cd docflow
docker compose up --build
```

Open [http://localhost:8000](http://localhost:8000), then choose **导入示例资料** to try a small local library.

### Current Verification

Latest local checks in this repository: `300` tests passed, `73` browser checks passed, retrieval eval `84/84` with Recall@5 `1.0`, parsing eval `31/31`, and the offline local-use check reported `0` unexpected outbound connections.

The eval sets are now broad enough for project regression checks. Treat these numbers as the current baseline, not a broad public benchmark.

### Privacy Promise

DocFlow does not include telemetry, analytics, automatic error reporting, or document upload for product analytics. Runtime documents, SQLite metadata, Qdrant vectors, backups, and generated indexes stay on your machine unless you explicitly configure an external feature.

Run the offline check:

```bash
docflow doctor --offline
```

### Skill Catalog

Not applicable. DocFlow is an app, not a Codex skill package.

### Project Structure

- `main.py`: command entry point.
- `src/api/`: browser app and API.
- `src/ingest/`: parsing, chunking, indexing, and storage.
- `src/query/`: retrieval, reranking, citations, and answer generation.
- `frontend/`: browser workspace.
- `docs/`: public documentation.
- `eval/`: committed retrieval and parsing evaluation inputs.

### Configuration

DocFlow creates `config.yaml` from `config.example.yaml` on first run. Configure watched folders, supported extensions, SQLite path, Qdrant connection, embedding model, local model backend, and privacy settings there.

### Development and Testing

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e .
docker compose up -d qdrant
docflow demo --create-only
.venv/bin/python -m pytest -q
docflow eval retrieval --write-results
docflow eval parsing --write-results
docflow browser-acceptance
docflow doctor --offline
```

### Contributing

Read [CONTRIBUTING.md](CONTRIBUTING.md). Keep changes focused, run tests before opening a pull request, and keep the normal browser UI free of command-line or developer-only wording.

### Maintenance Guide

See [Features](docs/features.md), [Architecture](docs/architecture.md), [Privacy](docs/privacy.md), [CLI](docs/cli.md), [Development](docs/development.md), [Evaluation](docs/evaluation.md), [Status](docs/status.md), and [Roadmap](ROADMAP.md).

### License

MIT. See [LICENSE](LICENSE).

## 简体中文

### 项目说明

DocFlow 会监听你机器上的本地文件夹，把资料写入本地索引，然后在浏览器工作台里回答问题并展示来源。它面向私人的个人知识管理：默认无遥测、本地存储、本地模型后端。

### 功能特性

- 支持基于 PDF、Markdown、DOCX、TXT、代码文本和可选图片内容提问。
- 可以查看来源片段、把有用回答保存为笔记，并管理本地资料库。
- 默认使用 SQLite 保存元数据、Qdrant 保存向量、本地模型处理问答。
- 提供测试、浏览器验收、检索评估、解析评估和离线网络检查来验证当前状态。

### 快速开始

```bash
git clone https://github.com/lingengyuan/docflow.git
cd docflow
docker compose up --build
```

打开 [http://localhost:8000](http://localhost:8000)，然后选择 **导入示例资料** 体验一个小型本地资料库。

### 当前验证结果

当前仓库最近一次本地验证结果：`300` 个测试通过，`73` 个浏览器检查通过，检索评估 `84/84` 且 Recall@5 `1.0`，解析评估 `31/31`，离线本地使用检查报告 `0` 个意外外连。

评估集已经足够用于项目回归检查。这些数字是当前基线，不应当当作大规模公开 benchmark。

### 隐私承诺

DocFlow 不包含遥测、分析统计、自动错误上报，也不会为了产品分析上传你的文档。运行时文档、SQLite 元数据、Qdrant 向量、备份和生成索引都保留在你的机器上，除非你明确配置外部功能。

运行离线检查：

```bash
docflow doctor --offline
```

### 技能目录

不适用。DocFlow 是应用，不是 Codex 技能包。

### 项目结构

- `main.py`：命令入口。
- `src/api/`：浏览器应用和接口。
- `src/ingest/`：解析、切块、索引和存储。
- `src/query/`：检索、重排、引用和回答生成。
- `frontend/`：浏览器工作台。
- `docs/`：公开文档。
- `eval/`：已提交的检索和解析评估输入。

### 配置

DocFlow 首次运行会从 `config.example.yaml` 创建 `config.yaml`。监听目录、支持文件类型、SQLite 路径、Qdrant 连接、Embedding 模型、本地模型后端和隐私设置都在这里配置。

### 开发与测试

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e .
docker compose up -d qdrant
docflow demo --create-only
.venv/bin/python -m pytest -q
docflow eval retrieval --write-results
docflow eval parsing --write-results
docflow browser-acceptance
docflow doctor --offline
```

### 贡献指南

请阅读 [CONTRIBUTING.md](CONTRIBUTING.md)。保持改动聚焦，提交前运行测试，普通浏览器界面不要出现命令行或开发者专用文案。

### 维护指南

见 [功能说明](docs/features.md)、[架构说明](docs/architecture.md)、[隐私说明](docs/privacy.md)、[命令说明](docs/cli.md)、[开发说明](docs/development.md)、[评估说明](docs/evaluation.md)、[状态说明](docs/status.md) 和 [路线图](ROADMAP.md)。

### 许可证

MIT。见 [LICENSE](LICENSE)。
