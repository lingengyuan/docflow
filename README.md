# DocFlow

Local, private document Q&A for PDFs, Markdown notes, Word files, text files, code snippets, and images.

DocFlow watches local folders, parses supported files, indexes them into local search stores, and answers questions from a browser UI. Data stays on the machine.

## English

### Project Description

DocFlow is a local-first knowledge assistant. It is designed for personal document collections and Obsidian-style notes: drop files into watched folders, let DocFlow index them, then ask questions with cited answers through the web interface.

The current implementation uses FastAPI, SQLite, Qdrant, local embedding and reranking models, and an MLX-backed local LLM by default.

### Features

- Multi-format ingest: PDF, Markdown, TXT, DOCX, code-like text files, and optional image formats.
- Obsidian-friendly Markdown parsing: frontmatter cleanup, wikilink cleanup, callout cleanup, block-id cleanup, and tag extraction.
- Structured chunking: heading-aware text chunks, table chunks, and table summary chunks.
- Parent context retrieval: small chunks are used for matching, while larger parent context is returned for answer generation.
- Hybrid retrieval: vector search plus SQLite FTS5 keyword search, adaptive routing, and reranking.
- Streaming answers: citations are sent first, followed by token streaming.
- Local model options: Qwen3 embedding, Qwen3 reranker, MLX LLM, optional Ollama OCR, optional VLM image parsing.
- Folder watching: multiple watched directories, recursive scans, debounce, and startup cleanup for deleted files.
- Ingest queue visibility: queue status includes current stage and chunk progress.
- Query history, favorites, file upload, source listing, file preview, and summary export endpoints.

### Requirements

- Apple Silicon Mac.
- Python 3.11 or newer.
- Docker Desktop for Qdrant.
- Ollama for OCR on scanned PDFs.
- Optional extra packages for DOCX and image ingest: `python-docx`, `mlx-vlm`, `Pillow`, `pillow-heif`.

### Quick Start

Start Qdrant:

```bash
docker run -d --name qdrant -p 6333:6333 qdrant/qdrant
```

Install Python dependencies:

```bash
cd ~/Projects/docflow
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Install optional format support when needed:

```bash
pip install python-docx mlx-vlm Pillow pillow-heif
```

Pull the OCR model when scanned PDF OCR is needed:

```bash
ollama pull glm-ocr
```

Run the app:

```bash
python main.py serve
```

Open the UI:

```bash
open http://localhost:8000
```

Useful commands:

```bash
python main.py scan
python main.py ingest /path/to/file.pdf
python main.py benchmark README.md docs/HANDOFF-v3.md
python main.py eval
```

### Configuration

Configuration lives in `config.yaml`.

Important settings:

- `paths.watch_dirs`: folders DocFlow scans and watches.
- `paths.supported_extensions`: extensions accepted by the ingest pipeline.
- `paths.db_path`: SQLite database path.
- `qdrant.host`, `qdrant.port`, `qdrant.collection`: Qdrant connection and collection.
- `embedding.model`, `embedding.backend`, `embedding.device`: embedding model and runtime.
- `chunking.chunk_size`, `chunking.chunk_overlap`: chunk size and overlap.
- `ingest.parent_context_chars`: maximum parent context size used when grouping adjacent chunks.
- `ingest.contextual_prefix`: optional contextual prefix generation for Markdown and table chunks. It is off by default.
- `ingest.contextual_prefix_mode`: `metadata` for deterministic local prefixes, or `ollama` for Ollama-generated prefixes.
- `llm.backend`: answer-generation backend. The current default is `mlx`.
- `vlm.enabled`: enables or disables image parsing.

Current default watched folders:

```yaml
paths:
  watch_dirs:
    - path: "~/Documents/DocFlow"
      recursive: false
    - path: "~/MyNotes/HughLin"
      recursive: true
      extensions: [".md"]
    - path: "~/Projects/CodeSnippets"
      recursive: true
      extensions: [".md", ".py", ".rs", ".ts", ".css", ".sh"]
```

### API Reference

Main endpoints:

| Endpoint | Method | Purpose |
|---|---:|---|
| `/api/query` | POST | Synchronous Q&A |
| `/api/query/stream` | POST | Streaming Q&A |
| `/api/ingest` | POST | Trigger scan of watched folders |
| `/api/queue` | GET | Ingest queue status |
| `/api/files` | GET | Indexed file list |
| `/api/upload` | POST | Upload a file into the first watched folder |
| `/api/file/{id}/preview` | GET | Preview the original file |
| `/api/file/{id}/chunks` | GET | Debug chunk list for one file |
| `/api/history` | GET, DELETE | Query history |
| `/api/history/search` | GET | Search query history |
| `/api/favorites` | GET | Favorite files |
| `/api/favorites/{id}` | POST | Toggle a favorite |
| `/api/summarize` | POST | Export file summaries as Markdown |
| `/api/debug/retrieve` | POST | Debug retrieval pipeline without answer generation |
| `/api/llm` | GET, POST | View or switch the active LLM |
| `/api/sources` | GET | Watched source folders |
| `/api/health` | GET | Dependency and capability health |

`/api/health` returns `ok`, `degraded`, or `unavailable`. SQLite and Qdrant are critical checks. Ollama and local model cache checks are reported as optional capabilities, so missing OCR or contextual-prefix support does not hide query/ingest availability. The response also lists whether query, ingest, OCR, and contextual prefix are currently enabled and available.

`/api/debug/retrieve` includes the router decision, candidate counts, retrieval stages, reranked results, parent-expanded context, and any degraded retrieval stages. If vector search is unavailable, DocFlow can still return FTS results from SQLite. If reranking fails, it returns the fused candidates instead of dropping all results. If answer generation fails after retrieval succeeds, the query response keeps the retrieved snippets and clearly says the answer model is unavailable.

### Development and Testing

Run the test suite:

```bash
cd ~/Projects/docflow
.venv/bin/python -m pytest
```

Run a dry-run ingest benchmark:

```bash
.venv/bin/python main.py benchmark README.md docs/HANDOFF-v3.md
```

Run the fixed retrieval evaluation set:

```bash
.venv/bin/python main.py eval
```

Check the FTS tables:

```bash
sqlite3 docflow.db "SELECT COUNT(*) FROM chunks_fts;"
sqlite3 docflow.db "SELECT * FROM chunks_fts WHERE chunks_fts MATCH '机器学习' LIMIT 3;"
```

### Project Structure

```text
docflow/
├── config.yaml
├── main.py
├── requirements.txt
├── README.md
├── frontend/
│   └── index.html
├── src/
│   ├── api/
│   │   └── app.py
│   ├── ingest/
│   │   ├── chunker.py
│   │   ├── embedder.py
│   │   ├── parsers/
│   │   ├── pdf_analyzer.py
│   │   ├── pipeline.py
│   │   ├── queue.py
│   │   ├── store.py
│   │   └── watcher.py
│   ├── query/
│   │   ├── engine.py
│   │   ├── generator.py
│   │   └── retriever.py
│   └── embedding_backend.py
└── tests/
```

### Skill Catalog

This repository does not define Codex skills. The project is a standalone local application.

### Contributing

Before changing behavior, read `config.yaml`, the relevant files under `src/`, and the tests that cover the area being changed. Keep README commands aligned with real entry points in `main.py`.

Run the tests before handing off changes:

```bash
.venv/bin/python -m pytest
```

### Maintenance Guide

Keep these constraints in mind when changing the project:

- MLX reranker and MLX LLM work should stay serialized through the shared inference executor.
- Ingest and query must use the same embedding backend configuration.
- The ingest pipeline and retriever share one embedding model instance after startup warmup.
- SQLite FTS row IDs are tied to `chunks.id`; update FTS through the store layer.
- Recursive scans intentionally skip `.obsidian/`, `.trash/`, and `.git/`.

### License

MIT. Add a standalone `LICENSE` file before publishing the project externally.

## 简体中文

### 项目说明

DocFlow 是一个本地优先的文档问答助手，面向个人文档库和 Obsidian 笔记。你把文件放进监控目录，DocFlow 自动解析和索引，然后可以在浏览器里提问，并得到带来源的回答。

当前实现使用 FastAPI、SQLite、Qdrant、本地向量模型、本地精排模型，以及默认的 MLX 本地大模型。

### 功能特性

- 多格式入库：PDF、Markdown、TXT、DOCX、代码类文本文件，以及可选图片格式。
- 适配 Obsidian 笔记：清理 frontmatter、wikilink、callout、block id，并提取标签。
- 结构化分块：支持按标题分块、表格分块、表格摘要分块。
- 父级上下文检索：用小块命中问题，再把更完整的上下文交给回答模型。
- 混合检索：向量检索加 SQLite FTS5 全文检索，自动调整候选数量，再做精排。
- 流式回答：先返回引用来源，再逐步返回答案内容。
- 本地模型：Qwen3 embedding、Qwen3 reranker、MLX LLM，可选 Ollama OCR 和图片理解模型。
- 文件夹监控：支持多个目录、递归扫描、延迟去重，以及启动时清理已删除文件。
- 入库队列可见：可以看到当前阶段和 chunk 处理进度。
- 已有接口覆盖查询历史、收藏、上传、来源列表、文件预览和摘要导出。

### 环境要求

- Apple Silicon Mac。
- Python 3.11 或更高版本。
- Docker Desktop，用于运行 Qdrant。
- Ollama，用于扫描 PDF 的 OCR。
- DOCX 和图片入库需要额外安装：`python-docx`、`mlx-vlm`、`Pillow`、`pillow-heif`。

### 快速开始

启动 Qdrant：

```bash
docker run -d --name qdrant -p 6333:6333 qdrant/qdrant
```

安装 Python 依赖：

```bash
cd ~/Projects/docflow
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

按需安装额外格式支持：

```bash
pip install python-docx mlx-vlm Pillow pillow-heif
```

如果需要扫描 PDF OCR，拉取 OCR 模型：

```bash
ollama pull glm-ocr
```

启动应用：

```bash
python main.py serve
```

打开界面：

```bash
open http://localhost:8000
```

常用命令：

```bash
python main.py scan
python main.py ingest /path/to/file.pdf
python main.py benchmark README.md docs/HANDOFF-v3.md
python main.py eval
```

### 配置

配置集中在 `config.yaml`。

重点配置：

- `paths.watch_dirs`：DocFlow 会扫描和监控的目录。
- `paths.supported_extensions`：入库支持的文件扩展名。
- `paths.db_path`：SQLite 数据库位置。
- `qdrant.host`、`qdrant.port`、`qdrant.collection`：Qdrant 连接和集合。
- `embedding.model`、`embedding.backend`、`embedding.device`：向量模型和运行方式。
- `chunking.chunk_size`、`chunking.chunk_overlap`：分块大小和重叠长度。
- `ingest.parent_context_chars`：相邻 chunk 合并为父级上下文时的最大长度。
- `ingest.contextual_prefix`：是否为 Markdown 和表格 chunk 生成上下文前缀，默认关闭。
- `ingest.contextual_prefix_mode`：`metadata` 表示使用本地固定前缀，`ollama` 表示使用 Ollama 生成前缀。
- `llm.backend`：回答生成方式。当前默认是 `mlx`。
- `vlm.enabled`：是否启用图片解析。

当前默认监控目录：

```yaml
paths:
  watch_dirs:
    - path: "~/Documents/DocFlow"
      recursive: false
    - path: "~/MyNotes/HughLin"
      recursive: true
      extensions: [".md"]
    - path: "~/Projects/CodeSnippets"
      recursive: true
      extensions: [".md", ".py", ".rs", ".ts", ".css", ".sh"]
```

### API 接口

主要接口：

| 接口 | 方法 | 用途 |
|---|---:|---|
| `/api/query` | POST | 普通问答 |
| `/api/query/stream` | POST | 流式问答 |
| `/api/ingest` | POST | 触发监控目录扫描 |
| `/api/queue` | GET | 入库队列状态 |
| `/api/files` | GET | 已入库文件列表 |
| `/api/upload` | POST | 上传文件到第一个监控目录 |
| `/api/file/{id}/preview` | GET | 预览原始文件 |
| `/api/file/{id}/chunks` | GET | 调试查看单个文件的切块列表 |
| `/api/history` | GET, DELETE | 查询历史 |
| `/api/history/search` | GET | 搜索查询历史 |
| `/api/favorites` | GET | 收藏文件 |
| `/api/favorites/{id}` | POST | 切换收藏 |
| `/api/summarize` | POST | 导出文件摘要 |
| `/api/debug/retrieve` | POST | 调试查看检索链路，不生成回答 |
| `/api/llm` | GET, POST | 查看或切换当前模型 |
| `/api/sources` | GET | 查看监控来源目录 |
| `/api/health` | GET | 依赖和能力健康状态 |

`/api/health` 会返回 `ok`、`degraded` 或 `unavailable`。SQLite 和 Qdrant 是关键检查；Ollama 和本地模型缓存作为可选能力展示，所以 OCR 或上下文前缀不可用时，不会掩盖查询和入库是否可用。返回结果也会说明查询、入库、OCR、上下文前缀当前是否启用且可用。

`/api/debug/retrieve` 会返回路由判断、候选数量、各阶段结果、精排结果、父级上下文展开结果，以及是否发生检索降级。向量检索不可用时，DocFlow 仍可从 SQLite 全文检索返回结果；精排失败时，会返回融合后的候选结果，而不是直接丢掉全部结果。如果检索成功但回答模型失败，查询接口会保留已找到的片段，并明确提示回答模型暂时不可用。

### 开发与测试

运行测试：

```bash
cd ~/Projects/docflow
.venv/bin/python -m pytest
```

运行一次不写入索引的入库测试：

```bash
.venv/bin/python main.py benchmark README.md docs/HANDOFF-v3.md
```

运行固定检索评估集：

```bash
.venv/bin/python main.py eval
```

检查全文检索表：

```bash
sqlite3 docflow.db "SELECT COUNT(*) FROM chunks_fts;"
sqlite3 docflow.db "SELECT * FROM chunks_fts WHERE chunks_fts MATCH '机器学习' LIMIT 3;"
```

### 项目结构

```text
docflow/
├── config.yaml
├── main.py
├── requirements.txt
├── README.md
├── frontend/
│   └── index.html
├── src/
│   ├── api/
│   │   └── app.py
│   ├── ingest/
│   │   ├── chunker.py
│   │   ├── embedder.py
│   │   ├── parsers/
│   │   ├── pdf_analyzer.py
│   │   ├── pipeline.py
│   │   ├── queue.py
│   │   ├── store.py
│   │   └── watcher.py
│   ├── query/
│   │   ├── engine.py
│   │   ├── generator.py
│   │   └── retriever.py
│   └── embedding_backend.py
└── tests/
```

### 技能目录

这个仓库本身没有定义 Codex 技能。它是一个独立运行的本地应用。

### 贡献指南

修改行为前，先阅读 `config.yaml`、`src/` 下相关文件，以及覆盖对应功能的测试。README 里的命令要和 `main.py` 里的真实入口保持一致。

交付前运行测试：

```bash
.venv/bin/python -m pytest
```

### 维护指南

修改项目时要注意这些约束：

- MLX 精排和 MLX 回答生成需要通过同一个推理通道串行运行。
- 入库和查询必须使用同一套向量配置。
- 启动预热后，入库管线和查询器会共享同一个向量模型实例。
- SQLite 全文检索表和 `chunks.id` 绑定，更新时要走存储层。
- 递归扫描会主动跳过 `.obsidian/`、`.trash/` 和 `.git/`。

### 许可证

MIT。公开发布前建议补一个独立的 `LICENSE` 文件。
