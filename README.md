# DocFlow

> 本地全私有的多格式知识助手 — A fully local, private multi-format RAG assistant

将 PDF、Markdown、Word、TXT、图片放入监控目录，DocFlow 自动解析、分块、向量化，通过 Web 界面流式问答，所有数据不离开本机。支持 Obsidian vault 集成，自动清洗 frontmatter / wikilink / callout 等语法。

Drop PDFs, Markdown, Word docs, TXT files, and images into a watched folder. DocFlow auto-parses, chunks, and indexes them. Ask questions via a streaming web UI — everything stays on your machine. Supports Obsidian vault integration with automatic syntax cleanup.

---

## 目录 / Contents

- [功能特性](#功能特性--features)
- [技术栈](#技术栈--tech-stack)
- [环境要求](#环境要求--requirements)
- [快速开始](#快速开始--quick-start)
- [配置说明](#配置说明--configuration)
- [API 接口](#api-接口--api-reference)
- [项目结构](#项目结构--project-structure)
- [开发与测试](#开发与测试--development--testing)

---

## 功能特性 / Features

**中文**

- **多格式支持**：PDF（含扫描件 OCR）、Markdown、Word（.docx）、TXT、图片（JPG/PNG/WEBP/HEIC）
- **Obsidian 集成**：自动清洗 YAML frontmatter、`[[wikilinks]]`、callout、`%%注释%%`、`^block-id`，提取 tags 存入索引
- **混合检索**：向量检索（Qwen3-Embedding）+ SQLite FTS5 全文检索，RRF 融合排序
- **智能路由**：QueryRouter 自动识别精确查询 vs. 语义查询，动态调整检索权重
- **生成式精排**：Qwen3-Reranker-0.6B（MLX 运行，比 PyTorch MPS 快 26×）
- **流式问答**：SSE 协议，引用先返回，逐 token 流式生成答案
- **本地 LLM**：Qwen3-4B / Qwen3-8B（mlx-lm 进程内运行，TTFT ~2–4s）
- **图片理解**：Qwen2.5-VL-7B-Instruct（mlx-vlm，VLM 懒加载）
- **多路径监控**：watchdog 监控多个目录，支持递归扫描，3 秒去抖防止连续保存重复 ingest
- **自动清理**：启动时自动检测并清除磁盘上已删除文件的 DB 记录和 Qdrant 向量
- **mtime 快跳**：大 vault 启动加速 — 文件 mtime 未变时跳过 SHA-256 hash 计算
- **完全本地**：Qdrant 向量库 + SQLite 元数据，数据不出本机

**English**

- **Multi-format**: PDF (native + OCR), Markdown, Word (.docx), TXT, images (JPG/PNG/WEBP/HEIC)
- **Obsidian integration**: Auto-cleans YAML frontmatter, `[[wikilinks]]`, callouts, `%%comments%%`, `^block-ids`; extracts tags into the index
- **Hybrid retrieval**: Dense vector search (Qwen3-Embedding) + SQLite FTS5 full-text search, fused with RRF
- **Query routing**: QueryRouter auto-detects exact vs. semantic queries and adjusts retrieval weights
- **Generative reranking**: Qwen3-Reranker-0.6B via MLX — 26× faster than PyTorch MPS
- **Streaming answers**: SSE protocol — citations first, then token-by-token generation
- **Local LLM**: Qwen3-4B / Qwen3-8B via mlx-lm (in-process, TTFT ~2–4s)
- **Image understanding**: Qwen2.5-VL-7B-Instruct via mlx-vlm (lazy-loaded)
- **Multi-path watching**: watchdog monitors multiple directories with recursion and 3s debounce
- **Auto-cleanup**: Detects deleted files on startup and removes orphaned DB records + Qdrant vectors
- **mtime fast-skip**: Speeds up startup for large vaults — skips SHA-256 when file mtime is unchanged
- **Fully local**: Qdrant + SQLite — nothing leaves the machine

---

## 技术栈 / Tech Stack

| 层 / Layer | 技术 / Technology | 说明 / Notes |
|---|---|---|
| 后端框架 | FastAPI + Uvicorn | SSE 流式，14 个 API 端点 |
| 向量数据库 | Qdrant（Docker 本地） | 1024 维，COSINE 距离 |
| 全文检索 | SQLite FTS5（BM25 + trigram） | 进程内，O(log N) |
| Embedding | Qwen3-Embedding-0.6B | sentence-transformers，CPU |
| 精排 | Qwen3-Reranker-0.6B | mlx-lm，Apple Silicon |
| LLM | Qwen3-4B / 8B-4bit | mlx-lm，进程内 |
| VLM（图片） | Qwen2.5-VL-7B-Instruct-4bit | mlx-vlm，懒加载 |
| OCR（扫描 PDF） | glm-ocr | via Ollama |
| 文件监控 | watchdog | 多目录，递归可选，3s debounce |
| 前端 | 单文件 HTML（vanilla JS） | SSE 流式，浅色主题 |

---

## 环境要求 / Requirements

- **Apple Silicon Mac**（M1/M2/M3/M4/M5），统一内存 ≥ 16 GB（推荐 32 GB）
- Python 3.11+
- [Docker Desktop](https://www.docker.com/products/docker-desktop/)（运行 Qdrant）
- [Ollama](https://ollama.com/)（仅 OCR 扫描件时需要）

> **内存参考**：正常使用（无 VLM）约 3.4 GB；启用图片 VLM 后约 7.4 GB。

---

## 快速开始 / Quick Start

### 1. 启动依赖 / Start dependencies

```bash
# Qdrant 向量库
docker run -d --name qdrant -p 6333:6333 qdrant/qdrant

# （仅扫描件 PDF 需要）OCR 模型
ollama pull glm-ocr
```

### 2. 安装依赖 / Install

```bash
cd ~/Projects/docflow
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install python-docx mlx-vlm Pillow pillow-heif   # 多格式 + 图片支持
```

### 3. 下载模型 / Download models

首次启动时自动从 HuggingFace 下载以下模型（共约 6 GB）：

| 模型 | 大小 | 用途 |
|------|------|------|
| Qwen/Qwen3-Embedding-0.6B | 1.1 GB | 向量检索 |
| Qwen/Qwen3-Reranker-0.6B | 1.1 GB | 精排 |
| mlx-community/Qwen3-4B-4bit | 2.3 GB | LLM 问答 |
| mlx-community/Qwen2.5-VL-7B-Instruct-4bit | ~4 GB | 图片理解（可选） |

### 4. 启动服务 / Start server

```bash
python main.py serve
# 或直接
.venv/bin/uvicorn src.api.app:app --host 0.0.0.0 --port 8000
```

### 5. 使用 / Use

```bash
# 打开 Web 界面
open http://localhost:8000

# 将文件放入监控目录（自动入库）
cp mydoc.pdf ~/Documents/DocFlow/
cp note.md ~/Documents/DocFlow/
cp report.docx ~/Documents/DocFlow/
cp diagram.png ~/Documents/DocFlow/

# 或手动触发全量扫描
python main.py scan

# 或直接 ingest 单个文件
python main.py ingest /path/to/file.pdf

# 或对真实语料做 dry-run benchmark（parse / chunk / embed，不写入索引）
python main.py benchmark README.md docs/HANDOFF-v3.md
```

---

## 配置说明 / Configuration

所有配置集中在 `config.yaml`：

```yaml
paths:
  watch_dirs:
    - path: "~/Documents/DocFlow"
      recursive: false
    # Obsidian vault 示例（自动清洗 frontmatter/wikilinks）：
    - path: "~/MyNotes/MyVault"
      recursive: true
      extensions: [".md"]
  # .obsidian/ .trash/ .git/ 目录自动排除，不会索引

llm:
  backend: "mlx"                              # local（Ollama）| mlx | claude
  mlx_model: "mlx-community/Qwen3-4B-4bit"   # 默认 LLM
  mlx_model_enhanced: "mlx-community/Qwen3-8B-4bit"  # 增强模式

vlm:
  enabled: true           # 设为 false 可禁用图片支持（模型未下载时）
  model: "mlx-community/Qwen2.5-VL-7B-Instruct-4bit"
  max_tokens: 512

embedding:
  device: "cpu"           # 保持 cpu，避免 MPS + MLX 双 Metal 运行时冲突

ingest:
  parse_workers: 2
  microbatch_max_files: 8
  microbatch_max_chunks: 128
  adaptive_batch_char_budget: 32768
  embedding_cache: true
```

**切换 LLM 模型（运行时）**：

```bash
curl -X POST http://localhost:8000/api/llm \
  -H "Content-Type: application/json" \
  -d '{"model": "mlx-community/Qwen3-8B-4bit"}'
```

---

## API 接口 / API Reference

| 端点 | 方法 | 说明 |
|------|------|------|
| `/api/query` | POST | 同步查询 |
| `/api/query/stream` | POST | SSE 流式查询（主用） |
| `/api/ingest` | POST | 手动触发全量扫描 |
| `/api/queue` | GET | Ingest 队列状态（含阶段、chunk 进度、最近一次完成结果） |
| `/api/files` | GET | 文件列表（含状态、tags） |
| `/api/upload` | POST | 上传文件（支持所有格式） |
| `/api/file/{id}/preview` | GET | 预览原始文件 |
| `/api/history` | GET/DELETE | 查询历史 |
| `/api/history/search` | GET | 全文搜索历史（trigram） |
| `/api/favorites` | GET | 收藏列表 |
| `/api/favorites/{id}` | POST | 切换收藏 |
| `/api/summarize` | POST | 批量摘要生成 |
| `/api/llm` | GET/POST | 查看/切换 LLM |
| `/api/sources` | GET | 当前监控目录列表 |
| `/api/health` | GET | 健康检查 |

**SSE 流式协议示例**：

```bash
curl -X POST http://localhost:8000/api/query/stream \
  -H "Content-Type: application/json" \
  -d '{"question": "合同的主要条款是什么？"}'

# 响应事件流：
# event: citations
# data: [{"file_name":"contract.pdf","page_num":3,"snippet":"...","score":0.92}]
#
# event: token
# data: "根据合同第三条..."
#
# event: done
# data: ""
```

---

## 项目结构 / Project Structure

```
docflow/
├── config.yaml                  # 全局配置
├── main.py                      # CLI 入口（serve / ingest / scan）
├── requirements.txt
├── docflow.db                   # SQLite（文件状态 + chunk 元数据 + FTS5 + tags）
├── qdrant_id_counter.txt        # Qdrant 单调 ID 计数器
│
├── src/
│   ├── api/app.py               # FastAPI 路由 + lifespan + SSE + 启动清理
│   ├── query/
│   │   ├── engine.py            # QueryEngine（编排）
│   │   ├── retriever.py         # HybridRetriever + MLXReranker + QueryRouter
│   │   └── generator.py         # AnswerGenerator（MLX / Ollama / Claude 后端）
│   └── ingest/
│       ├── pipeline.py          # IngestPipeline（文件 → chunks → vectors）
│       ├── parsers/             # FileParser Protocol + Registry
│       │   ├── pdf_parser.py    # PDF（含 OCR）
│       │   ├── markdown_parser.py  # Obsidian 语法清洗 + tag 提取
│       │   ├── docx_parser.py
│       │   ├── txt_parser.py
│       │   └── image_parser.py  # 图片 → VLM 描述
│       ├── pdf_analyzer.py      # PyMuPDF 解析 + GLM-OCR
│       ├── chunker.py           # 结构化分块（512 tokens，10% overlap）
│       ├── embedder.py          # Embedding + Qdrant 写入
│       ├── store.py             # SQLite CRUD + FTS5 + tags + 删除清理
│       ├── queue.py             # 异步 ingest 队列
│       └── watcher.py           # 多目录文件监控（watchdog + debounce）
│
├── frontend/index.html          # 浅色主题 Web 界面
├── docs/                        # LESSONS.md 踩坑记录、交接文档
│   └── LESSONS.md               # 12 条开发踩坑经验
└── tests/                       # pytest 单元测试
```

---

## 开发与测试 / Development & Testing

```bash
# 运行单元测试
cd ~/Projects/docflow
.venv/bin/pytest tests/ -v

# 检查内存占用
vmmap $(lsof -ti:8000 | tail -1) | grep "Physical footprint:"

# 验证 FTS5 索引
sqlite3 docflow.db "SELECT COUNT(*) FROM chunks_fts;"
sqlite3 docflow.db "SELECT * FROM chunks_fts WHERE chunks_fts MATCH '机器学习' LIMIT 3;"

# 查看文件 tags
sqlite3 docflow.db "SELECT file_name, tags FROM files WHERE tags != '[]';"

# 健康检查
curl http://localhost:8000/api/health
```

**关键架构约束**（修改前必读）：

1. 所有 MLX 推理（Embedding 除外）必须通过 `ml_executor`（`max_workers=1`）串行执行
2. Embedding 固定用 CPU，禁止切换到 MPS（PyTorch MPS + MLX 双 Metal 运行时会导致内存爆炸至 21 GB+）
3. `pipeline.embedder._model` 与 `retriever._embed_model` 共享同一实例，绕过懒加载后须显式调用 `_ensure_collection()`
4. FTS5 rowid == `chunks.id`，写入由 `store.add_chunks()` 负责，不要绕过
5. 递归扫描自动排除 `.obsidian/`、`.trash/`、`.git/` 目录

---

## 许可证 / License

MIT

---

*正常使用内存约 3.4 GB（无 VLM）/ ~7.4 GB（含 VLM），适合 Apple Silicon Mac 16 GB 及以上机型。*
