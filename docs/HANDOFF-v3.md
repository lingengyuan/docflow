# DocFlow 交接文档 v3

> 日期：2026-03-21
> 状态：Phase 1–3 完成，Phase 4（MLX LLM）待实现

---

## 1. 项目简介

DocFlow 是一个**本地部署的 PDF 知识助手（RAG 系统）**。用户将 PDF 放入 `~/Documents/DocFlow/`，系统自动解析、分块、向量化。通过 Web 界面提问，系统检索相关片段并用 LLM 流式生成答案。

**运行环境**：M5 Mac，32GB 统一内存，macOS

---

## 2. 当前技术栈

| 层 | 技术 | 设备 | 说明 |
|----|------|------|------|
| 后端框架 | FastAPI + Uvicorn | CPU | 14 个 API 端点，SSE 流式 |
| 向量数据库 | Qdrant（本地） | — | 1024 维，COSINE 距离 |
| Embedding | Qwen3-Embedding-0.6B | **CPU** | sentence-transformers |
| 精排 | Qwen3-Reranker-0.6B | **MLX** | mlx-lm，26x 快于原 MPS 版 |
| 关键词检索 | BM25（rank-bm25）+ jieba | CPU | 中文分词已优化 |
| LLM | qwen2.5:7b via Ollama | Ollama | 默认；增强模式 qwen3:8b |
| OCR | glm-ocr via Ollama | Ollama | 扫描件 PDF 专用 |
| PDF 解析 | PyMuPDF | CPU | 原生文本提取 |
| 元数据 | SQLite（docflow.db） | — | files/chunks/history/favorites |
| 前端 | 单文件 HTML（vanilla JS） | — | 浅色 Ethereal Canvas 主题 |
| 分块 | StructuredChunker | CPU | 512 tokens，10% overlap，表格感知 |

---

## 3. 目录结构

```
docflow/
├── config.yaml                  # 全局配置（模型路径、设备、路径）
├── main.py                      # CLI 入口（serve / ingest / scan）
├── requirements.txt
├── docflow.db                   # SQLite
├── bm25_index.pkl               # BM25 持久化索引
├── qdrant_id_counter.txt        # Qdrant 单调 ID 计数器
│
├── src/
│   ├── api/app.py               # FastAPI 路由 + lifespan + SSE
│   ├── query/
│   │   ├── engine.py            # QueryEngine（编排）
│   │   ├── retriever.py         # HybridRetriever + MLXReranker
│   │   └── generator.py         # AnswerGenerator（Ollama / Claude 后端）
│   └── ingest/
│       ├── pipeline.py          # IngestPipeline（PDF → chunks → vectors）
│       ├── pdf_analyzer.py      # PDF 解析（原生 + GLM-OCR）
│       ├── chunker.py           # 结构化分块
│       ├── embedder.py          # Embedding + Qdrant + BM25 写入
│       ├── store.py             # SQLite CRUD
│       ├── queue.py             # 异步 ingest 队列（ml_executor）
│       └── watcher.py           # 文件夹监控（watchdog）
│
├── frontend/index.html          # 浅色主题前端（已重构完成）
│
└── docs/
    ├── HANDOFF-v3.md            # ★ 本文档（当前版本）
    ├── HANDOFF-v2.md            # 上版交接（重构前）
    ├── PLAN.md                  # 开发计划（含 Phase 4）
    ├── LESSONS.md               # 踩坑记录（文章素材，6 个坑）
    ├── DESIGN.md                # UI 设计规范
    └── code.html                # UI 设计稿 HTML
```

---

## 4. 启动方式

```bash
cd ~/Projects/docflow

# 前置条件（确认运行中）
docker ps | grep qdrant          # Qdrant 向量库
curl http://localhost:11434      # Ollama LLM 服务

# 启动
.venv/bin/uvicorn src.api.app:app --host 0.0.0.0 --port 8000

# 访问
open http://localhost:8000
```

---

## 5. 关键架构决策

### 单线程 ML Executor
```python
ml_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="ml-inference")
```
所有 MLX 推理（Embedding、Reranker、未来的 LLM）都通过这个 executor 串行执行，原因：
- MLX 的 Metal command queue 不是线程安全的
- 避免跨线程 Metal shader JIT 编译产生数值不一致

### 共享 Embedding 模型实例
```python
# app.py lifespan
shared_embed = query_engine.retriever._embed_model
pipeline.embedder._model = shared_embed
pipeline.embedder._ensure_collection(pipeline.embedder._vector_dim)
```
ingest pipeline 和 retriever 共用同一个 SentenceTransformer 对象，避免两个实例 MPS 编译出不同 shader 变体导致向量空间不一致。**绕过懒加载时必须显式调用 `_ensure_collection`。**

### Embedding 设备：CPU（非 MPS）
```yaml
embedding:
  device: "cpu"
```
保持 CPU，原因：
- PyTorch MPS + MLX 同时运行会产生 9000+ Metal IOAccelerator 区段，导致进程 Physical Footprint 达 21.5GB
- CPU Embedding 耗时仅从 0.1s → 0.3s，完全可接受
- 只保留一套 Metal 运行时（MLX）

### Embedding backend：默认 torch，ONNX 为实验开关
项目现在已经支持共享 `embedding.backend` 配置，ingest 与 query 会走同一套 backend 装载逻辑，
避免“索引时一种 backend、查询时另一种 backend”带来的向量空间偏移。

当前额外验证结论：
- `Qwen/Qwen3-Embedding-0.6B` 的本地 ONNX 导出需要补 `position_ids`，否则 encode 会失败
- 兼容性修复已经落地，测试通过
- 但在当前 M5 Mac（32GB）上，对真实 Markdown chunks 的实测里，`torch CPU` 仍明显快于 base `ONNX CPU`
- 因此 `config.yaml` 默认仍保持 `embedding.backend: "torch"`；ONNX 暂时只作为实验入口保留

小型实测（同一批 128 个真实 chunks）：
- `torch`: `34.388s`（`0.2687s/chunk`）
- `onnx`: `73.892s`（`0.5773s/chunk`）

结论：下一轮如果继续追 embedding 性能，更应该优先验证 TEI / Infinity 这类独立 runtime，而不是继续在当前本地 base ONNX 上深挖。

---

## 6. 当前性能（warm，4 个 PDF / 52 chunks）

| 阶段 | 耗时 |
|------|------|
| Embedding 查询向量 | 0.15–0.2s |
| Qdrant 向量检索 | 0.01s |
| BM25 检索 | 0.01s |
| MLX Reranker（~10 pairs） | 0.6–1.2s |
| **总检索** | **0.85–1.4s** |
| LLM 首 token（TTFT，warm） | 6–8s |
| LLM 生成速度 | ~21 t/s |
| **引用返回给用户** | **~0.9s** |

Python 进程 Physical Footprint：**30MB**（历史最高 21.5GB）

---

## 7. API 端点速查

| 端点 | 方法 | 用途 |
|------|------|------|
| `/api/health` | GET | 健康检查 |
| `/api/query` | POST | 同步查询 |
| `/api/query/stream` | POST | SSE 流式查询（主用） |
| `/api/ingest` | POST | 触发文件夹全量扫描 |
| `/api/queue` | GET | Ingest 队列状态（含阶段 / chunk 级进度） |
| `/api/files` | GET | 文件列表（含状态） |
| `/api/file/{id}/preview` | GET | PDF 原文（用于引用跳转） |
| `/api/upload` | POST | 上传 PDF |
| `/api/history` | GET/DELETE | 查询历史 |
| `/api/favorites` | GET | 收藏列表 |
| `/api/favorites/{id}` | POST | 切换收藏 |
| `/api/summarize` | POST | 批量摘要生成 |
| `/api/llm` | GET/POST | 查看/切换 LLM |

### SSE 流式协议

```
POST /api/query/stream
Body: {"question": "...", "file_filter": ["file.pdf"]}

event: citations
data: [{"file_name":"x.pdf", "page_num":3, "snippet":"...", "score":0.92}]

event: token
data: "根据"

event: done
data: ""
```

---

## 8. 模型清单

| 用途 | 模型 | 大小 | 位置 | 加载方式 |
|------|------|------|------|---------|
| Embedding | Qwen3-Embedding-0.6B | 1.1GB | HuggingFace cache | sentence-transformers |
| Reranker | Qwen3-Reranker-0.6B | 1.1GB | HuggingFace cache | mlx-lm |
| LLM（默认） | qwen2.5:7b | 4.7GB | Ollama | HTTP API |
| LLM（增强） | qwen3:8b | 5.2GB | Ollama | HTTP API |
| OCR | glm-ocr | 2.2GB | Ollama | HTTP API |

**Ollama 注意**：`qwen3:4b` / `qwen3:8b` 在 Ollama 上是 Thinking-only 特化版，无法关闭思考模式，不适合流式 RAG（详见 `docs/LESSONS.md` 坑 6）。

---

## 9. 已知问题与待办

| # | 问题 | 优先级 | 对应 Phase |
|---|------|--------|-----------|
| 1 | LLM TTFT 6–8s，主要是 Ollama 加载 context | P0 | **Phase 4** |
| 2 | qwen2.5:7b 无法用 qwen3 新特性 | P0 | **Phase 4** |
| 3 | Ollama thinking 模式与流式 RAG 冲突 | P0 | **Phase 4** |
| 4 | 无错误重试机制（LLM 超时直接失败） | P1 | — |
| 5 | chunk 去重用 text[:128] 可能漏重复 | P2 | — |
| 6 | 无集成测试 | P2 | — |

---

## 10. 下一步：Phase 4 MLX LLM（见 PLAN.md）

将 LLM 后端从 Ollama 迁移到 in-process MLX，使用原版 Qwen3 HuggingFace 模型（非 Thinking-only），通过 `enable_thinking=False` 禁用思考模式。

**预期收益**：
- TTFT 从 6–8s → 2–4s
- 彻底不需要 Ollama 进程（释放 ~5GB 内存）
- Qwen3-4B / Qwen3-8B 真正可用，质量优于 qwen2.5:7b

**新会话启动步骤**：
1. 读本文档了解现状
2. 读 `docs/PLAN.md` 的 Phase 4 章节
3. 读 `docs/LESSONS.md` 了解已踩的坑（避免重复）
4. 开始实现 `generator.py` 的 `mlx` 后端

---

## 11. 常用命令

```bash
# 启动
cd ~/Projects/docflow
.venv/bin/uvicorn src.api.app:app --host 0.0.0.0 --port 8000

# 检查内存
vmmap $(lsof -ti:8000) | grep "Physical footprint:"

# 检索性能
grep "\[perf\]" /tmp/docflow.log | tail -10

# 查看 Qdrant 状态
curl http://localhost:6333/collections/docflow

# 健康检查
curl http://localhost:8000/api/health

# 手动 ingest
curl -X POST http://localhost:8000/api/ingest

# 切换 LLM
curl -X POST http://localhost:8000/api/llm -H "Content-Type: application/json" \
  -d '{"model": "qwen3:8b"}'
```
