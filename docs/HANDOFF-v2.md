# DocFlow 交接文档 v2

> 日期：2026-03-20
> 上下文：准备新会话进行重构开发，本文档提供完整项目状态

---

## 1. 项目简介

DocFlow 是一个**本地部署的 PDF 知识助手**。用户将 PDF 放入 `~/Documents/DocFlow/`，系统自动解析、分块、向量化。通过 Web 界面提问，系统检索相关片段并用 LLM 生成答案（SSE 流式输出）。

**运行环境**：M5 Mac Air, 32GB RAM, 1TB SSD, macOS

---

## 2. 技术栈

| 层 | 技术 | 说明 |
|----|------|------|
| 后端框架 | FastAPI + Uvicorn | 14 个 API 端点，含 SSE 流式 |
| 向量数据库 | Qdrant（本地目录模式） | 1024 维，COSINE 距离 |
| 向量模型 | Qwen3-Embedding-0.6B | sentence-transformers，MPS 加速 |
| 精排模型 | Qwen3-Reranker-0.6B | 生成式 reranker，PyTorch MPS（**慢，需迁移 MLX**） |
| 关键词检索 | BM25（rank-bm25） | tokenizer 用 split()（**需改 jieba**） |
| LLM | Ollama（qwen2.5:7b 默认） | 可选 qwen3.5:27b（17GB）、Claude API |
| OCR | GLM-OCR via Ollama | 扫描件 PDF 专用 |
| PDF 解析 | PyMuPDF | 原生 PDF 文本提取 |
| 元数据 | SQLite（docflow.db） | files / chunks / history / favorites |
| 前端 | 单文件 HTML（vanilla JS） | **待按设计稿重构** |
| 分块 | StructuredChunker | 512 tokens, 10% overlap, 表格感知 |

---

## 3. 目录结构

```
docflow/
├── config.yaml                  # 全局配置
├── main.py                      # CLI 入口（serve / ingest / scan）
├── requirements.txt             # Python 依赖（100 个包）
├── docflow.db                   # SQLite（files/chunks/history/favorites）
├── bm25_index.pkl               # BM25 持久化索引
├── qdrant_id_counter.txt        # Qdrant 单调递增 ID 计数器
├── qdrant_storage/              # Qdrant 本地向量存储
│
├── src/
│   ├── api/
│   │   └── app.py               # FastAPI 路由 + 中间件 + SSE
│   ├── query/
│   │   ├── engine.py            # QueryEngine（编排检索 + 生成）
│   │   ├── retriever.py         # HybridRetriever + Qwen3Reranker
│   │   └── generator.py         # AnswerGenerator（Ollama / Claude）
│   └── ingest/
│       ├── pipeline.py          # IngestPipeline（PDF → chunks → vectors）
│       ├── pdf_analyzer.py      # PDF 解析（原生 + GLM-OCR）
│       ├── chunker.py           # 结构化分块（text/table/table_summary）
│       ├── embedder.py          # Embedding + Qdrant + BM25 写入
│       ├── store.py             # SQLite CRUD
│       ├── queue.py             # 异步 ingest 队列
│       └── watcher.py           # 文件夹监控（watchdog）
│
├── frontend/
│   └── index.html               # 当前前端（暗色原型，待重构）
│
├── tests/                       # 46 个单元测试
│   ├── test_chunker.py          # 14 tests
│   ├── test_generator.py        # 7 tests
│   ├── test_retriever.py        # 7 tests
│   ├── test_store.py            # 8 tests
│   └── test_pdf_analyzer.py     # 10 tests
│
├── docs/
│   ├── PLAN.md                  # ★ 重构计划（本次新建）
│   ├── HANDOFF-v2.md            # ★ 本交接文档（本次新建）
│   ├── HANDOFF.md               # 上一版交接文档（性能优化阶段）
│   ├── DESIGN.md                # UI 设计规范（Ethereal Canvas）
│   ├── code.html                # UI 设计稿 HTML 代码
│   └── screen.png               # UI 设计效果图
│
└── data/                        # 测试数据或临时数据
```

---

## 4. 启动方式

```bash
# 前置条件
# 1. Qdrant 运行中（Docker）
docker start qdrant  # 或 docker run -d --name qdrant -p 6333:6333 qdrant/qdrant

# 2. Ollama 运行中
ollama serve  # 或确认已在后台运行

# 3. 模型已下载
ollama list  # 确认 qwen2.5:7b, glm-ocr 存在

# 启动 DocFlow
cd ~/Projects/docflow
python main.py serve
# 访问 http://localhost:8000
```

---

## 5. API 端点速查

| 端点 | 方法 | 用途 |
|------|------|------|
| `/api/health` | GET | 健康检查 |
| `/api/query` | POST | 同步查询 `{question, file_filter?}` |
| `/api/query/stream` | POST | SSE 流式查询（主用） |
| `/api/ingest` | POST | 触发文件夹扫描 |
| `/api/queue` | GET | Ingest 队列状态 |
| `/api/files` | GET | 文件列表 |
| `/api/file/{id}/preview` | GET | PDF 原文预览 |
| `/api/upload` | POST | 上传 PDF |
| `/api/history` | GET/DELETE | 查询历史 |
| `/api/favorites` | GET | 收藏列表 |
| `/api/favorites/{id}` | POST | 切换收藏 |
| `/api/summarize` | POST | 批量摘要 |
| `/api/llm` | GET/POST | 查看/切换 LLM |

### SSE 协议

```
POST /api/query/stream
Body: {"question": "...", "file_filter": [1, 2]}  // file_filter 可选

Response (text/event-stream):
event: citations
data: [{"file_name":"...", "page_num":1, "snippet":"...", "score":0.8}]

event: token
data: "根据"

event: token
data: "文档"

event: done
data: ""
```

---

## 6. 数据库 Schema

```sql
-- 文件表
CREATE TABLE files (
    id INTEGER PRIMARY KEY,
    file_path TEXT UNIQUE,
    file_name TEXT,
    file_hash TEXT,           -- SHA256 去重
    status TEXT DEFAULT 'pending',  -- pending|processing|done|error
    total_pages INTEGER,
    is_scanned INTEGER DEFAULT 0,
    chunk_count INTEGER DEFAULT 0,
    error_msg TEXT,
    created_at TIMESTAMP,
    updated_at TIMESTAMP
);

-- 分块表
CREATE TABLE chunks (
    id INTEGER PRIMARY KEY,
    file_id INTEGER REFERENCES files(id),
    qdrant_id INTEGER,        -- 对应 Qdrant 中的点 ID
    chunk_type TEXT,           -- text|table|table_summary
    page_num INTEGER,
    section TEXT,              -- 面包屑路径 "Chapter > 1.2 Background"
    char_count INTEGER
);

-- 查询历史
CREATE TABLE history (
    id INTEGER PRIMARY KEY,
    question TEXT,
    answer TEXT,
    citations TEXT,            -- JSON 字符串
    file_filter TEXT,          -- JSON 字符串
    created_at TIMESTAMP
);

-- 收藏
CREATE TABLE favorites (
    id INTEGER PRIMARY KEY,
    file_id INTEGER UNIQUE REFERENCES files(id),
    created_at TIMESTAMP
);
```

---

## 7. 当前数据状态

- **已索引文件**：4 个 PDF，52 个 chunks
- **Qdrant**：`docflow` collection，1024 维，COSINE 距离
- **Qdrant ID 计数器**：见 `qdrant_id_counter.txt`
- **模型缓存**（`~/.cache/huggingface/hub/`）：
  - `Qwen/Qwen3-Embedding-0.6B` (584MB) ← 使用中
  - `Qwen/Qwen3-Reranker-0.6B` (999MB) ← 使用中
  - `BAAI/bge-m3` ← 备用
- **Ollama 模型**：
  - `qwen2.5:7b` (4.7GB) ← 默认 LLM
  - `qwen3.5:27b` (17GB) ← 增强模式
  - `glm-ocr` (2.2GB) ← OCR 专用

---

## 8. 查询流程（含耗时）

```
用户提问
  │
  ▼
QueryEngine.query_stream(question, file_filter)
  │
  ├── HybridRetriever.retrieve()
  │   ├── Qwen3-Embedding 编码查询          0.3-0.5s
  │   ├── Qdrant 向量检索 top-20            0.01s
  │   ├── BM25 关键词检索 top-20            0.01s
  │   ├── RRF 融合 + 阈值过滤(0.4)          0.01s
  │   ├── 去重 (file, page, text[:128])     0.01s
  │   └── Qwen3-Reranker 精排 → top-5      ⚠️ 0.6-14s（瓶颈）
  │
  └── AnswerGenerator.generate_stream()
      └── Ollama qwen2.5:7b 流式生成        1-15s（流式体验可接受）
```

**关键瓶颈**：Reranker 在 PyTorch MPS 上的延迟。生成式 reranker 每对 pair 需完整因果 LM 前向传播，MPS 不擅长此类顺序操作。

---

## 9. 性能优化历史

| 轮次 | 问题 | 方案 | 结果 |
|------|------|------|------|
| 第 1 轮 | 三模型 21GB 挤爆内存 | 4B→0.6B 模型，移除 reranker | 分钟级 → 12-17s |
| 第 2 轮 | 同步等待 + 重复引用 | SSE 流式 + 去重 + 阈值过滤 | 感知秒级 |
| 第 3 轮 | 加回 reranker 质量提升但慢 | Qwen3-Reranker-0.6B | 质量好但 MPS 慢 |
| **待做** | Reranker MPS 慢 | **迁移到 MLX 框架** | 预估 14s → <1s |

---

## 10. 设计稿说明

新前端已有完整设计：

- **`docs/DESIGN.md`**：设计规范（Ethereal Canvas 风格）
  - 浅色主题、无边框规则、毛玻璃浮层
  - 三层 Surface 色彩体系
  - 字体：PingFang SC + Inter
  - Material Symbols 图标
  
- **`docs/code.html`**：设计稿 HTML 实现代码（Tailwind CSS）

- **`docs/screen.png`**：最终效果图
  - 左侧：图标导航（Chat / Library / History / 设置）
  - 中间：对话区（气泡消息 + 引用卡片 + 操作按钮）
  - 右侧：PDF 预览面板（可折叠，显示页码）
  - 顶栏：Logo + 文件过滤 pill + 模型选择 + 状态灯

---

## 11. 已知问题清单

| # | 问题 | 位置 | 优先级 |
|---|------|------|--------|
| 1 | Reranker MPS 极慢（10-14s/20 pairs） | `src/query/retriever.py` Qwen3Reranker | P0 |
| 2 | 前端与设计稿不匹配 | `frontend/index.html` | P0 |
| 3 | BM25 中文分词无效 | `retriever.py` + `embedder.py` | P1 |
| 4 | 无错误重试机制 | `generator.py` Ollama 调用 | P1 |
| 5 | Metal shader 首次编译慢 +5s | MPS 冷启动 | P2 |
| 6 | 无集成/端到端测试 | `tests/` | P2 |
| 7 | chunk 去重用 text[:128] 可能漏重复 | `retriever.py` | P3 |

---

## 12. 重构计划入口

详见 **`docs/PLAN.md`**，核心三个 Phase：

1. **Phase 1**：后端性能 — MLX Reranker + BM25 中文分词 + 错误处理
2. **Phase 2**：前端重构 — 按 DESIGN.md + screen.png 重做 UI
3. **Phase 3**：稳定性 — 测试补全 + 文档 + CI

**建议新会话启动步骤**：
1. 读取本文档 `docs/HANDOFF-v2.md` 获取全貌
2. 读取 `docs/PLAN.md` 了解具体任务
3. 从 Phase 1.1（MLX Reranker）开始，或根据需要调整优先级

---

## 13. 常用命令

```bash
# 启动服务
python main.py serve

# 手动 ingest 单文件
python main.py ingest /path/to/file.pdf

# 扫描 watch_dir
python main.py scan

# 运行测试
pytest tests/ -v

# 查看 Qdrant 状态
curl http://localhost:6333/collections/docflow

# 查看 Ollama 模型
ollama list

# 健康检查
curl http://localhost:8000/api/health
```
