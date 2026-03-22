# DocFlow 演进规划

## Context

DocFlow 是一个本地 PDF 知识助手，当前 Phase 3 已完成（MLX Reranker + 前端重设计 + BM25 jieba），Phase 4（MLX LLM 后端）代码已基本就绪。用户提出的演进方向：Rust 重写评估、多格式支持、智能路由、桌面客户端技术栈、多路径监控、万级文档规模支撑、图片知识库管理。

---

## Q1: Python → Rust 重写？

### 决策：不做全量重写。Tauri v2 做桌面壳，Python 后端保留。

**理由**：DocFlow 95% 的计算在 ML 推理（embedding、reranker、LLM），全部依赖 Python 生态（sentence-transformers、mlx-lm、PyTorch）。Rust 的 Candle/Mistral.rs 目前不支持 Qwen3-Embedding 等模型。重写只能替换 I/O 层和文本处理，收益极小。

**架构**：
```
Tauri v2 Shell (.app / .dmg)
├── WebView (Solid.js + Tailwind)
├── Rust Core (生命周期管理、系统托盘、全局快捷键、文件拖放)
└── Python Sidecar (FastAPI + ML 推理，现有代码基本不变)
```

**Tauri 带来的收益**：
| 维度 | 当前 (浏览器标签) | Tauri |
|------|-------------------|-------|
| 分发 | clone + pip install + 手动启动 | `.app` 一键安装 |
| 体验 | 浏览器标签 | 系统托盘、全局热键、原生文件拖放 |
| 内存 | 浏览器开销 | ~30MB WebView |

**PyO3 加速**：当前没有值得用 Rust 加速的瓶颈。Phase 8 按需考虑。

---

## Q2: 多格式文件支持 + 是否需要向量化？

### 决策：支持 `.md` / `.docx` / `.txt`，所有文件统一走完整 pipeline（chunk → embed → BM25）。

**为什么不区分"短文件跳过向量化"**：
1. Embedding 成本极低：32 条文本 CPU 批处理 ~0.3s
2. 两条代码路径（有向量/无向量）增加查询复杂度，违背 DRY
3. BM25-only 无法捕获语义相似（"机器学习" ≠ "AI 算法"）

### 实现方案：FileParser Protocol + ParserRegistry

```
文件到达 → 扩展名匹配 → ParserRegistry.resolve(path) → Parser.parse()
→ ParsedDocument（统一格式） → StructuredChunker → Embedder → Qdrant + BM25
```

| Parser | 依赖 | 说明 |
|--------|------|------|
| `PDFParser` | PyMuPDF (已有) | 从 `pdf_analyzer.py` 重构 |
| `MarkdownParser` | 无新依赖 | UTF-8 读取，chunker 原生支持 `#` headers |
| `DocxParser` | `python-docx` (新增) | 段落 + 表格提取，heading style → `#` |
| `TxtParser` | 无新依赖 | UTF-8 读取，单页 ParsedDocument |

**需要修改的文件**：

| 文件 | 修改内容 |
|------|----------|
| `src/ingest/parsers/` (新建) | `base.py` (Protocol), `pdf_parser.py`, `markdown_parser.py`, `docx_parser.py`, `txt_parser.py`, `__init__.py` (Registry) |
| `src/ingest/pipeline.py:26-28,45-48,65,97` | `PDFAnalyzer` → `ParserRegistry`，`ingest()` 按扩展名解析 |
| `src/ingest/watcher.py` | 后缀过滤从 `.pdf` 改为可配置集合 |
| `src/api/app.py` | `/api/upload` 接受所有支持格式；`trigger_ingest` glob 扩展 |
| `main.py` | `scan()` glob 所有支持扩展名 |
| `config.yaml` | 新增 `supported_extensions` |
| `requirements.txt` | 新增 `python-docx` |

---

## Q2.5: 多路径监控（Obsidian、任意文件夹）

### 决策：`watch_dirs` 改为列表，支持多目录 + 递归扫描。

**当前问题**（代码审查确认）：
- `config.yaml` 中 `paths.watch_dir` 是单个字符串
- `watcher.py:56` 用 `recursive=False` 只监控一层
- `watcher.py:64-70` 用 `self.watch_dir.glob("*.pdf")` 浅层扫描
- `app.py:88` 只创建一个 `FolderWatcher` 实例

### config.yaml 改为：

```yaml
paths:
  watch_dirs:                          # 列表，替代原 watch_dir
    - path: "~/Documents/DocFlow"
      recursive: false                 # 默认行为，兼容旧版
    - path: "~/Obsidian/MyVault"
      recursive: true                  # 递归扫描子目录
      extensions: [".md"]              # 可选：覆盖全局 supported_extensions
    - path: "~/Work/Reports"
      recursive: true
  supported_extensions: [".pdf", ".md", ".docx", ".txt"]
  db_path: "~/Projects/docflow/docflow.db"
  bm25_index: "~/Projects/docflow/bm25_index.pkl"
```

### 实现改动

**`src/ingest/watcher.py`**：
- `FolderWatcher.__init__` 接收 `list[WatchDir]` 而非单个 `Path`
- 为每个目录注册独立的 watchdog observer（`recursive` 参数透传）
- `scan_existing()` 遍历所有目录，按各自 `recursive` 和 `extensions` 配置 glob

**`src/api/app.py`**：
- 解析 `watch_dirs` 列表
- 传入多目录配置给 `FolderWatcher`
- `/api/ingest` 手动扫描时遍历所有目录

**`main.py`**：
- `scan()` 命令遍历所有 `watch_dirs`

**前端**：
- Settings 页面展示已监控目录列表
- 支持添加/移除监控目录（调用新 API）

### API 新增

```
GET  /api/sources          → 返回所有监控目录及其状态
POST /api/sources          → 添加新目录 {"path": "...", "recursive": true}
DELETE /api/sources/{id}   → 移除监控目录
```

运行时支持动态添加/移除（不需要重启），通过 watchdog observer 的 `schedule/unschedule` 实现。

---

## Q3: 是否需要 Agent 做智能路由？

### 决策：不需要 LLM Agent。用规则路由，~30 行代码。

**理由**：
- LLM Agent 每次查询增加 500-2000ms 延迟
- DocFlow 只有两种检索策略（向量 + BM25），决策是确定性的
- Reranker 已经在做质量判断
- 规则可测试、可调试、0ms 开销

### QueryRouter 设计（加入 `src/query/retriever.py`）

```python
class QueryRouter:
    KEYWORD_PATTERNS = [
        r'"[^"]+"',            # 引号短语 → 精确匹配
        r'\b\d{4}[-/]\d{2}',  # 日期 → 关键词
        r'\.\w{2,4}\b',       # 文件扩展名 → 关键词
    ]

    @classmethod
    def classify(cls, query: str) -> dict:
        keyword_signals = sum(1 for p in cls.KEYWORD_PATTERNS if re.search(p, query))
        if keyword_signals >= 2:
            return {"bm25_weight": 2.0, "vec_weight": 0.5}
        elif len(query) > 50 and keyword_signals == 0:
            return {"bm25_weight": 0.5, "vec_weight": 2.0}
        return {"bm25_weight": 1.0, "vec_weight": 1.0}
```

**升级路径**：如果未来查询模式变复杂，再引入 LLM 分类器。YAGNI。

---

## Q4: 桌面客户端技术栈 + 项目方向

### 推荐技术栈

| 层 | 技术 | 理由 |
|----|------|------|
| 桌面壳 | **Tauri v2** | 97% 小于 Electron，30MB 内存，原生安全模型 |
| 前端 | **Solid.js + Tailwind** | 7KB runtime，JSX 语法，编译为高效 DOM 操作 |
| 后端 | **Python FastAPI** (现有) | ML 生态不可替代，作为 Tauri sidecar 运行 |
| 向量库 | **Qdrant** → **LanceDB** (Phase 7) | LanceDB 文件级、进程内、无 Docker 依赖 |
| ML 推理 | **MLX** (Apple Silicon) | 已集成，性能优于 PyTorch MPS |
| 模型分发 | 首次启动下载 | 缓存于 `~/Library/Application Support/DocFlow/models/` |

---

## Q5: 万级文档规模评估

### 规模假设

| 指标 | 数值 |
|------|------|
| 文档数 | 30,000 份 |
| 平均 chunk/文档 | 50 |
| 总 chunk 数 | 1,500,000 |
| 向量维度 | 384 (Qwen3-Embedding) |
| 平均 chunk 文本 | ~2KB |

### 瓶颈分析（代码审查确认）

#### 1. BM25 — 最大瓶颈

**现状**：
- `embedder.py:46-50`：整个 BM25 语料库以 `list[str]` 存在内存
- `embedder.py:197-199`：每次 `_rebuild_bm25()` 对全量语料做 jieba 分词
- `retriever.py:279-311`：BM25 搜索是 O(N) 全量扫描
- `app.py:83`：每次 ingest 完成后 `reload_bm25()` 反序列化整个 pickle

**1.5M chunks 时的问题**：
| 指标 | 当前 (1K chunks) | 预估 (1.5M chunks) |
|------|-------------------|---------------------|
| 内存占用 | ~5MB | **3-4GB** |
| Pickle 加载 | <0.1s | **10-30s** |
| BM25 搜索延迟 | <10ms | **1-5s** |
| 每次 ingest 后 rebuild | 不感知 | **30s+ 阻塞** |

**解决方案：BM25 迁移到 SQLite FTS5**：
- SQLite FTS5 内置全文搜索，支持 BM25 排序
- 磁盘存储，O(log N) 查询，增量更新
- 不需要全量加载到内存，不需要 pickle
- 中文分词：自定义 tokenizer 或 jieba 预分词后存入
- 完全消除 `bm25_index.pkl` 和内存中的 `_bm25_corpus`

```python
# 新方案：SQLite FTS5 替代 rank_bm25
# 在 docflow.db 中创建 FTS5 虚拟表
CREATE VIRTUAL TABLE chunks_fts USING fts5(
    text,
    content='chunks',          -- 外部内容表
    content_rowid='id',
    tokenize='unicode61'       -- 或自定义 jieba tokenizer
);
```

**文件修改**：
- `src/ingest/embedder.py`：移除 `_bm25_corpus`、`_rebuild_bm25()`、pickle 相关代码
- `src/ingest/store.py`：新增 FTS5 表创建 + 增量插入
- `src/query/retriever.py`：BM25 搜索改为 FTS5 SQL 查询
- 删除 `bm25_index.pkl`

#### 2. Qdrant — 可以支撑

**Qdrant 本地模式**在百万级向量时表现良好：
- 1.5M × 384-dim × 4 bytes = **2.3GB** 向量存储
- 使用 HNSW 索引，查询 O(log N)，top-20 查询 <50ms
- 注意：当前 `embedder.py:138` 在 payload 中存了全文 `text`（与 SQLite 重复），应移除以节省空间

**优化项**：
- `embedder.py:138`：payload 移除 `text` 字段（查询时从 SQLite 取），节省 ~3GB
- 配置 Qdrant 的 `on_disk=True`（已是默认）

#### 3. SQLite — 可以支撑

- 1.5M chunks + 30K files：数据量 <1GB，SQLite 轻松处理
- `store.py:93-95` 已有索引（`idx_chunks_file_id`, `idx_files_hash`, `idx_files_status`）
- **需要修复**：`store.py:204-213` 的 `list_files()` 无分页 → 加 `LIMIT/OFFSET`

#### 4. 文件监控 — 需要限流

30,000 文件分布在多目录，watchdog 可能产生大量事件。
- 添加 debounce（已有 `_debounce` dict，OK）
- 初始扫描时用队列限流，避免一次性提交 30K ingest 任务

### 规模支撑总结

| 组件 | 当前能否支撑万级？ | 需要的改动 |
|------|---------------------|------------|
| BM25 (pickle/内存) | **不能** — 3-4GB 内存 + 秒级延迟 | **迁移到 SQLite FTS5** |
| Qdrant 向量检索 | 可以 | 移除 payload 中的 text 冗余 |
| SQLite 元数据 | 可以 | list_files() 加分页 |
| 文件监控 | 可以（需限流） | 初始扫描加队列限流 |
| Embedding 推理 | 可以（批处理） | 无需改动 |

---

## Q6: 图片知识库管理

### 决策：支持图片入库，通过 VLM 生成描述文本 → 走标准文本 pipeline。

### 架构设计

```
图片到达 (.jpg/.png/.webp/.heic)
    │
    ▼
ImageParser (新增)
    ├─ 格式标准化 → WebP (pillow + pillow-heif)
    ├─ VLM 描述生成 → Qwen2.5-VL-7B (mlx-vlm, 4-bit)
    │   输出: 中文描述文本 (50-200 字)
    ├─ 可选: OCR 提取 (图片中的文字)
    └─ 构建 ParsedDocument:
        pages=[PageContent(text=描述文本, page_num=1)]
        metadata: {type: "image", original_path, dimensions, format}
    │
    ▼
标准 pipeline (StructuredChunker → Embedder → Qdrant + FTS5)
```

**核心思路**：图片 → 文本描述 → 和文档走同一条 pipeline。不需要单独的向量集合或检索路径。

### 为什么不用 CLIP 做图片向量？

| 方案 | 优点 | 缺点 |
|------|------|------|
| **VLM → 文本 → Qwen3-Embedding** (推荐) | 统一 pipeline，单一向量空间，查询不需要改 | VLM 推理 2-3s/图片 |
| **CLIP 图片向量** | 支持图搜图 | 384-dim CLIP ≠ 384-dim Qwen3（不同投影空间），需要第二个集合 + 查询合并逻辑 |

VLM 方案更简单、更符合 DRY/正交性。图搜图场景如果未来需要，再加 CLIP 集合。

### 技术选型

| 组件 | 选择 | 理由 |
|------|------|------|
| VLM | **Qwen2.5-VL-7B-4bit** (mlx-vlm) | Apple Silicon 原生，2-3s/图，4-7GB 内存，中文优秀 |
| 格式转换 | **Pillow + pillow-heif** | 支持 JPG/PNG/WEBP/HEIC，纯 Python |
| SVG | **cairosvg → PNG → VLM** | SVG 需先栅格化 |
| 存储 | WebP 缩略图 + 原始路径引用 | 不复制原图，只存引用 |

### 新增依赖

```
mlx-vlm          # VLM 推理
pillow-heif       # HEIC 支持
cairosvg          # SVG → PNG (可选)
```

### 需要修改/新增的文件

| 文件 | 说明 |
|------|------|
| `src/ingest/parsers/image_parser.py` (新建) | ImageParser：格式检测 → VLM 描述 → ParsedDocument |
| `src/ingest/parsers/__init__.py` | 注册 ImageParser，扩展名 `.jpg/.png/.webp/.heic` |
| `config.yaml` | 新增 `vlm` 配置段 + `supported_extensions` 加图片格式 |
| `frontend/` | 图片缩略图展示、图片详情页 |

### config.yaml 新增

```yaml
vlm:
  model: "mlx-community/Qwen2.5-VL-7B-Instruct-4bit"
  prompt: "请用中文详细描述这张图片的内容，包括文字、图表、场景等信息。"
  max_tokens: 512

paths:
  supported_extensions: [".pdf", ".md", ".docx", ".txt", ".jpg", ".png", ".webp", ".heic"]
```

### 图片处理资源估算

| 指标 | 数值 |
|------|------|
| VLM 推理 | 2-3s/图 (Qwen2.5-VL-7B-4bit, Apple Silicon) |
| VLM 内存 | 4-7GB (与 LLM 共享统一内存，可按需加载/卸载) |
| 每张图存储开销 | ~2KB (描述文本) + ~2KB (向量) + ~0.5KB (元数据) |
| 10,000 张图 | ~45MB 额外向量/文本存储（忽略不计） |
| 批量处理 10K 图 | ~6-8 小时（后台队列，不阻塞使用） |

---

## 分阶段路线图

### Phase 4：MLX LLM 后端（当前，代码已就绪）
- `generator.py` 已实现 `_stream_mlx` / `_call_mlx` / `_load_mlx_model`
- `config.yaml` 已配置 `llm.backend: "mlx"`
- **剩余工作**：端到端测试、验证流式输出、基准 TTFT（目标 2-4s）
- **文件**：`src/query/generator.py`, `src/api/app.py`
- **工作量**：1-2 天

### Phase 5：多格式 + 多路径 + BM25 升级
- **5a. FileParser Protocol + ParserRegistry**
  - 新建 `src/ingest/parsers/` 目录
  - 重构 `PDFAnalyzer` → `PDFParser`
  - 实现 `MarkdownParser` / `DocxParser` / `TxtParser`
- **5b. 多路径监控**
  - `config.yaml` 的 `watch_dir` → `watch_dirs` 列表
  - `FolderWatcher` 支持多目录 + 递归
  - 新增 `/api/sources` CRUD 端点（运行时动态增删）
- **5c. BM25 → SQLite FTS5**（为万级规模做准备）
  - `store.py` 新增 FTS5 虚拟表
  - `embedder.py` 移除 pickle/内存 BM25
  - `retriever.py` BM25 搜索改为 FTS5 查询
  - 删除 `bm25_index.pkl`
- **5d. Qdrant 瘦身**
  - payload 移除 `text` 字段（从 SQLite 取）
  - `list_files()` 加分页
- **文件**：`pipeline.py`, `watcher.py`, `app.py`, `main.py`, `store.py`, `embedder.py`, `retriever.py`, `config.yaml`
- **工作量**：5-7 天

### Phase 6：Query Router + 图片支持
- **6a. QueryRouter**（~30 行）
  - 修改 `_rrf_fuse()` 接受权重
  - 接入 `HybridRetriever.retrieve()`
  - 添加文件类型过滤
- **6b. ImageParser**
  - 新增 `src/ingest/parsers/image_parser.py`
  - 集成 mlx-vlm (Qwen2.5-VL-7B-4bit) 生成描述
  - 格式支持：JPG/PNG/WEBP/HEIC
  - VLM 模型按需加载/卸载（与 LLM 共享内存）
- **文件**：`retriever.py`, `engine.py`, `image_parser.py`, `config.yaml`
- **工作量**：4-5 天

### Phase 7：Tauri 桌面客户端
- 初始化 `docflow-desktop/` Tauri v2 项目
- 前端迁移：Solid.js + Tailwind
- Tauri sidecar：打包 Python 后端
- Qdrant → LanceDB 迁移（新建 `VectorStore` Protocol）
- 系统托盘、全局热键、原生文件拖放
- 模型首次启动下载 + 进度条
- **工作量**：2-3 周

### Phase 8：性能优化（按需）
- 仅在 profiling 发现瓶颈后执行
- 候选：Rust BM25（PyO3）、DuckDB、批量 reranking、WebSocket 进度推送
- CLIP 向量集合（图搜图，如果有需求）

---

## 验证方案

### Phase 5 验证（多格式 + 多路径 + FTS5）
```bash
# 1. 配置多路径
# config.yaml 添加 watch_dirs 包含 Obsidian vault

# 2. 测试多格式 ingest
python main.py ingest ~/Obsidian/MyVault/note.md
python main.py ingest ~/Documents/report.docx
python main.py ingest ~/Documents/readme.txt

# 3. 验证 FTS5
sqlite3 docflow.db "SELECT * FROM chunks_fts WHERE chunks_fts MATCH '关键词' ORDER BY rank LIMIT 5"

# 4. 验证多路径 watcher
# 在 Obsidian vault 中新建 .md 文件，观察自动 ingest 日志

# 5. 验证动态源管理
curl -X POST http://localhost:8000/api/sources -d '{"path": "~/NewFolder", "recursive": true}'
curl http://localhost:8000/api/sources
```

### Phase 6 验证（Query Router + 图片）
```bash
# 图片 ingest
python main.py ingest ~/Photos/diagram.png
# 日志应显示：VLM 描述生成 → chunk → embed

# 查询包含图片描述
curl -X POST http://localhost:8000/api/query -d '{"question": "有没有关于架构图的内容"}'
# 应能检索到图片的描述文本

# Query Router 日志
curl -X POST http://localhost:8000/api/query -d '{"question": "\"精确短语\" 2024-01"}'
# 日志：bm25_weight=2.0, vec_weight=0.5
```

### Phase 7 验证（Tauri）
```bash
cd docflow-desktop && npm run tauri dev
# 验证：系统托盘、WebView、Python sidecar、查询、图片缩略图
npm run tauri build
# 在 clean Mac 上安装 .app，验证首次启动模型下载 + 完整功能
```

---

## 关键技术决策总结

| 问题 | 决策 | 理由 |
|------|------|------|
| Rust 重写？ | 不。Tauri 壳 + Python 后端 | ML 推理依赖 Python 生态 |
| 短文件跳过向量化？ | 不。统一 pipeline | embedding 极低成本，两条路径增加复杂度 |
| LLM Agent 路由？ | 不。规则 QueryRouter | 0ms vs 2000ms，reranker 已处理质量 |
| 多路径？ | `watch_dirs` 列表 + 运行时 CRUD | 支持 Obsidian / 任意目录 |
| 万级文档？ | **BM25 迁移 FTS5** + Qdrant payload 瘦身 | 现有 pickle BM25 在 100K+ chunks 会爆内存 |
| 图片入库？ | VLM 生成描述 → 走文本 pipeline | 统一向量空间，不需要 CLIP 第二集合 |
| 桌面客户端？ | Tauri v2 + Solid.js + LanceDB | 轻量、原生、无 Docker 依赖 |
