# DocFlow 交接文档

> 日期：2026-03-20
> 状态：性能优化阶段完成，待前端重构 + 后端 MLX 迁移

---

## 1. 项目概览

DocFlow 是一个本地部署的 PDF 知识助手。用户将 PDF 放入 `~/Documents/DocFlow/` 文件夹，系统自动解析、分块、向量化，用户通过 Web 界面提问，系统检索相关片段并用 LLM 生成答案。

**技术栈**：Python 3.12 + FastAPI + Qdrant (本地) + Ollama + sentence-transformers + HuggingFace transformers

**运行环境**：M5 Mac Air, 32GB RAM + 1TB SSD

---

## 2. 当前架构

```
用户提问 (浏览器)
    │
    ▼
FastAPI (/api/query/stream)  ← SSE 流式响应
    │
    ├── Retrieval 阶段（~1-11s，取决于 reranker 候选数）
    │   ├── Qwen3-Embedding-0.6B (MPS) → 查询向量编码 ~0.3-0.5s
    │   ├── Qdrant 向量检索 top-20 ~0.02s
    │   ├── BM25 关键词检索 top-20 ~0.01s
    │   ├── RRF 融合 + 向量分数阈值过滤 (0.4)
    │   ├── 去重
    │   └── Qwen3-Reranker-0.6B (MPS) 精排 → top-5  ← 当前瓶颈
    │
    └── Generation 阶段（~1-15s，取决于模型冷热和答案长度）
        └── Ollama qwen2.5:7b → 流式 token 输出
```

### 文件结构

```
docflow/
├── config.yaml              # 全局配置（模型、路径、参数）
├── main.py                  # CLI 入口（serve / ingest / scan）
├── docflow.db               # SQLite 元数据（files/chunks/history/favorites）
├── bm25_index.pkl           # BM25 索引
├── qdrant_storage/          # Qdrant 本地存储（1024 维向量）
├── src/
│   ├── api/app.py           # FastAPI 路由 + SSE 流式端点
│   ├── query/
│   │   ├── engine.py        # QueryEngine（串联 retriever + generator）
│   │   ├── retriever.py     # HybridRetriever + Qwen3Reranker
│   │   └── generator.py     # AnswerGenerator（Ollama 流式 + Claude API）
│   └── ingest/
│       ├── pipeline.py      # IngestPipeline（PDF → chunks → vectors）
│       ├── pdf_analyzer.py  # PDF 解析（原生 + OCR）
│       ├── chunker.py       # 结构化分块（512 tokens, 10% overlap）
│       ├── embedder.py      # Embedding + Qdrant + BM25 写入
│       ├── store.py         # SQLite 操作层
│       ├── queue.py         # 异步 ingest 队列
│       └── watcher.py       # 文件夹监控
├── frontend/
│   └── index.html           # 当前前端（待重构）
└── docs/
    ├── DESIGN.md             # 新 UI 设计规范（Ethereal Canvas）
    ├── code.html             # 新 UI 设计稿 HTML
    └── screen.png            # 新 UI 设计效果图
```

---

## 3. 本次优化历程

### 3.1 第一轮：根治内存 swap（几分钟 → 12-17s）

**问题**：三个大模型 (4B+4B+7B ≈ 21GB) 挤爆 32GB 内存，触发 swap。

**改动**：
- Embedding: `Qwen3-Embedding-4B` → `BAAI/bge-m3` (568M)
- Reranker: `Qwen3-Reranker-4B` → 完全移除
- Embedding 设备: CPU → MPS

**结果**：模型内存从 ~21GB 降到 ~6.5GB，检索从分钟级降到 <1.5s。

### 3.2 第二轮：体验优化（12-17s → 感知秒级）

**问题**：①同一 PDF 的不同页面重复出现在引用中 ②同步等待 Ollama 完整生成

**改动**：
- 添加 SSE 流式端点 (`/api/query/stream`)，前端逐 token 显示
- 添加向量相似度阈值 (0.4)，过滤不相关文档
- 引用按文件去重，每个文件只保留得分最高的一条

**结果**：引用 0.5s 出现，首 token ~1s（模型热），流式输出体验接近秒级。

### 3.3 第三轮：模型升级评估（进行中）

**尝试**：切换到 `Qwen3-Embedding-0.6B` + `Qwen3-Reranker-0.6B` 全家桶

**发现**：
- Qwen3-Embedding-0.6B 表现优秀（embed ~0.3s，中文质量比 bge-m3 好）
- Qwen3-Reranker-0.6B 质量最好（CMTEB-R 71.31）但在 PyTorch MPS 上极慢
  - 1 pair: 0.59s
  - 10 pairs: 10.45s
  - 20 pairs: 13.66s
- 根因：生成式 reranker 每对 pair 需完整 causal LM 前向传播，MPS 不擅长

**结论**：模型选择正确，推理框架是瓶颈。MLX 可以比 PyTorch MPS 快 200 倍。

---

## 4. 当前配置状态

**config.yaml**：
```yaml
embedding:
  model: "Qwen/Qwen3-Embedding-0.6B"  # 当前生效
  batch_size: 32
  device: "mps"

reranker:
  model: "Qwen/Qwen3-Reranker-0.6B"   # 当前生效，但慢（见上）
  instruction: "Given a search query, retrieve relevant text passages that answer the query."

ollama:
  llm_model: "qwen2.5:7b"              # 默认
  llm_model_enhanced: "qwen3.5:27b"    # 可通过 UI 切换（需 ~17GB 内存）
```

**Qdrant collection**：`docflow`，1024 维，COSINE 距离，当前 52 个向量点
**已索引文件**：4 个 PDF，52 个 chunks

---

## 5. 当前性能数据

### Retrieval 阶段（模型热状态）

| 阶段 | Qwen3-0.6B 全家桶 | bge-m3 无 reranker |
|------|-------------------|-------------------|
| Embedding | 0.28-0.43s | 0.40-0.50s |
| Vector search | 0.01-0.03s | 0.01-0.02s |
| BM25 search | 0.00-0.01s | 0.00-0.01s |
| Reranker | 0.59-13.66s (取决于 pairs 数) | 0s |
| **总计** | 0.89-14.00s | 0.45-0.58s |

### 端到端（模型热状态）

| 指标 | 当前（Qwen3 全家桶） | bge-m3 方案 |
|------|---------------------|-------------|
| Citations 出现 | 0.89-14.00s | 0.14-0.80s |
| 首 token (TTFT) | 2.50-25.24s | 1.11-7.74s |
| 总完成 | 13.84-30.01s | 3.59-17.10s |

### 基准对比

| 方案 | 检索 (warm) | 端到端 (warm) | 内存 |
|------|------------|--------------|------|
| 原始 4B+4B | 6-8s + swap | 几分钟 | ~21GB |
| bge-m3 无 reranker | **0.5s** | **3-5s** | ~1.5GB |
| Qwen3-0.6B 全家桶 (MPS) | 1-14s | 3-30s | ~3GB |
| Qwen3-0.6B 全家桶 (MLX) | **预估 <0.5s** | **预估 2-4s** | ~3GB |

---

## 6. 已有 API 端点

| 端点 | 方法 | 说明 |
|------|------|------|
| `/api/query` | POST | 同步查询（返回完整 JSON） |
| `/api/query/stream` | POST | SSE 流式查询（citations → tokens → done） |
| `/api/ingest` | POST | 触发全量扫描 |
| `/api/queue` | GET | Ingest 队列状态 |
| `/api/files` | GET | 文件列表 |
| `/api/file/{id}/preview` | GET | PDF 原文预览 |
| `/api/upload` | POST | 上传 PDF |
| `/api/history` | GET/DELETE | 查询历史 |
| `/api/favorites` | GET | 收藏列表 |
| `/api/favorites/{id}` | POST | 切换收藏 |
| `/api/summarize` | POST | 批量摘要生成 |
| `/api/llm` | GET/POST | 查看/切换 LLM 模型 |
| `/api/health` | GET | 健康检查 |

### SSE 流式协议

```
event: citations
data: [{"file_name": "...", "page_num": 1, "snippet": "...", "score": 0.03}]

event: token
data: "根据"

event: token
data: "文档"

...

event: done
data: ""
```

---

## 7. 已缓存的模型

```
~/.cache/huggingface/hub/
├── models--Qwen--Qwen3-Embedding-0.6B    (584M)
├── models--Qwen--Qwen3-Embedding-4B      (7.5G)  ← 已弃用
├── models--Qwen--Qwen3-Reranker-0.6B     (999M)
├── models--Qwen--Qwen3-Reranker-4B       (7.5G)  ← 已弃用
└── models--BAAI--bge-m3                   (已缓存)

Ollama models:
├── qwen2.5:7b      (4.7GB)
├── qwen3.5:27b     (17GB)
└── glm-ocr         (2.2GB)
```

---

## 8. 已知问题

1. **Reranker 在 PyTorch MPS 上太慢**：生成式 reranker 10-20 pairs 需要 10-14s，不可接受
2. **Metal shader 首次编译**：首次查询 embed 和 rerank 各额外耗时 ~5s（一次性）
3. **BM25 对中文分词粗糙**：当前用 `text.lower().split()` 做 tokenize，对中文效果差
4. **前端需要重构**：当前 index.html 是暗色主题的快速原型，需按 DESIGN.md 重做
5. **无错误重试机制**：Ollama / Qdrant 连接断开时无优雅降级
