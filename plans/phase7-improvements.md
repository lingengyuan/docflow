# DocFlow Phase 7 改造计划

> 日期：2026-03-22
> 基于 insight-collector 对知识库交叉分析的输出

---

## 背景

Phase 6 完成后，cross-reference CodeSnippets 知识库（fts5_fuzzy_search.py × session_tracker.py × zvec_inprocess_vector.py）发现了 4 个高价值改造点，按收益/成本比排序如下。

---

## P1：FTS5 Trigram 降级（OCR 容错）⭐ 最高优先级

### 问题

当前 FTS5 层只支持精确 jieba token 匹配（`"tok1" OR "tok2"`）。OCR 错字（识别错误的字符）、简繁混用会导致 FTS5 层零结果，完全丢失相关 chunks。

### 方案

在 `chunks_fts`（精确 unicode61 分词）旁新增 `chunks_fts_trigram`（trigram 分词），`_fts_search()` 精确层返回空时自动降级到 trigram 子串匹配。

```
查询降级链：
  jieba 精确 FTS5（chunks_fts）→ 若空 → trigram 子串 FTS5（chunks_fts_trigram）
```

### 涉及文件

| 文件 | 修改内容 |
|------|---------|
| `src/ingest/store.py` | `_init_db()` 新增 `chunks_fts_trigram` 表；`add_chunks()` 同步写入；新增 `search_fts_trigram()` |
| `src/query/retriever.py` | `_fts_search()` 内精确层失败后调用 `store.search_fts_trigram()` 降级 |

### 预计代码量

~70 行

---

## P2：语言检测 + 分词路由

### 问题

英文文档（技术论文、README）被 jieba 切分后，FTS5 索引质量极差（英文单词被拆成字符碎片）。当前 supported_extensions 包含 `.md`/`.txt`，这些文件大量为英文。

### 方案

ingest 时检测文档语言：
- **中文主导**（CJK 字符占比 > 20%）→ jieba 分词（现有逻辑）
- **英文主导** → 直接空格分词，利用 FTS5 unicode61 tokenizer 的英文处理能力

语言检测用纯 Python（检查 Unicode 范围），无额外依赖。

### 涉及文件

| 文件 | 修改内容 |
|------|---------|
| `src/ingest/pipeline.py` | `_fts_tokenize()` 新增语言检测逻辑 |

### 预计代码量

~15 行

---

## P3：查询历史 FTS5 搜索

### 问题

`history` 表已存在（question + answer），但没有全文检索能力。用户想找回"上次问过的关于 XX 的问题"只能靠时间排序翻页。

### 方案

新增 `history_fts` FTS5 表（question 字段），`add_history()` 时同步写入，新增 `GET /api/history/search?q=` 端点。

### 涉及文件

| 文件 | 修改内容 |
|------|---------|
| `src/ingest/store.py` | `_init_db()` 新增 `history_fts` 表；`add_history()` 同步写入；新增 `search_history()` |
| `src/api/app.py` | 新增 `GET /api/history/search` 端点 |

### 预计代码量

~40 行

---

## P4：zvec 向量后端（消除 Docker 依赖）

### 问题

Qdrant 需要 Docker（`docker compose up`），提高了部署门槛。对于分享给他人使用的场景，Docker 是最大障碍。

### 方案

抽象 `VectorBackend` Protocol，实现 `QdrantBackend`（现有逻辑封装）和 `ZvecBackend`（in-process）。`config.yaml` 新增 `vector_backend: qdrant | zvec` 选项。

> ⚠️ P4 是架构级改动，不在本次改造范围内，单独排期。

---

## 实施顺序

```
P1（FTS5 trigram）→ P2（语言检测）→ P3（历史搜索）
```

三个改造互相独立，无依赖，可按顺序逐一完成。

---

## 验证方法

| 改造点 | 验证方式 |
|-------|---------|
| P1 | 索引一个含 OCR 错字的 PDF，搜索其中一个错字的正确形式，应能通过 trigram 层命中 |
| P2 | 索引一个英文 MD 文件，搜索其中的英文单词，FTS5 层应有结果 |
| P3 | 提问两次后，`GET /api/history/search?q=关键词` 应返回相关历史记录 |
