# DocFlow × Obsidian 集成计划（v2）

> 修订于 2026-03-22
> 基于 [kepano/obsidian-skills](https://github.com/kepano/obsidian-skills) 重新设计
> 目标：DocFlow 做语义检索引擎，Obsidian CLI 做 vault 读写，各司其职。

---

## 核心思路

```
DocFlow 的角色：语义检索引擎（向量 + BM25 hybrid）
  → Obsidian CLI search 只做关键词匹配，找不到"概念相关"的内容
  → DocFlow 的独特价值是 embedding-based semantic search

Obsidian CLI 的角色：vault 读写操作
  → search / read / append / property:set / backlinks / tags
  → Claude 通过 obsidian-cli skill 已经知道怎么用

两者互补：DocFlow 语义找到笔记 → Claude 通过 CLI 读全文/写回
```

---

## 实施层级

| 优先级 | 改动 | 代码量 | 状态 |
|--------|------|--------|------|
| Tier 0 | config.yaml 加 vault 路径 | 0 行 | ✅ 已完成 |
| Tier 0.5 | 安装 obsidian-cli skill | 0 行代码 | 手动 |
| Tier 1 | MarkdownParser 文本清洗 | ~40 行 | 待实施 |
| Tier 1.5 | mtime 快跳 + watcher debounce | ~30 行 | 待实施 |

### ~~已砍掉的 Tier 2~~

原计划的 `source_type`、`obsidian_note_name`、`obsidian_tags` in citation 全部不做：
- `obsidian_note_name` = `file_name.removesuffix('.md')`，客户端一行推导
- `source_type` 可从 file_path 对 watch_dirs 做路径匹配推导，不需要 DB 字段
- tags in citation：Claude 需要 tags 时可 `obsidian tags` 或直接读文件
- 写回：Claude 已有 obsidian-cli skill，直接 `obsidian append file="note"` 即可

---

## Tier 0 — 已完成 ✅

`config.yaml` 已配置：

```yaml
paths:
  watch_dirs:
    - path: "~/Documents/DocFlow"
      recursive: false
    - path: "~/MyNotes/HughLin"
      recursive: true
      extensions: [".md"]
```

## Tier 0.5 — 安装 obsidian-cli skill

将 [obsidian-skills](https://github.com/kepano/obsidian-skills) 的 skill 文件放入项目 `.claude/` 目录，Claude 即获得 vault 读写能力。

## Tier 1 — MarkdownParser 文本清洗

**文件**：`src/ingest/parsers/markdown_parser.py`

按 `obsidian-markdown` skill 的精确语法规范，清洗以下内容：

| 语法 | 处理方式 |
|------|---------|
| YAML frontmatter (`---...---`) | strip from text，提取 `tags` 存 files 表 |
| `[[note]]` | → `note` |
| `[[note\|显示文字]]` | → `显示文字` |
| `![[embed]]` | strip 整行（嵌入内容不属于本文语义） |
| `%%隐藏注释%%` | strip |
| `> [!type] title` | 去 callout 标记，保留 title + 内容 |
| `#tag`（非行首 `# heading`） | 提取加入 tag 列表，原处保留文字 |
| `^block-id` | strip |

**数据流**：

```
MarkdownParser.parse()
├── 1. 分离 frontmatter（yaml.safe_load）→ 提取 tags/aliases
├── 2. 正文清洗：strip wikilinks, embeds, callouts, comments, block-ids
├── 3. return ParsedDocument(pages=[干净文本], metadata={"tags": [...]})
└── pipeline.ingest() 读 doc.metadata["tags"] → store.upsert_file(tags=...)
```

**关联改动**：
- `pdf_analyzer.py`：ParsedDocument 加 `metadata: dict = field(default_factory=dict)`
- `store.py`：files 表加 `tags TEXT DEFAULT '[]'` 列
- `pipeline.py`：传递 tags 到 store

## Tier 1.5 — 大 vault 性能优化

### mtime 快速跳过

`needs_ingest()` 当前对每个文件计算 SHA-256。大 vault（1000+ 文件）启动慢。

优化：先比较 mtime，没变则跳过 hash。

```python
# mtime 没变 → 跳过（大概率未修改）
if file_mtime_ns <= row["mtime_ns"]:
    return False
# mtime 变了才算 hash（防 touch 但内容没变）
return row["file_hash"] != self.compute_hash(file_path)
```

**关联改动**：files 表加 `mtime_ns INTEGER DEFAULT 0` 列。

### Watcher debounce

Obsidian 每次 Ctrl+S 触发文件变更事件。无去抖会导致连续编辑时重复 ingest。

在 `FileEventHandler._handle()` 加 3 秒去抖窗口：记录 `{path: last_event_time}`，只有距上次事件 ≥3 秒才触发 ingest。

---

## 不需要改动的部分

| 组件 | 原因 |
|------|------|
| `watcher.py` 核心逻辑 | watchdog 已支持 .md，只加 debounce |
| `chunker.py` | 按 token 分块，对清洗后的 .md 同样适用 |
| `embedder.py` | embedding 模型不需要知道文件来源 |
| `query/engine.py` | QueryRouter + HybridRetriever 不变 |
| `query/generator.py` | LLM 后端不变 |
| `src/api/app.py` | 无需加 Obsidian 专属 API |

---

## 集成后的工作流

```
Obsidian Vault (.md)
    │
    ├─ watchdog 自动监控（已有，带 debounce）
    ↓
docflow ingest
    ├─ MarkdownParser 清洗语法 + 提取 tags（Tier 1）
    ├─ mtime 快跳避免无谓 hash 计算（Tier 1.5）
    ├─ Chunker 分块 → Embedding → Qdrant + SQLite
    ↓
Claude 收到问题
    ├─ POST /api/query/stream → DocFlow 语义检索
    │    citation: {file_name: "my-note.md", ...}
    │
    ├─ obsidian read file="my-note"    ← CLI skill，读全文
    └─ obsidian append file="my-note"  ← CLI skill，写回
```
