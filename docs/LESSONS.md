# DocFlow 踩坑记录

> 记录开发过程中遇到的非显而易见的坑，可作为文章材料。

---

## 坑 1：Apple Silicon 上 Metal Shader 编译缓存导致内存爆炸

### 现象

DocFlow 后端（Python FastAPI）在 Activity Monitor 里显示占用 **21.5GB 内存**，看起来像内存泄漏。

### 排查过程

```bash
# 查进程真实内存
vmmap <PID> | grep "Physical footprint"
# → Physical footprint: 21.5G

ps -o rss= -p <PID>
# → 73792 (≈ 72 MB)
```

RSS 只有 72MB，但 Physical Footprint 是 21.5GB。两者相差 300 倍。

进一步排查：
```bash
vmmap <PID> | grep "IOAccelerator" | wc -l
# → 9014
```

进程里有 **9014 个 Metal IOAccelerator 内存区段**。这是 GPU kernel 编译缓存。

### 根本原因

Apple Silicon 是统一内存架构（UMA），CPU 和 GPU 共享同一块物理内存。`Physical Footprint` 包含了进程在 GPU 侧分配的所有统一内存，不是"Python 在吃 21.5GB RAM"。

问题的具体来源：DocFlow 同时加载了两套 Metal 运行时：
- **PyTorch MPS**（用于 sentence-transformers Embedding 模型）
- **MLX**（用于 Qwen3 Reranker）

两套运行时各自进行 Metal shader JIT 编译，产生大量 GPU kernel 缓存，叠加后占满了统一内存的 GPU 侧。

### 解法

把 Embedding 模型从 MPS 改到 CPU，只保留 MLX 一套 Metal 运行时：

```yaml
# config.yaml
embedding:
  device: "cpu"   # 原来是 "mps"
```

Embedding 每次 encode 耗时从 0.1s 增加到约 0.3s，完全可接受。Python Physical Footprint 从 21.5GB 降至约 3–5GB。

### 教训

- `Physical Footprint ≠ RSS`：在 Apple Silicon 上，凡是用了 Metal 的库（PyTorch MPS、MLX、CoreML），都会在 GPU 统一内存侧分配大量 shader 缓存，这些都被计入 Physical Footprint。
- 多套 Metal 运行时共存会放大这个问题——每套都有自己的 shader 编译缓存，不会复用。
- **监控方法**：用 `vmmap` 而不是 Activity Monitor，区分 RSS（进程私有内存）和 Physical Footprint（含 GPU 统一内存）。

---

## 坑 2：MPS Metal Shader JIT 跨实例非确定性导致向量空间不一致

### 现象

上传 PDF 后检索，所有文档的余弦相似度全部为 0.0，查询返回空结果。

### 排查过程

```python
# 直接查 Qdrant 里存储的向量
records = client.scroll(collection_name='docflow', limit=5, with_vectors=True)
for r in records:
    v = np.array(r.vector)
    print(np.linalg.norm(v))  # → 0.0000, 0.0000, 0.0000 ...
```

向量全是零向量。但单独测试 `SentenceTransformer.encode()` 是正常的（norm ≈ 1.0）。

进一步排查发现：Qdrant collection 不存在，upsert 返回 404，异常被 `except Exception` 吞掉，文件却被标记为 `done`。

### 根本原因

两个嵌套的 bug：

**Bug A（直接原因）**：`_ensure_collection()` 只在 `model` 属性懒加载时调用。当我们绕过懒加载、直接给 `embedder._model` 赋值时，`_ensure_collection` 从未执行，Qdrant collection 不存在，upsert 静默失败。

```python
# 问题代码（app.py）
pipeline.embedder._model = shared_embed  # 绕过了 lazy loader
# → _ensure_collection 未被调用！
# → 后续 upsert → 404 → 异常被吞 → 文件状态却写成 "done"
```

**Bug B（深层原因，即使 collection 存在时）**：Ingest pipeline 和 query retriever 各自加载了独立的 `SentenceTransformer` 实例。Apple MPS 在每次 session 启动时 JIT 编译 Metal shader，不同实例可能编译出不同的 float 精度变体，导致同一段文字编码出的向量在数值上完全不同，余弦相似度趋近于 0。

### 解法

```python
# app.py lifespan：warmup 后，让 ingest pipeline 复用 retriever 已加载的模型实例
shared_embed = query_engine.retriever._embed_model
if shared_embed is not None:
    pipeline.embedder._model = shared_embed
    pipeline.embedder._vector_dim = shared_embed.get_sentence_embedding_dimension()
    # 显式补充调用，因为绕过了 lazy loader
    pipeline.embedder._ensure_collection(pipeline.embedder._vector_dim)
```

同时，所有 MPS 推理（embed + rerank + ingest embedding）都通过同一个 `ThreadPoolExecutor(max_workers=1)` 串行执行，确保共享同一个 Metal command queue。

### 教训

- **共享模型实例**：同一进程内，ingest 和 query 必须使用同一个 `SentenceTransformer` 对象，不能各自 `load()`。
- **绕过懒加载要谨慎**：直接给私有属性赋值等于跳过了初始化逻辑，要手动补上被跳过的步骤。
- **异常不要静默吞掉**：`except Exception: pass` 加上 `status="done"` 是定时炸弹，数据损坏无任何提示。
- **切换到 CPU device 可以从根本上消除跨实例 shader 非确定性问题**（见坑 1 的解法）。

---

## 坑 3：BM25 中文分词缺失导致中文检索召回率极差

### 现象

对中文文档提问，向量检索返回相关结果，但 BM25 路径返回 0 结果。RRF 融合后排名也受影响。

### 根本原因

BM25 的分词用的是 `text.lower().split()`，对英文（空格分隔）有效，对中文完全无效：

```python
"Zig 算法实现".split()  # → ["Zig", "算法实现"]  # "算法实现"没拆开
"比较算法效率".split()  # → ["比较算法效率"]       # 整句变一个 token
```

查询时 "算法" 无法匹配语料中的 "算法实现"，BM25 得分全为 0。

### 解法

引入 `jieba` 分词：

```python
import jieba

def _tokenize(text: str) -> list[str]:
    return [t for t in jieba.cut(text.lower()) if t.strip()]

# "Zig 算法实现" → ["zig", " ", "算法", "实现"]（过滤空白后）
# "比较算法效率" → ["比较", "算法", "效率"]
```

需要同时修改 **embedder**（建索引时）和 **retriever**（查询时），两侧分词逻辑必须一致，否则 BM25 匹配仍然为零。

修改后需要**重新 ingest 所有文档**，BM25 索引格式发生变化。

### 教训

- BM25 的质量上限由分词质量决定。对中英混合文档，英文空格分词加中文 jieba 分词缺一不可。
- 索引和查询的分词逻辑必须严格一致，任何一侧改了都要重新 ingest。
- 引入 jieba 的代价：启动时有约 0.5s 的词典加载时间（可接受）。

---

## 坑 4：`embed-rerank` PyPI 包根本不存在

### 现象

开发计划文档（PLAN.md）里写了用 `pip install embed-rerank` 来启动本地 MLX 推理服务，结果 PyPI 上不存在这个包。

### 解法

直接在进程内用 `mlx-lm` 加载 MLX 模型：

```python
from mlx_lm import load
model, tokenizer = load("Qwen/Qwen3-Reranker-0.6B")
```

不需要独立的 HTTP 服务，减少一层网络开销，延迟更低，部署更简单。

### 教训

- AI 生成的方案文档里提到的第三方工具，一定要先验证存在再写进 Roadmap。
- 在进程内直接加载 MLX 模型比启动独立服务更简单，对于单机场景没有理由多一层 HTTP。

---

## 坑 5：`mlx-embeddings` 不支持 Qwen3-Embedding 文字模型

### 现象

尝试用 `mlx-embeddings` 包来加速 Embedding，安装后调用报错：

```
AttributeError: Qwen2Tokenizer has no attribute batch_encode_plus
```

### 根本原因

`mlx-embeddings` 针对的是图像/多模态 embedding 模型，对纯文字 embedding 模型（如 Qwen3-Embedding）的 tokenizer 接口不兼容。

### 解法

文字 Embedding 继续用 `sentence-transformers`，不换。`mlx-lm` 只用来跑 Reranker（生成式模型）。两个库各司其职：

| 任务 | 库 | 设备 |
|------|----|----- |
| 文字 Embedding | sentence-transformers | CPU（见坑 1） |
| 生成式 Reranker | mlx-lm | MLX/Metal |
| LLM 生成回答 | Ollama | Ollama 管理 |

---

## 坑 6：Ollama 的 qwen3:4b/8b 是 Thinking-only 特化版，无法关闭思考模式

### 现象

将 LLM 从 `qwen2.5:7b` 切换到 `qwen3:4b` 后，流式查询返回引用（1.3s）后完全卡住，长时间没有任何 token 输出。

### 排查过程

```bash
# 查 Ollama chat API 原始响应
curl http://localhost:11434/api/chat \
  -d '{"model":"qwen3:4b","messages":[{"role":"user","content":"hello"}],"stream":true}'
# → 所有 token 出现在 "thinking" 字段，"content" 字段全部为空

# 查模型元信息
curl http://localhost:11434/api/show -d '{"name":"qwen3:4b"}' | jq .model_info
# → general.finetune: "Thinking"
# → general.name: "Qwen3-4B-Thinking-2507"
```

`qwen3:4b` 实际上是 `Qwen3-4B-Thinking-2507`，一个针对思考模式专门微调的变体。

### 根本原因

Ollama 打包时选用了 **Thinking 特化版**，而非原版 Qwen3-4B Instruct（支持软切换 thinking/non-thinking）。区别：

| 版本 | 类型 | 思考开关 |
|------|------|---------|
| Qwen3-4B (HuggingFace 原版) | Hybrid instruct | 支持 `/no_think` 关闭 |
| Qwen3-4B-Thinking-2507 (Ollama qwen3:4b) | Thinking-only | 思考模式烤进权重，无法关闭 |

尝试的绕过方式均无效：
- `options: {"think": false}` → 思考内容仍然生成，只是改变了输出字段路由
- 自定义 Modelfile 去掉 `<think>` 模板前缀 → 模型仍然在 content 里输出思考文本
- `raw` 模式 + 手动注入 `<think>\n\n</think>` 空思考块 → 无效

### 对流式 RAG 的影响

qwen3:4b thinking 模式下的时间线：
```
t=0   用户提问
t=1.3s  检索完成，引用返回
t=1.3s–8s  模型思考中（200–500 thinking tokens），前端无任何输出
t=8s  第一个 content token 出现，开始流式输出
```

这比 `qwen2.5:7b`（TTFT 2s，立即流出）体验更差。

### 当前解法

流式 RAG 继续使用 `qwen2.5:7b`，qwen3 thinking 模型留作非流式复杂查询备用（增强模式）。

### 潜在改进方向

在前端显示 thinking 进度（"思考中… 已生成 N 个 token"），让用户看到模型在工作，降低等待感。这是一个有意思的 UX 功能点，也能作为文章素材。

### 教训

- `ollama pull` 时不一定拿到原版 HuggingFace 模型，可能是经过 Ollama 团队二次选择的特化版本。
- 模型行为在权重层面改变时，任何 prompt 工程或 API 参数都无效。
- 验证方法：`curl http://localhost:11434/api/show -d '{"name":"MODEL"}' | jq .model_info` 检查 `general.finetune` 字段。

---

## 坑 7：transformers 升级到 v5 改变了 Qwen3-Embedding 的 prompt 行为

### 现象

安装 `mlx-lm` 时，`transformers` 被顺带升级从 4.57.6 → 5.3.0。之后 Qwen3-Embedding 模型的 `default_prompt_name` 变为 `None`，不再自动附加 query instruction 前缀。

### 影响

需要手动在查询向量化时加 instruction 前缀：

```python
# retriever.py：encode query 时显式加前缀
instructed_query = f"Instruct: {QUERY_INSTRUCTION}\nQuery: {query}"
query_vec = self.embed_model.encode([instructed_query], ...)

# embedder.py：encode document 时不加前缀（Qwen3-Embedding 规范）
dense_vecs = self.model.encode(batch_texts, ...)
```

查询侧加前缀、文档侧不加前缀，这是 Qwen3-Embedding 的设计规范，升级前后逻辑不变，但需要确认没有被 transformers v5 的默认行为改掉。

### 教训

- 安装任何包之前先检查它的依赖会不会升级关键库（`pip install X --dry-run`）。
- `transformers` 大版本升级是高风险操作，embedding 模型的 prompt 行为、tokenizer 接口都可能静默变化。

---

## 坑 8：前端 `accept=".pdf"` 硬编码导致多格式支持形同虚设

### 现象

后端 Phase 5 已支持 PDF / MD / TXT / DOCX / 图片全格式解析和入库，但用户在前端只能上传 PDF，其他格式无论拖拽还是点击都无法选择。

### 根本原因

前端 `index.html` 有 3 处硬编码：

```html
<!-- 两个 file-input 的 accept 属性 -->
<input type="file" accept=".pdf" ...>

<!-- handleDrop 过滤器 -->
await uploadFiles([...e.dataTransfer.files].filter(f => f.name.endsWith('.pdf')));

<!-- filter chip 名称处理 -->
btn.textContent = f.file_name.replace(/\.pdf$/i, '');
```

后端 `/api/upload` 从一开始就支持所有格式，卡点完全在前端。

### 解法

```html
<input type="file" accept=".pdf,.md,.txt,.docx,.jpg,.jpeg,.png,.webp,.heic,.heif" ...>
```

`handleDrop` 改为检查 `SUPPORTED` 数组，filter chip 名称处理扩展到所有已知后缀。

### 教训

- 后端加新功能时，前端的入口点（`accept` 属性、拖拽过滤器、UI 文案）要同步更新，否则功能对用户不可见。
- 前端 `accept` 属性只影响文件选择器的默认过滤，不影响后端安全性，不要过度依赖它做格式校验。

---

## 坑 9：`needs_ingest()` 不处理 `processing` 状态导致中断后文件永久跳过

### 现象

服务在 ingest 过程中被强制中断（kill/crash），重启后这些文件状态为 `processing`，但 `needs_ingest()` 返回 `False`，导致文件被永久跳过，知识库数据残缺。

```python
# 问题代码
def needs_ingest(self, file_path):
    ...
    if row["status"] == "error":
        return True   # error 重试
    return False      # processing 被漏掉了 → 变成"永远跳过"
```

### 根本原因

`needs_ingest()` 只处理了 `error` 状态，没有把 `processing` 也视为"需要重新入库"。而 ingest 开始时状态就会被设成 `processing`，中断后就永久卡住。

### 解法

```python
if row["status"] in ("error", "processing"):
    return True
```

### 教训

- 持久化状态机的每个中间状态都要考虑"进程崩溃后如何恢复"。`processing` 不等于"正在处理中"，也可能是"上次没处理完"。
- 只有 `done` 才是真正的终态，其他所有非终态在重启时都应该重新处理。

---

## 坑 10：PyTorch CPU 默认只用 4 线程，M5 的 10 核没有充分利用

### 现象

31 个 chunks 的 CPU embedding 耗时 **6 分钟**（约 12s/chunk），与 M5 的 CPU 性能完全不符。

### 排查过程

```python
import torch
print(torch.get_num_threads())   # → 4（只用了 4 核）
print(os.cpu_count())            # → 10（实际有 10 核）
```

PyTorch 在 macOS 上默认只使用 4 个线程，不自动探测可用核心数。

### 解法

在模型懒加载时显式设置：

```python
import os, torch
torch.set_num_threads(os.cpu_count() or 4)
```

设置后实测：**1.23s/chunk**，提速约 10x。

### 教训

- PyTorch CPU 线程数不会自动适配硬件，必须手动设置。
- 性能测试时先用 `torch.get_num_threads()` 确认当前线程数，再考虑其他优化方向。
- 只适用于 CPU embedding，不涉及 MLX/Metal，设置后不影响统一内存安全性（见坑 1）。

---

## 坑 11：Obsidian inline tag 提取的三个误匹配陷阱

### 现象

MarkdownParser 清洗 Obsidian 语法时，`#tag` 提取逻辑将以下内容误识别为 inline tag：

```
docflow-plan.md  tags=['一竞品现状与核心问题', '二技术选型调研结论', ..., 'page']
vibe-kb-plan.md  tags=['1-项目定位', ..., 'F7F5F0', '7A9E87']
```

### 根本原因

三类误匹配：

| 误匹配内容 | 来源 | 为什么 regex 命中 |
|-----------|------|-----------------|
| `(#一竞品现状与核心问题)` | Markdown 目录锚链接 `[text](#anchor)` | `(` 不是 `\w`，negative lookbehind `(?<!\w)` 不拦 |
| `#F7F5F0` | CSS/hex 颜色码 | `F` 是字母，regex 无法区分 tag 和颜色 |
| `#page=1` → 匹配出 `pag` | PDF URL fragment `#page=N` | regex `*` 贪婪匹配失败后**回溯**，截短到 `pag` 绕过 lookahead |

第三个尤其隐蔽：用 `(?![=)])` negative lookahead 试图排除 `#page=1`，但 regex 引擎的回溯机制使 `#page` 缩短为 `#pag`（`e` 后面不是 `=`，通过了 lookahead）。Python regex 没有 possessive quantifier `*+` 来阻止回溯。

### 解法

不依赖 regex lookahead，改用三步后过滤：

```python
# 1. 清除锚链接：先把 (#anchor) 替换掉再提取
cleaned = _MD_ANCHOR_RE.sub("", text)

# 2. 排除 hex 颜色码
if _HEX_COLOR_RE.match(tag):  # 纯 hex 3-8 位
    continue

# 3. 排除 URL fragment：检查匹配结束位置的下一个字符
if end < len(line) and line[end] == "=":
    continue
```

### 教训

- Regex negative lookahead + 贪婪量词 = 回溯陷阱。当引擎无法匹配完整 pattern 时，会缩短量词匹配长度重试，产生意想不到的部分匹配。
- 对于"匹配X但排除Y"的需求，先清除Y再匹配X（预处理）比在 regex 内部做排除更可靠。
- Obsidian tag 的合法字符集（字母/数字/下划线/连字符/斜杠）与 markdown 锚链接、hex 颜色码、URL fragment 高度重叠，纯 regex 无法安全区分。

---

## 坑 12：IngestQueue 使用 `ml_executor` 导致 PyTorch 线程清理触发 executor shutdown

### 现象

服务启动后，第一个大文件 embedding 完成，后续文件 `ml_executor.submit()` 抛出：

```
RuntimeError: cannot schedule new futures after shutdown
```

后续所有待入库文件全部失败，知识库入库中途中止。

### 根本原因

`ml_executor` 是 `ThreadPoolExecutor(max_workers=1)`，原本设计用于 MLX/Metal 推理（reranker、LLM），确保 Metal command queue 串行。

`IngestQueue` 把 ingest 任务（含 CPU embedding）也提交给了这个 executor。当 `torch.set_num_threads(10)` 开启多线程 embedding 后，PyTorch 在 embedding 完成时清理 OpenMP worker threads，这触发了 Python 内部的"解释器关闭"检测逻辑，导致 `ml_executor` 被标记为 `_shutdown=True`，后续 `submit()` 全部失败。

### 解法

CPU embedding 不需要 Metal executor，ingest 直接在 `IngestQueue` 自己的 worker 线程里跑：

```python
# app.py
ingest_queue = IngestQueue(
    pipeline,
    on_done=None,
    ml_executor=None,   # CPU embedding 不走 MLX executor
)
```

`ml_executor` 只保留给 MLX 推理（reranker、LLM 生成）。

### 教训

- `ThreadPoolExecutor` 的职责要单一：MLX Metal 推理用一个，CPU 计算用另一个（或直接用普通线程）。把不同 runtime 的任务混进同一个 executor，任何一方的清理行为都可能影响另一方。
- 增加 CPU 并行度（如 `torch.set_num_threads`）后，要重新审视线程安全性和 executor 的 shutdown 时序。
