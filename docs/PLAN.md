# DocFlow 重构开发计划

> 日期：2026-03-21
> 优先级：后端 MLX 迁移 > 前端重构 > BM25 优化

---

## 现状 & 目标

**核心问题**：Qwen3-Reranker-0.6B 在 PyTorch MPS 上太慢（10 pairs = 10.45s），导致检索阶段 1-14s 不可接受。
**目标**：换用 MLX runtime，将 reranker 从瓶颈变成优势；同时按设计稿重构前端 UI。

| 指标 | 当前（PyTorch MPS） | 目标（MLX） |
|------|---------------------|------------|
| 检索耗时 (warm) | 0.89–14.00s | < 0.5s |
| 端到端 TTFT (warm) | 2.5–25s | 1–3s |
| 内存占用 | ~3GB | ~3GB（不变） |

---

## Phase 1：后端 MLX 迁移（最高优先级）

### 目标
用 MLX runtime 替换 PyTorch MPS，解决 reranker 慢的根本问题。

### 方案选型

推荐使用 **`embed-rerank`** PyPI 包（FastAPI 内嵌服务，支持 Qwen3-Embedding + Qwen3-Reranker）：

```bash
pip install embed-rerank
embed-rerank --model Qwen/Qwen3-Embedding-0.6B --rerank-model Qwen/Qwen3-Reranker-0.6B
# 启动后暴露本地 HTTP 接口：/embed 和 /rerank
```

备选：`mlx-embeddings`（仅 embedding，无 reranker）或 `qwen3-embeddings-mlx`（高吞吐 embedding server）。

### 任务分解

#### Task 1.1：集成 embed-rerank 服务

**文件**：`src/query/retriever.py`

当前：`Qwen3Reranker` 类直接加载 PyTorch HuggingFace 模型，在进程内推理。
目标：改为 HTTP client，调用本地 embed-rerank 服务。

```python
# 改动方向：retriever.py 中的 Qwen3Reranker 类
class MLXReranker:
    """通过 HTTP 调用本地 embed-rerank 服务"""
    def __init__(self, base_url: str = "http://localhost:8001"):
        self.base_url = base_url

    def compute_score(self, pairs: list[list[str]]) -> list[float]:
        # POST /rerank  {"query": "...", "documents": [...]}
        # 返回 scores 列表
        ...
```

**注意**：`embed-rerank` 的 rerank 接口格式为 `{"query": str, "documents": list[str]}`，而当前代码传 `[[query, doc], ...]`，需适配。

#### Task 1.2：集成 MLX Embedding

**文件**：`src/ingest/embedder.py`、`src/query/retriever.py`

当前：`SentenceTransformer(device="mps")` 在进程内推理。
目标：同样改为 HTTP client，调用 embed-rerank 的 `/embed` 接口。

```python
class MLXEmbedder:
    def __init__(self, base_url: str = "http://localhost:8001"):
        self.base_url = base_url

    def encode(self, texts: list[str], ...) -> np.ndarray:
        # POST /embed  {"input": [...]}
        # 返回 embeddings 列表
        ...
```

#### Task 1.3：config.yaml 新增 MLX 配置项

```yaml
mlx_service:
  base_url: "http://localhost:8001"   # embed-rerank 服务地址
  enabled: true                        # false = 回退到 PyTorch MPS

embedding:
  model: "Qwen/Qwen3-Embedding-0.6B"
  device: "mlx"   # 新增 mlx 选项，由 MLXEmbedder 处理

reranker:
  model: "Qwen/Qwen3-Reranker-0.6B"
  device: "mlx"   # 同上
```

#### Task 1.4：启动脚本

`scripts/start.sh`：
```bash
#!/bin/bash
# 先启动 MLX 推理服务，再启动 FastAPI
embed-rerank \
  --model Qwen/Qwen3-Embedding-0.6B \
  --rerank-model Qwen/Qwen3-Reranker-0.6B \
  --port 8001 &
sleep 3
python main.py serve
```

#### Task 1.5：warmup 改造

`app.py` 中 `_warmup_models()` 改为调用 HTTP health check，确认 embed-rerank 服务就绪。

### 回退方案

如果 embed-rerank 集成遇到问题，使用 **无 reranker + vec_score 阈值** 方案（已验证，检索 0.5s，性能可接受）：

```python
# retriever.py：将 reranker 设为 None，跳过精排
if self.reranker is not None:
    candidates = self._rerank(query, candidates)
```

config.yaml 中 `reranker.enabled: false` 即可。

---

## Phase 2：前端重构

### 目标
按 `docs/DESIGN.md` + `docs/code.html` 设计稿，重写 `frontend/index.html`。

### 设计稿关键特征

参见 `docs/code.html`（完整 HTML）和 `docs/screen.png`（效果图）：

**布局**：
- 左侧固定宽度 icon 导航栏（w-20），Chat / Library / History / Settings
- 主区域单列，max-width 720px，居中
- 无右侧 PDF 预览面板（设计稿也是单列布局）

**颜色系统**（已在 code.html 的 tailwind config 中定义完整 token）：
- Base: `#f7f9fb`（surface）
- Card: `#ffffff`（surface-container-lowest）
- Sidebar: `#f0f4f7`（surface-container-low）
- Primary: `#515f74`
- Text: `#2a3439`（on-surface）

**组件**：
- 用户消息：primary 背景，rounded-2xl rounded-tr-none
- AI 回复：surface-container-lowest/50 背景，prose 排版
- Citations：横向 pill 标签，hover 浮层显示原文 snippet
- 输入框：border-outline-variant/15，focus:border-primary/40，rounded-2xl
- 发送按钮：bg-primary，rounded-xl

### 任务分解

#### Task 2.1：从 code.html 迁移静态结构

1. 复制 `docs/code.html` 的完整 HTML 骨架到 `frontend/index.html`（保留 Tailwind CDN + Material Symbols + tailwind config）
2. 清空 code.html 中的 demo 内容（硬编码消息、外部图片 URL）
3. 添加 JS 动态区域锚点（`id="messages"`, `id="input"` 等）

#### Task 2.2：迁移 JS 业务逻辑

从当前 `frontend/index.html` 搬运以下逻辑到新模板：

| 功能 | 当前实现 | 新 UI 适配 |
|------|----------|-----------|
| SSE 流式查询 | `sendMessage()` fetch + ReadableStream | 不变，适配新 DOM 节点 |
| Citations 渲染 | 暗色样式 div | 改为 code.html 的 pill 样式 + hover snippet tooltip |
| Token 逐字追加 | `answerDiv.textContent +=` | 适配新 AI 回复容器，追加到 prose div |
| 历史记录面板 | 独立 section | 移到 History 视图（侧边栏切换） |
| 文件库面板 | 独立 section | 移到 Library 视图 |
| PDF 预览 | iframe embed | 点击 citation 后在新 tab 打开（简化） |
| LLM 切换 | 下拉 select | 改为 header 右侧的 pill selector（参考 code.html） |
| Upload | input[file] | 保留，挂到 Library 视图的上传按钮 |

#### Task 2.3：三个视图实现（侧边栏切换）

```
Chat 视图：当前对话 + 输入框（默认显示）
Library 视图：文件列表 + 上传按钮 + 状态标签
History 视图：查询历史列表（可点击展开引用）
```

每个视图对应一个 `<section id="view-chat/library/history">` div，JS 控制 hidden/block。
侧边栏按钮点击时切换 active 样式（bg-surface-container-lowest + shadow）。

#### Task 2.4：Citations 交互

按 code.html 样式（`group` + `group-hover:opacity-100` tooltip）：
- 每条 citation 是一个 group div，hover 显示 snippet tooltip
- 点击 citation → 新 tab 打开 `/api/file/{file_id}/preview`
- file_id 通过 `/api/files` 接口用 file_name 反查

#### Task 2.5：Streaming cursor

沿用 code.html 中的 `.streaming-cursor::after` CSS（闪烁竖线），流式输出时给最后段落加此 class，done 后移除。

### 注意事项

- **禁用 emoji**（DESIGN.md 规定），全部改用 Material Symbols 图标
- **无 1px border 分割线**，用背景色差区分区域
- **字体**：`-apple-system, "PingFang SC", "Inter", sans-serif`（tailwind config 已定义）
- Tailwind CDN：`https://cdn.tailwindcss.com?plugins=forms,container-queries`

---

## Phase 3：BM25 中文分词优化（低优先级）

**问题**：`text.lower().split()` 对中文无效，BM25 中文检索精度差。

**方案**：引入 `jieba` 分词：

```python
# embedder.py _rebuild_bm25()
import jieba
def _tokenize(text: str) -> list[str]:
    return list(jieba.cut(text.lower()))

tokenized = [_tokenize(t) for t in self._bm25_corpus]
```

**代价**：需重新 ingest 所有文件（BM25 索引格式变化）。建议在 Phase 1/2 完成后再做。

---

## 执行顺序

```
Phase 1（MLX 迁移）: ✅ 完成 2026-03-21
  - 删除 Qwen3Reranker（PyTorch MPS），替换为 MLXReranker（mlx-lm）
  - 实测：6 pairs rerank = 0.40s（原 10.45s），提速 26x
  - 检索总耗时：0.74s（原 1-14s）
  - 安装：.venv/bin/pip install mlx-lm

Phase 2（前端重构）: ✅ 完成 2026-03-21
  - 按 docs/code.html + docs/DESIGN.md 完整重写 frontend/index.html
  - 浅色主题、左侧 icon 导航栏、citation pill + hover tooltip
  - streaming cursor、Material Symbols 图标、LLM 下拉选择器

Phase 3（BM25 优化）: ✅ 完成 2026-03-21
  - pip install jieba，embedder + retriever 双侧改 _tokenize
  - 新增 IngestQueue on_done 回调，ingest 完成后热重载 BM25
  - 修复共享 Embedding 模型实例 + _ensure_collection 显式调用

Phase 4（MLX LLM）: 待实现
  目标：LLM 后端从 Ollama 迁移到 in-process MLX，TTFT 6–8s → 2–4s
```

---

## Phase 4：MLX LLM 后端（下一步）

### 目标

用 `mlx-lm` 在进程内运行 Qwen3-4B / Qwen3-8B，取代 Ollama HTTP 调用。

**为什么**：
- Ollama 上的 `qwen3:4b` 是 Thinking-only 特化版，思考模式无法关闭，流式 RAG 体验差（6–15s 无声等待）
- HuggingFace 原版 Qwen3 支持 `enable_thinking=False`，直接输出答案
- mlx-lm 已在进程内运行（Reranker 已用），无额外依赖
- 消除 Ollama HTTP 开销，TTFT 预估降至 2–4s

### 可用模型（已确认存在）

| 模型 | 大小 | 用途 |
|------|------|------|
| `mlx-community/Qwen3-4B-4bit` | 2.3GB | 日常默认 |
| `mlx-community/Qwen3-8B-4bit` | 4.6GB | 增强模式 |

### 任务分解

#### Task 4.1：generator.py 新增 mlx 后端

**正交性原则**：新后端作为独立分支，不改动现有 ollama/claude 后端。

```python
# generator.py 新增 MLXGenerator 类（或在 AnswerGenerator 内新增 backend="mlx"）
class AnswerGenerator:
    def __init__(self, backend: str = "local", ...):
        # backend: "local"（Ollama）| "mlx" | "claude"
        ...

    def _load_mlx_model(self):
        """懒加载 MLX LLM，复用 ml_executor 线程。"""
        from mlx_lm import load
        self._mlx_model, self._mlx_tokenizer = load(self.mlx_model_name)

    def _build_prompt_nothink(self, system: str, user: str) -> str:
        """构建 enable_thinking=False 的 prompt，注入空 think 块。"""
        messages = [{"role": "system", "content": system},
                    {"role": "user", "content": user}]
        return self._mlx_tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,  # 注入 <think>\n\n</think>\n\n 前缀
        )

    def _stream_mlx(self, system: str, user: str):
        """yield token strings via mlx_lm.stream_generate。"""
        from mlx_lm import stream_generate
        prompt = self._build_prompt_nothink(system, user)
        for response in stream_generate(self._mlx_model, self._mlx_tokenizer,
                                        prompt=prompt, max_tokens=2048):
            yield response.text
```

**ETC 原则**：保持 `generate()` / `generate_stream()` 公共接口不变，调用方无感知切换。

#### Task 4.2：config.yaml 新增 mlx 配置

```yaml
llm:
  backend: "mlx"                          # local | mlx | claude
  mlx_model: "mlx-community/Qwen3-4B-4bit"
  mlx_model_enhanced: "mlx-community/Qwen3-8B-4bit"
  ollama_model: "qwen2.5:7b"             # 保留作 fallback
  ollama_model_enhanced: "qwen3:8b"
```

**DRY 原则**：模型名集中在 config，不在代码里硬编码。

#### Task 4.3：app.py lifespan 集成

MLX LLM 需要在 `ml_executor` 线程内加载（与 Reranker 共享 Metal command queue）：

```python
# lifespan warmup 阶段加入 LLM 预热
await loop.run_in_executor(ml_executor, _warmup_models)
# _warmup_models 内加：
if query_engine.generator.backend == "mlx":
    query_engine.generator._load_mlx_model()
    query_engine.generator._mlx_model("warmup")  # dummy forward
```

所有 MLX LLM 推理也通过 ml_executor 提交，与 Reranker 串行，避免 Metal 并发问题。

#### Task 4.4：流式生成路径适配

`app.py` 的 `query_stream` 端点当前在 `ml_executor` 的普通线程里跑 `_run()`，里面调用 `query_engine.query_stream()`。MLX stream_generate 是同步生成器，直接在 `_run()` 里 yield 即可，无需额外改动。

#### Task 4.5：下载模型

```bash
# 后台下载（2.3GB，约 3–5 分钟）
.venv/bin/python -c "
from huggingface_hub import snapshot_download
snapshot_download('mlx-community/Qwen3-4B-4bit')
"
```

### 预期内存影响

```
现状（in-process）：
  MLX Reranker 0.6B     ~300MB Metal
  CPU Embedding 0.6B    ~600MB RAM
  Python Physical FP    ~30MB

Phase 4 后：
  MLX Reranker 0.6B     ~300MB Metal    （不变）
  CPU Embedding 0.6B    ~600MB RAM      （不变）
  MLX Qwen3-4B-4bit     ~2.3GB Metal   （新增）
  Python Physical FP    ~3GB            （+Metal shader 少量增加）
  Ollama                不再需要        （释放 ~5GB）
```

### 回退方案

`config.yaml` 改 `backend: "local"` 即可回退到 Ollama，不需要改代码。

## 关键文件速查

| 文件 | 说明 | Phase |
|------|------|-------|
| `src/query/retriever.py` | Qwen3Reranker → MLXReranker | 1 |
| `src/ingest/embedder.py` | SentenceTransformer → MLXEmbedder | 1 |
| `src/api/app.py` | warmup 逻辑改造 | 1 |
| `config.yaml` | 新增 mlx_service 配置块 | 1 |
| `scripts/start.sh` | 一键启动（新建） | 1 |
| `frontend/index.html` | 整体重写，参照 docs/code.html | 2 |
| `docs/code.html` | 新 UI 设计稿（完整可运行 HTML） | 参考 |
| `docs/DESIGN.md` | UI 设计规范（颜色、字体、禁忌） | 参考 |
| `docs/screen.png` | 最终视觉效果图 | 参考 |
| `docs/HANDOFF.md` | 项目现状交接文档 | 参考 |
