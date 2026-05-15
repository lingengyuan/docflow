# DocFlow 自我批评清单 — 2026-05

本文是对 DocFlow（v0.58.0）当前状态的一次刻意尖锐的自我审视，目标读者是后续要把项目推到"全维度优秀（≥90 分）"的维护者。**评分体系与改进顺序见 [`improvement-roadmap.md`](improvement-roadmap.md)，本次 PR 之后的诚实评分见 [`scoring-2026-05.md`](scoring-2026-05.md)。**

格式：每条问题写明 **现象 → 证据 → 影响 → 严重度**。严重度 P0/P1/P2 对应"必须立刻修 / 下一两个 PR 修 / 见缝插针修"。

---

## 1. 产品定位与差异化（Product / Positioning）

### 1.1 定位描述偏防守、缺正向叙事
- **现象**：README "Why DocFlow" 用 "smaller than X / simpler than Y / not Z" 的对比句，三条都是减法。
- **证据**：`README.md` "vs AnythingLLM — smaller…", "vs Khoj — simpler…", "vs rolling your own LangChain — ships with…"。
- **影响**：读者看完不知道 DocFlow **必须存在**的理由是什么；对比的三个对象本身定位不重叠（产品级 chat、个人 AI 副驾、库），用同一句"我比它小"无法说服任何一类用户。
- **严重度**：P0。

### 1.2 "personal knowledge workspace" 的口号没有配套能力
- **现象**：README 自称 "knowledge workspace"，但只有 Q&A + saved-answer + library 视图，没有 PKM 类工具最低限度的双向链接、标签、图谱、Daily Note、block-level 引用、间隔重复。
- **证据**：`src/` 与 `frontend/js/` 下无 graph / backlinks / tags / spaced-repetition 模块；`docs/features.md` 列举的能力均围绕"问答 + 引用 + 笔记保存"。
- **影响**：与 Obsidian、Logseq、Reflect、Mem 等真正的 PKM 产品形成明确能力差距；"workspace" 当前等于"带笔记本的搜索框"。
- **严重度**：P0。

### 1.3 `obsidian-plugin/` 与主仓共存，但定位、版本、所有权全未声明
- **现象**：仓内存在 `obsidian-plugin/docflow-assistant/`（4 个文件 ~32KB），但 README、ROADMAP、CHANGELOG、`docs/features.md` 全未提及它的状态。
- **证据**：`find obsidian-plugin -type f` 返回 `manifest.json` / `main.js` / `styles.css` / `README.md`，主 README 无引用。
- **影响**：要么"DocFlow 是 Obsidian 插件"（那 PKM 借 Obsidian 即可），要么"DocFlow 是独立产品"（那这个插件应该拆出去）。两种立场不能同时占。读者无法判断主线在哪。
- **严重度**：P1（决策待 ADR-0003）。

---

## 2. 用户体验 / 前端（UI / UX）

### 2.1 前端是 27 个手写 JS 模块、98 处 `innerHTML`，仍然是 demo-grade
- **现象**：`frontend/js/` 下 27 个 `.js` 文件、4404 行手写无框架 JS，全文 98 处 `innerHTML` 字符串拼接。
- **证据**：`find frontend/js -name "*.js" | wc -l` = 27；累计 4404 行；`grep -rn innerHTML frontend/js/ | wc -l` = 98。
- **影响**：（a）`innerHTML` + 字符串拼接是 XSS 高风险面，即便目前数据是本地的，未来"webpage import"接入第三方 HTML 时就会出问题；（b）无组件复用机制，每次状态变更靠手工 diff DOM，UI 难以演进；（c）任何想做的 ⌘K 命令面板、split-view、图谱面板都会在这个架构上摔跤。
- **严重度**：P0（UI 维度想达到 90+ 的硬性前提）。

### 2.2 缺少现代知识工具的"心智总线"
- **现象**：无 ⌘K / Cmd-K 全局命令面板，无键盘优先操作，无三栏 split-view，无"侧边引用浮窗"，无 hover-preview。
- **证据**：`frontend/js/` 与 `frontend/partials/app.html` 无 command-palette / kbar / split-view 相关结构。
- **影响**：相比 Obsidian / Linear / Raycast / Notion，DocFlow 是"看着像但用着不顺"的工具——能完成任务，但不会让用户进入心流。
- **严重度**：P0。

### 2.3 视觉一致性与无障碍未被持续验证
- **现象**：项目原则要求"UI 必须像成品、与保存的参考视觉一致"，但没有 axe-core / Playwright visual-diff CI。
- **证据**：`.github/workflows/ci.yml` 有 `browser-acceptance` 步骤，但没有视觉回归 / a11y 检查。
- **影响**：每次前端改动都靠人眼判断，回归无报警。
- **严重度**：P1。

---

## 3. 检索 / 答题质量（Retrieval & Answer Quality）

### 3.1 公开数据集结果不入仓
- **现象**：`eval/public_retrieval_v1.jsonl` 是自建公共域语料（547 case），没有 BEIR / MTEB / C-MTEB / MIRACL / RAG benchmark 等业界公认数据集的结果，且每次跑的结果不入仓——只在 CI 里跑一次。
- **证据**：`docs/evaluation.md` 提到 "public-domain retrieval smoke" 但未引用任何外部基准；仓内无 `eval/results/` 历史快照。
- **影响**：用户无法对照"DocFlow 在 SciFact / NQ / 中文 C-MTEB 上是什么水平"。"Measured checks" 这句宣称在外部读者那里**没有可验证的锚点**。
- **严重度**：P0（与"no masking fallbacks"原则正面冲突）。

### 3.2 没有公开的大库压测
- **现象**：README 强调本地优先，但没有给出"10k 文档 / 1M chunk / 索引磁盘占用 / 查询 P95 延迟"的实测数字。
- **证据**：`docs/status.md` 引用了性能 smoke，但量级是"几十个 demo 文档"。
- **影响**：评估者不知道在 MacBook / 一般 Linux 工作站上跑真正用户级语料是否成立。
- **严重度**：P1。

### 3.3 答题路径缺"事实性 / 引用一致性"自动评测
- **现象**：有 retrieval 评测，没有针对**最终答案**的 faithfulness / citation-correctness 自动评测（即"答案是否真出自被引段落"）。
- **证据**：`eval/` 下无 faithfulness 数据集，`src/quality/` 无对应模块。
- **影响**：RAG 类产品最常见的失败模式（幻觉、错引）没有 CI 兜底。
- **严重度**：P1。

---

## 4. 工程质量（Engineering）

### 4.1 mypy 覆盖范围仅 4 个路径
- **现象**：`pyproject.toml [tool.mypy] files = ["src/api/schemas.py", "src/api/runtime.py", "src/api/services", "src/query"]`。
- **证据**：同上。`src/` 下有 99 个 `.py` 文件，mypy 仅强制覆盖约 1/3。
- **影响**：`src/ingest/`、`src/maintenance/`、`src/quality/`、`src/embedding_backend.py`、`src/vector_store.py` 等热区无类型约束。
- **严重度**：P1。

### 4.2 依赖管理仍走 `requirements*.txt`，未提供 PEP 621 extras
- **现象**：`requirements.txt` / `requirements-dev.txt` / `requirements-mac-mlx.txt` / `requirements-vision.txt` 四个文件互相 `-r`；`pyproject.toml` 只 `dynamic = ["dependencies"]` 引一个。
- **证据**：`pyproject.toml` line 13；四个 txt 文件并存。
- **影响**：（a）用户装额外能力必须知道哪个 txt；（b）`pip install docflow[vision]` 这种符合 PEP 621 习惯的入口不可用；（c）`pip install -e .` 拿不到 dev 工具。
- **严重度**：P1。

### 4.3 没有 ADR / 决策记录
- **现象**：`docs/architecture.md` 描述当前结构，但"为什么 SQLite + Qdrant 而不是 PG+pgvector"、"为什么本地优先且不接受 SaaS fallback"、"为什么 Python 全栈而不引 Rust ingest"这些**决策**没有记录。
- **证据**：`docs/` 下无 `adr/` 目录。
- **影响**：新贡献者会反复提同样的 PR / issue；维护者每次都要口头解释。
- **严重度**：P1。

### 4.4 CI 没有覆盖率门禁
- **现象**：CI 有 `pytest` 但无 `--cov`，也没有 coverage 上限/下限。
- **证据**：`.github/workflows/ci.yml`。
- **影响**：测试可以悄悄变薄，无人察觉。
- **严重度**：P2（先收集基线，再设门）。

---

## 5. 代码质量（Code Quality）

### 5.1 前端字符串拼接 HTML 等价于"硬编码模板"
- **现象**：同 §2.1。98 处 `innerHTML`，文本与 HTML 在 JS 里互相穿插。
- **影响**：可读性、可测试性、安全面都差。
- **严重度**：P0。

### 5.2 Python 侧模块边界**总体清晰**，但部分大文件值得拆分
- **现象**：从 `src/` 99 个 py 文件 + 55 个测试看，模块化做得不差；但 `src/api/services` / `src/maintenance/` 中存在文件较大、职责偏多的迹象（具体在下次 refactor PR 量化）。
- **严重度**：P2。

### 5.3 测试组织以单元测试为主，缺端到端剧本测试
- **现象**：`tests/` 55 个文件，但 README 自称的"browser acceptance" 在 CI 中作为独立步骤跑，仓内无 Playwright 剧本式 e2e（如"导入示例 → 提问 → 保存笔记 → 复习"完整闭环）。
- **严重度**：P1。

---

## 6. 文档与对外沟通（Docs / Communication）

### 6.1 ROADMAP 与本批评的"差距"未对齐
- **现象**：`ROADMAP.md` 反映已完成里程碑，但没有"我们已知差什么 / 什么时候补"的前向视图。
- **影响**：外部读者只能看到已经做完的部分，没法判断这个项目的雄心。
- **严重度**：P1（本 PR 通过新增 `improvement-roadmap.md` 部分缓解）。

### 6.2 文档对"评测局限"的诚实度可以更高
- **现象**：`docs/evaluation.md` 已经分离了 public/internal eval，但没有明确说"我们的评测仅覆盖：smoke 级别公开语料 + 84 case 内部 retrieval；不覆盖 BEIR/MTEB/外部 RAG 基准"。
- **影响**：读者会高估覆盖度。
- **严重度**：P2。

---

## 与项目原则的冲突点（必须修，否则违反自家规则）

项目 `AGENTS.md` 已经写明几条强约束：

1. **"no masking fallbacks: fallback behavior must not hide failures, data loss, stale data, or reduced answer quality."**  
   ↔ §3.1（评测结果不入仓 = 隐藏了"我们没在 BEIR 上跑过"这一事实）。这一条必须最先修。

2. **"User-facing UI must feel like a finished personal knowledge product, not a developer console."**  
   ↔ §2.1 / §2.2 / §5.1（27 个 vanilla JS、innerHTML 拼接、无命令面板）= 工程实现层面尚未达到这条目标所需的支撑能力。

3. **"Unit tests are required for behavior changes." + "End-to-end testing is required after feature work."**  
   ↔ §5.3（缺剧本式 e2e）部分违反。

---

## 修复执行约束

- 任何"修复"都不应破坏既有 CI（4 个矩阵 × 多步骤），改动需逐步。
- 任何"评分提升"必须用**仓内可验证证据**支撑（数据入仓 / 截图入仓 / 工作流入仓），不接受"已规划"或"即将上线"作为加分项。
- 本 PR 修的只是 P0 中**可在单 PR 内安全落地**的部分；其它在 [`improvement-roadmap.md`](improvement-roadmap.md) 中分 PR 列出。

