# DocFlow 优化路线图 — 走到全维度 ≥ 90 分

本文件配合 [`critique-2026-05.md`](critique-2026-05.md) 使用。批评列出**问题**，本文给出**解决顺序**与**评分定义**。

## 评分维度与口径

DocFlow 自评分按四个维度，各项满分 100，"优秀"门槛 = **每一维 ≥ 90**。维度定义如下，加分必须有**仓内可验证证据**支撑。

| 维度 | 90+ 的判据（举证形式） |
| --- | --- |
| **产品定位 / 差异化（Product）** | README 有正向叙事；"personal knowledge workspace" 配套至少含**双向链接 + 标签 + 图谱 + 间隔重复**四项之一以上；Obsidian 插件归属明确。 |
| **用户体验 / 前端（UX）** | 前端走组件框架（Preact / Lit / Svelte 任一）；`innerHTML` 字符串拼接 ≤ 5 处；有 ⌘K 命令面板、三栏 split-view、键盘流；Playwright 视觉回归 + axe-core a11y 进 CI；视觉参考与现状逐页 diff 入仓。 |
| **检索 / 答题质量（Quality）** | 至少一个外部公开基准（BEIR 子集 / C-MTEB 子集 / MIRACL 子集）的真实结果入仓 `eval/results/`；faithfulness 评测脚本与基线入仓；大库压测（≥10k 文档）的指标入仓 `docs/status.md`。 |
| **工程 / 代码质量（Engineering）** | mypy 覆盖 `src/` 全量（含必要 `# type: ignore` 白名单）；PEP 621 `optional-dependencies` 完整；ADR 至少 5 条；CI 含 coverage 基线门禁；端到端剧本测试入仓。 |

**当前自评：见 [`scoring-2026-05.md`](scoring-2026-05.md)。该文件每次有显著变更都要更新。**

---

## 分阶段路线

每一阶段对应一个 PR；除非另注，所有 PR 必须保持 CI 绿。

### Phase 0 — 本 PR（你正在 review 的这个）
**目标**：把"问题清单 / 路线图 / ADR / 评分"四份文档写入仓；把依赖管理从只读 `requirements.txt` 升级为 PEP 621 extras，但保留旧文件以兼容 CI。**不**触前端、**不**改 retrieval、**不**扩 mypy。

**预期评分变化**：
- Product：+5（差异化讨论显性化、Obsidian 归属问题显性化）
- Engineering：+5（PEP 621 extras、ADR 入仓）
- UX / Quality：0（本 PR 不动）

### Phase 1 — Engineering hardening（一个独立 PR）
1. mypy 覆盖扩到 `src/` 全量，必要处加 per-module override。
2. CI 增加 `pytest --cov=src --cov-report=xml`，先**只收集**不门禁；第二个 PR 设 `--cov-fail-under=<基线 - 2>`。
3. 把 `requirements-*.txt` 改为从 `pyproject.toml` 生成（或干脆删除，并在 CONTRIBUTING 里更新指引）。
4. 拆分明显过大的模块（依据 cyclomatic / LOC 阈值给出量化报告）。
5. 新增 ADR 0004 / 0005（依据下面 Phase 引入的决策）。

**目标分**：Engineering ≥ 90。

### Phase 2 — Eval honesty（一个独立 PR）
1. 接入 BEIR-SciFact 或 NFCorpus 子集；脚本入仓 `eval/external/scifact_runner.py`；结果入 `eval/results/<date>/scifact.json`；`docs/status.md` 引用 latest。
2. 接入 C-MTEB 中文子集（至少 1 个 retrieval 任务）。
3. 新增 faithfulness 评测（answer ↔ cited chunk 的句级一致性），入 `eval/faithfulness/`。
4. 把所有 eval 结果按日期归档入仓（不再是 CI 跑完即丢）。

**目标分**：Quality ≥ 75（仍未到 90，因为大库压测和外部基准还没全跑）。

### Phase 3 — Eval at scale（一个独立 PR）
1. 大库压测脚本：10k 合成文档（合成可重放）的索引时间、磁盘占用、查询 P95，结果入 `docs/status.md`。
2. 多 retrieval 基准的回归门禁。

**目标分**：Quality ≥ 90。

### Phase 4 — Frontend rebase（一个独立 PR，最大的一个）
1. 选定框架（推荐 Preact + signals，原因：体积小、SSR 友好、迁移路径短）。
2. 引入设计令牌（design tokens）：颜色、间距、字号、阴影、圆角。
3. 把当前 27 个 JS 模块按 4 个视图（Chat / Library / Notes / Settings）切成组件；逐视图迁移并保留 fallback；`innerHTML` 拼接清零（最多保留 ≤5 处带注释豁免点）。
4. 引入 Playwright 视觉回归 + axe-core a11y 检查，进 CI。

**目标分**：UX ≥ 75（还差 ⌘K / split-view / hover-preview / 图谱面板）。

### Phase 5 — Knowledge tooling baseline（一个独立 PR）
1. ⌘K 全局命令面板。
2. 三栏 split-view（侧栏 / 主区 / 引用面板）。
3. Hover-preview（引用 hover 弹出原文片段）。
4. 键盘优先：所有主要操作可纯键盘完成。

**目标分**：UX ≥ 90。

### Phase 6 — PIM minimal closure（一个独立 PR，决定产品分能不能上 90）
按 ADR-0004（待写）选 PIM 第一刀切哪里。建议**四选二**：
- 双向链接（[[wiki-link]]）+ 反向链接面板。
- 标签 / 标签树。
- 图谱视图。
- FSRS 间隔重复（答题保存为 note 后进入复习队列）。

**目标分**：Product ≥ 90。

### Phase 7 — Obsidian plugin 归属决策（轻 PR）
按 ADR-0003 落定结论：
- 选项 A：拆出独立 repo `docflow-obsidian`，主仓移除目录，README 加跨链。
- 选项 B：留在主仓但纳入 CI / 文档主线、版本号同步、明确写入 `docs/features.md`。

任一选项执行后定型，**不可两可**。

---

## 反作弊原则（写给自己 / 后续维护者）

1. **不接受"已规划"作为加分项。** 评分只看仓内可验证证据。
2. **不接受没跑过的基准。** 任何外部基准必须有 commit-in 的 `eval/results/<date>/…json`。
3. **不接受空壳组件。** Phase 4-6 任何"看起来像 ⌘K 但按了没反应"的占位代码不计分。
4. **每个 PR 结束更新 `scoring-2026-05.md`** —— 如果分没动就如实写没动。

