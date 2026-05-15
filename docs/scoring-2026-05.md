# DocFlow 自评分 — 2026-05（Phase 0 之后）

按 [`improvement-roadmap.md`](improvement-roadmap.md) 定义的四维口径自评。**任何 ≥90 都需要仓内可验证证据，本表只填实际可举证的分数。**

## 总览

| 维度 | Phase 0 前 | Phase 0 后（本 PR） | 90+ 还差什么 | 修复 PR |
| --- | --- | --- | --- | --- |
| 产品定位 / 差异化（Product） | 60 | **63** | 双向链接 / 标签 / 图谱 / 间隔重复 中至少 2 项；Obsidian 归属定型 | Phase 6, Phase 7 |
| 用户体验 / 前端（UX） | 58 | **58** | 前端框架化；`innerHTML` 拼接清零；⌘K + split-view + hover-preview；Playwright/a11y CI | Phase 4, Phase 5 |
| 检索 / 答题质量（Quality） | 68 | **68** | BEIR 或 C-MTEB 子集结果入仓；faithfulness 评测；10k 文档压测入 `docs/status.md` | Phase 2, Phase 3 |
| 工程 / 代码质量（Engineering） | 76 | **82** | mypy 全量；coverage 基线门禁；端到端剧本测试；模块拆分量化报告 | Phase 1 |

**当前总均值：67.75 → 67.75（Phase 0 主要修 Engineering & Product 的"可见性"，未修 UX / Quality 的实质问题）。**

> 注：表中数字是**主观自评**，不代表外部独立审计结果。低于 90 的项都映射到具体后续 PR；只有该 PR 合并后才能调整对应分数。

## 各维度详评（Phase 0 后）

### 产品定位 / 差异化 — **63**
**+3 增量（相对 Phase 0 前）**
- `docs/critique-2026-05.md` 把 "knowledge workspace" 的口号与现有能力的差距显性化。
- `docs/adr/0003-obsidian-plugin-scope.md` 把 Obsidian 插件归属问题立项（提案待决）。

**距 90 的硬缺口**
- README "Why DocFlow" 仍是减法叙事，本 PR 没有改写（避免一边写一边自夸而无新能力支撑）。
- PIM 最小闭环（双链 / 标签 / 图谱 / 间隔重复）一个都还没做。
- Obsidian 插件归属仍未定型。

### 用户体验 / 前端 — **58**
**无变化**：本 PR 不动前端。

**距 90 的硬缺口**
- 27 个手写 JS、4404 行、98 处 `innerHTML`。
- 无 ⌘K、无 split-view、无 hover-preview、无键盘流。
- 无视觉回归 / a11y CI。

### 检索 / 答题质量 — **68**
**无变化**：本 PR 不动 retrieval / eval。

**距 90 的硬缺口**
- 公开基准（BEIR / MTEB / C-MTEB / MIRACL）一个都没跑。
- 没有答案 faithfulness 评测。
- 没有 ≥10k 文档的压测数字。

### 工程 / 代码质量 — **82**
**+6 增量**
- `pyproject.toml` 增加 PEP 621 `[project.optional-dependencies]`（`dev` / `vision` / `mlx`），并保留旧 `requirements-*.txt` 作兼容路径。`pip install docflow[vision]` 成立。
- 引入 ADR 目录与三条 ADR（0001 模块边界、0002 本地优先与无遥测、0003 Obsidian 归属提案）。
- 问题清单 + 路线图 + 评分三份文档进仓，建立"以仓内文档驱动迭代"的工作面。

**距 90 的硬缺口**
- mypy 仍只覆盖 4 个路径，未扩 `src/` 全量（Phase 1）。
- CI 无 coverage 门禁（Phase 1）。
- 模块拆分量化报告未做（Phase 1）。
- 端到端剧本测试缺失（Phase 1）。

## 反作弊声明

- 本评分表所有数字均**基于本 PR 真实落地的文件与证据**，不计已规划项。
- 后续每个 Phase PR 必须在合并前更新此文件，且**只能在该 PR 内提供的证据范围内**改分。
- 若发现哪怕一项分数没有对应可验证证据，应立即降回。

