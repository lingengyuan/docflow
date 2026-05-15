# DocFlow 自评分 — 2026-05（Phase 0 之后）

按 [`improvement-roadmap.md`](improvement-roadmap.md) 定义的四维口径自评。**任何 ≥90 都需要仓内可验证证据，本表只填实际可举证的分数。**

## 总览

| 维度 | Phase 0 前 | Phase 0 后 | Phase 1（部分）后 | 90+ 还差什么 | 修复 PR |
| --- | --- | --- | --- | --- | --- |
| 产品定位 / 差异化（Product） | 60 | 63 | **63** | 双向链接 / 标签 / 图谱 / 间隔重复 中至少 2 项；Obsidian 归属定型 | Phase 6, Phase 7 |
| 用户体验 / 前端（UX） | 58 | 58 | **58** | 前端框架化；`innerHTML` 拼接清零；⌘K + split-view + hover-preview；Playwright/a11y CI | Phase 4, Phase 5 |
| 检索 / 答题质量（Quality） | 68 | 68 | **68** | BEIR 或 C-MTEB 子集结果入仓；faithfulness 评测；10k 文档压测入 `docs/status.md` | Phase 2, Phase 3 |
| 工程 / 代码质量（Engineering） | 76 | 82 | **86** | 33 个白名单 mypy 错误清空；coverage 基线门禁；god module 拆分（999/728 → ≤800）；剧本测试 | Phase 1 剩余 |

**当前总均值：65.5（Phase 0 前） → 67.75（Phase 0 后） → 68.75（Phase 1 部分后）。**
仍然没有触动 UX 与 Quality 的实质问题——按反作弊原则，那两个维度只会在对应 Phase 落地时变动。

> 注：表中数字是**主观自评**，不代表外部独立审计结果。低于 90 的项都映射到具体后续 PR；只有该 PR 合并后才能调整对应分数。

## Phase 1（部分）后增量（Engineering 82 → 86，+4）

本 PR 落地了 Phase 1 五项中的三项（roadmap §Phase 1 第 1、2、4 项的"前半"）：

1. **mypy 全量入仓**：`tool.mypy.files = ["src", "main.py"]`，检查文件数 24 → 100；
   现存 33 个错误通过 `[[tool.mypy.overrides]] ignore_errors = true` 显式白名单。
   白名单是**只能缩小不能增长**的清单，构成下一批 PR 的具体 backlog。
2. **CI 收集 coverage**：`pytest-cov` 接入；Ubuntu/3.12 leg 跑 `--cov=src`，
   `coverage.xml` 作为 artifact 上传。当前基线 ≈ **75%**。
   按 ADR-0004，先 collect 一段时间再设 `--cov-fail-under`。
3. **模块大小预算**：`scripts/check_module_sizes.py` 默认 800 LOC，
   `browser_acceptance.py`(999) 与 `startup.py`(728) 进 grandfathered 上限。
   新增模块越界即 CI fail。

**为何只 +4 而不是 +8**：
- 33 个 mypy 错误用白名单挡住，没有真的修 → 不应当全额计分。
- coverage 只收集、没门禁 → 防退化能力为 0。
- god module 上限锁死了但没拆 → 改善的是**未来**，不是现在。

Engineering ≥ 90 仍需：白名单清空 + 设 `--cov-fail-under` + `browser_acceptance.py` / `startup.py` 实际拆分。


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

