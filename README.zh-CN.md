# DocFlow

[English](README.md) · 简体中文

---

## 简体中文

### 项目说明

DocFlow 会监听你机器上的本地文件夹，把资料写入本地索引，然后在浏览器工作台里回答问题并展示来源。它面向私人的个人知识管理：默认无遥测、本地存储，本地模型后端由你自己选择。

### 功能特性

- 支持基于 PDF、Markdown、DOCX、TXT、代码文本和可选图片内容提问。
- 可以查看来源片段、把有用回答保存为笔记，并查看主题、相似资料、知识卡片、活跃概念和问题到来源的轨迹。
- 默认使用 SQLite 保存元数据、Qdrant 保存向量、本地模型处理问答。
- 提供测试、浏览器验收、检索评估、解析评估和离线网络检查来验证当前状态。

### 快速开始

```bash
git clone https://github.com/lingengyuan/docflow.git
cd docflow
docker compose -f docker-compose.image.yml up
```

打开 [http://localhost:8000](http://localhost:8000)，然后选择 **导入示例资料** 体验一个小型本地资料库。
README 中的截图来自内置示例资料库，不包含个人文件夹或真实笔记内容。

首次运行成本需要提前知道：Docker 和 Qdrant 会占用本地空间，当前验证机器上应用镜像约 0.5 GB；你选择的本地问答模型会额外占用空间，常见 7B Ollama 模型通常约 4-5 GB。图片理解和 Apple Silicon MLX 支持都是可选安装。

### 当前验证结果

当前仓库最近一次本地验证结果：`478` 个测试通过，`82` 个浏览器检查通过，公开可复现检索评估 `547/547`，BEIR SciFact-lite 外部子集 Recall@5 `0.95`，BEIR NFCorpus-lite 外部子集 Recall@5 `0.30`，源文件过滤后的项目回归检索评估 `84/84`，解析评估 `120/120`，回答可信度检查 `14/14`，1 万份合成资料的本地查找基准通过，性能冒烟检查和发布门面检查通过，离线本地使用检查报告 `0` 个意外外连。

这些数字适合用来守住项目回归；BEIR-lite 是外部子集结果，不等同于大规模公开 benchmark，也不等同于完整公开排行榜成绩。

### 隐私承诺

DocFlow 不包含遥测、分析统计、自动错误上报，也不会为了产品分析上传你的文档。运行时文档、SQLite 元数据、Qdrant 向量、备份和生成索引都保留在你的机器上，除非你明确启用网页导入、模型下载或云端模型这类外部功能。

运行离线检查：

```bash
docflow doctor --offline
```

公开镜像启动默认使用 `docker-compose.image.yml` 拉取 `ghcr.io/lingengyuan/docflow:edge`，避免每次本地构建。源码开发时再使用 `docker compose up --build`。

### 项目结构

- `main.py`：命令入口。
- `src/api/`：浏览器应用和接口。
- `src/ingest/`：解析、切块、索引和存储。
- `src/query/`：检索、重排、引用和回答生成。
- `frontend/`：浏览器工作台。
- `docs/`：公开文档。
- `eval/`：已提交的检索和解析评估输入。

### 配置

DocFlow 首次运行会从 `config.example.yaml` 创建 `config.yaml`。监听目录、支持文件类型、SQLite 路径、Qdrant 连接、Embedding 模型、本地模型后端、隐私设置和回答质量阈值都在这里配置。

### 开发与测试

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e .
docker compose up -d qdrant
docflow demo --create-only
scripts/run_ci.sh
docflow doctor --offline
```

评估和发布验证命令见 [评估说明](docs/evaluation.md) 与 [发布说明](docs/release.md)。

### 贡献指南

请阅读 [CONTRIBUTING.md](CONTRIBUTING.md)。保持改动聚焦，提交前运行测试，普通浏览器界面不要出现命令行或开发者专用文案。

### 维护指南

见 [功能说明](docs/features.md)、[架构说明](docs/architecture.md)、[隐私说明](docs/privacy.md)、[威胁模型](docs/threat-model.md)、[模型许可](docs/model-licenses.md)、[命令说明](docs/cli.md)、[开发说明](docs/development.md)、[评估说明](docs/evaluation.md)、[发布说明](docs/release.md)、[状态说明](docs/status.md) 和 [路线图](ROADMAP.md)。

### 许可证

MIT。见 [LICENSE](LICENSE)。
