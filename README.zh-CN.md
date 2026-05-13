# DocFlow

[English](README.md) · 简体中文

---

## 简体中文

### 项目说明

DocFlow 会监听你机器上的本地文件夹，把资料写入本地索引，然后在浏览器工作台里回答问题并展示来源。它面向私人的个人知识管理：默认无遥测、本地存储，本地模型后端由你自己选择。

### 功能特性

- 支持基于 PDF、Markdown、DOCX、TXT、代码文本和可选图片内容提问。
- 可以查看来源片段、把有用回答保存为笔记，并查看主题、相似资料和知识卡片。
- 默认使用 SQLite 保存元数据、Qdrant 保存向量、本地模型处理问答。
- 提供测试、浏览器验收、检索评估、解析评估和离线网络检查来验证当前状态。

### 快速开始

```bash
git clone https://github.com/lingengyuan/docflow.git
cd docflow
docker compose up --build
```

打开 [http://localhost:8000](http://localhost:8000)，然后选择 **导入示例资料** 体验一个小型本地资料库。

### 当前验证结果

当前仓库最近一次本地验证结果：`309` 个测试通过，`73` 个浏览器检查通过，源文件过滤后的项目回归检索评估 `84/84`，解析评估 `31/31`，离线本地使用检查报告 `0` 个意外外连。

这些数字适合用来守住项目回归，不等同于大规模公开 benchmark。

### 隐私承诺

DocFlow 不包含遥测、分析统计、自动错误上报，也不会为了产品分析上传你的文档。运行时文档、SQLite 元数据、Qdrant 向量、备份和生成索引都保留在你的机器上，除非你明确启用网页导入、模型下载或云端模型这类外部功能。

运行离线检查：

```bash
docflow doctor --offline
```

### 项目结构

- `main.py`：命令入口。
- `src/api/`：浏览器应用和接口。
- `src/ingest/`：解析、切块、索引和存储。
- `src/query/`：检索、重排、引用和回答生成。
- `frontend/`：浏览器工作台。
- `docs/`：公开文档。
- `eval/`：已提交的检索和解析评估输入。

### 配置

DocFlow 首次运行会从 `config.example.yaml` 创建 `config.yaml`。监听目录、支持文件类型、SQLite 路径、Qdrant 连接、Embedding 模型、本地模型后端和隐私设置都在这里配置。

### 开发与测试

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e .
docker compose up -d qdrant
docflow demo --create-only
.venv/bin/python -m pytest -q
docflow eval retrieval --refresh-sources --source-filter --write-results
docflow eval parsing --write-results
docflow browser-acceptance
docflow doctor --offline
```

### 贡献指南

请阅读 [CONTRIBUTING.md](CONTRIBUTING.md)。保持改动聚焦，提交前运行测试，普通浏览器界面不要出现命令行或开发者专用文案。

### 维护指南

见 [功能说明](docs/features.md)、[架构说明](docs/architecture.md)、[隐私说明](docs/privacy.md)、[命令说明](docs/cli.md)、[开发说明](docs/development.md)、[评估说明](docs/evaluation.md)、[发布说明](docs/release.md)、[状态说明](docs/status.md) 和 [路线图](ROADMAP.md)。

### 许可证

MIT。见 [LICENSE](LICENSE)。
