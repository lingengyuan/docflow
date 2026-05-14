# DocFlow — Drop a folder, ask anything, never leave your laptop

English · [简体中文](README.zh-CN.md)

![DocFlow chat workspace](docs/assets/chat.png)

The screenshots in this README are captured from the bundled demo library, not a personal vault.

DocFlow is a local-first document Q&A and knowledge workspace. Point it at a folder of PDFs, Markdown, DOCX, code, or images. Ask questions in your browser. Get answers with cited sources.

- **Auditable local defaults.** No telemetry, analytics, or document upload. Optional webpage import, model downloads, and cloud backends are explicit.
- **Measured checks.** Current local checks include a 50-case public-domain smoke eval, 84 source-filtered internal retrieval cases, 31 parsing fixtures, 381 tests, and 81 browser checks.
- **Drop-in local models.** Works with Ollama, LM Studio, or any OpenAI-compatible local endpoint.

Quick start:

```bash
git clone https://github.com/lingengyuan/docflow.git && cd docflow
docker compose up --build
```

Then open <http://localhost:8000> and click the demo-library card, **导入示例资料**,
to try a small local library. The browser UI is Chinese-first today; the language
button switches common navigation and status labels while fuller localization continues.

For real answers, run a local model server such as Ollama or LM Studio and select it in Settings. The Docker path starts the app and Qdrant; model weights are still managed by the local model tool you choose.

Expected first-run cost: Docker plus Qdrant, about 0.5 GB for the base app image after build on the current validation machine, plus local model weights you choose. A 7B Ollama model is usually 4-5 GB. Image understanding and Apple Silicon MLX support are optional installs.

## Why DocFlow

- **vs AnythingLLM** — smaller, no account system, no SaaS fallback paths.
- **vs Khoj** — simpler stack (SQLite + Qdrant), focused on personal documents not chat agents.
- **vs rolling your own LangChain** — ships with repeatable tests, browser checks, parsing fixtures, and retrieval regression inputs.

## How it works

```mermaid
flowchart LR
    A[Watched folders] --> B[Parse + Chunk]
    B --> C[(SQLite<br>metadata)]
    B --> D[(Qdrant<br>vectors)]
    C --> E[Retrieve + Rerank]
    D --> E
    E --> F[Local LLM<br>Ollama / LM Studio]
    F --> G[Browser UI<br>with citations]
```

## Features

- Ask questions across PDFs, Markdown, DOCX, TXT, code-like files, and optional image content.
- Review source snippets, save useful answers as notes, and inspect topics, similar files, and knowledge cards.
- Metadata in SQLite, vectors in Qdrant, model traffic local by default when a local backend is selected.
- Verify behavior end-to-end: unit tests, browser acceptance checks, retrieval eval, parsing eval, and an offline network check.

## Privacy

DocFlow ships with zero telemetry, zero analytics, zero automatic error reporting, and zero product-analytics document upload. Your documents, SQLite metadata, Qdrant vectors, backups, and indexes stay on your machine unless you explicitly enable an external feature.

```bash
docflow doctor --offline
```

…checks the covered local startup, ingest, query, model-status, and source-preview paths for unexpected outbound connections.

## Project structure

- `main.py` — command entry point
- `src/api/` — browser app and HTTP API
- `src/ingest/` — parsing, chunking, indexing, storage
- `src/query/` — retrieval, reranking, citations, answer generation
- `frontend/` — browser workspace
- `docs/` — public documentation
- `eval/` — committed retrieval and parsing eval inputs

## Configuration

DocFlow generates `config.yaml` from `config.example.yaml` on first run. Configure watched folders, supported extensions, SQLite path, Qdrant connection, embedding model, local model backend, and privacy settings there.

## Development

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e .
docker compose up -d qdrant
docflow demo --create-only
.venv/bin/python -m pytest -q
docflow eval public --write-results
docflow eval retrieval --refresh-sources --source-filter --write-results
docflow eval parsing --write-results
docflow eval performance --write-results
docflow browser-acceptance
docflow doctor --offline
```

Release builds also support `docker-compose.image.yml` with `ghcr.io/lingengyuan/docflow` after a tagged image is published.

## Contributing

Read [CONTRIBUTING.md](CONTRIBUTING.md). Keep changes focused, run tests before opening a PR, and keep the normal browser UI free of command-line or developer-only wording.

## Documentation

[Features](docs/features.md) · [Architecture](docs/architecture.md) · [Privacy](docs/privacy.md) · [CLI](docs/cli.md) · [Development](docs/development.md) · [Evaluation](docs/evaluation.md) · [Release](docs/release.md) · [Status](docs/status.md)

## License

MIT. See [LICENSE](LICENSE).
