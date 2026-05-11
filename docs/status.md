# Project Status

DocFlow is now beyond the original prototype phase and has a clearer public project surface.

## Current State

- README is concise and points to focused public docs.
- New users can start from `config.example.yaml`, `docker-compose.yml`, and the `docflow` command.
- Docker Compose now defines both the DocFlow app and Qdrant service for first-run startup.
- First-run users can create a small demo library from the CLI or browser empty state.
- Runtime dependencies are smaller and optional image support is split out.
- CI, ruff, pre-commit, issue template, and PR template are present.
- CI now runs full ruff, mypy, and pytest on Ubuntu and macOS.
- Local privacy has an offline doctor check.
- Citations include chunk identity and source span metadata.
- Source preview can highlight the cited source range when citation span metadata is available.
- Local answer generation uses deterministic defaults.
- Retrieval evaluation reports Recall@5, MRR@5, nDCG@5, and pass rate.
- Parsing regression checks cover the committed Markdown and plain-text corpus.

## Latest Local Validation

- Unit/integration tests: 287 passed.
- Ruff: passed.
- Mypy: passed.
- Browser acceptance: 73 checks passed.
- Retrieval eval: 8/8 passed, Recall@5 1.0, MRR@5 1.0, nDCG@5 1.0.
- Parsing eval: 2/2 passed.
- Offline doctor from Phase44: 0 unexpected outbound connections on the startup health path.

## Remaining Gaps

- Model libraries can still perform their own cache/download checks outside DocFlow's `src.net` layer.
- Citation source opening carries chunk identity and span metadata, and source preview highlights the cited range when the matching chunk is available.
- Retrieval and parsing evaluation sets are still small and should grow before public release claims.
- Large-file and large-library benchmarks are not yet part of the standard CI path.
- Public API, storage, and retrieval entry points are now small facades, but their implementation modules still need deeper internal splitting before the codebase feels fully mature to outside contributors.
