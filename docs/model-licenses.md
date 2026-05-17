# Model Licenses

DocFlow source code is MIT licensed. Model weights are separate artifacts and keep
their own upstream license terms. Users and maintainers must review those upstream
terms before commercial use, redistribution, or packaging model weights with DocFlow.

## Configured Model Families

Default templates reference these model families:

| Purpose | Default reference | Where it is used |
| --- | --- | --- |
| Local answers through Ollama-compatible tools | `qwen2.5:7b`, `qwen3:8b` | Local chat backend and optional contextual prefixes |
| Embeddings | `Qwen/Qwen3-Embedding-0.6B` | Local vector search |
| Reranking | `Qwen/Qwen3-Reranker-0.6B` | Retrieval reranking |
| OCR | `glm-ocr` | Optional PDF/image text extraction through Ollama-compatible OCR |
| Image understanding | `mlx-community/Qwen3-VL-8B-Instruct-4bit` | Optional image parser |
| Apple Silicon MLX answers | `mlx-community/Qwen3-4B-4bit`, `mlx-community/Qwen3-8B-4bit` | Optional MLX local answer backend |
| Cloud answers | `claude-sonnet-4-6` | Optional Claude backend when explicitly configured |

## Packaging Policy

- The repository does not include model weights.
- Docker images and Python package artifacts do not bundle model weights.
- `privacy.allow_model_download: false` keeps DocFlow from silently fetching missing
  model files.
- A maintainer may document a recommended model, but must not imply that the MIT
  license for DocFlow also covers that model.
- If a future release ships a preconfigured model download path, the release notes
  must identify the model provider, license, expected size, and whether commercial
  use is allowed by the upstream terms.
