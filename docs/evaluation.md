# Evaluation

DocFlow currently has several validation paths:

```bash
.venv/bin/python -m pytest
python main.py eval
python main.py maturity-eval
python main.py browser-acceptance
python main.py restore-drill
```

## Current Limitation

The maturity score is useful as an internal progress signal, but it is not enough as an external quality claim.

The 90-point roadmap replaces subjective maturity scoring with more measurable outputs:

- Retrieval metrics.
- Citation alignment.
- Parsing regression checks.
- Incremental indexing checks.
- Reproducibility checks.
- Offline privacy checks.

## Reproducibility

Local answer generation defaults to deterministic settings in `config.example.yaml`:

```yaml
query:
  seed: 42
  temperature: 0.0
```

Cloud LLM backends are reported as not reproducible because DocFlow cannot control their full serving environment.
