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
