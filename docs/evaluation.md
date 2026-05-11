# Evaluation

DocFlow currently has several validation paths:

```bash
.venv/bin/python -m pytest
docflow eval retrieval --write-results
docflow eval parsing --write-results
docflow maturity-eval
docflow browser-acceptance
docflow restore-drill
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

## Retrieval Metrics

`docflow eval retrieval --write-results` runs `eval/qa_v1.jsonl` and reports:

- Recall@5
- MRR@5
- nDCG@5
- pass rate

Results are written under `eval/results/`, which is local runtime output and is not committed.

## Parsing Regression

`docflow eval parsing --write-results` checks the committed corpus in `eval/parsing_corpus/` against expectations in `eval/parsing_expected/`.

## Reproducibility

Local answer generation defaults to deterministic settings in `config.example.yaml`:

```yaml
query:
  seed: 42
  temperature: 0.0
```

Cloud LLM backends are reported as not reproducible because DocFlow cannot control their full serving environment.
