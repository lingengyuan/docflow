# Evaluation

DocFlow currently has several validation paths:

```bash
.venv/bin/python -m pytest
docflow eval retrieval --refresh-sources --source-filter --write-results
docflow eval parsing --write-results
docflow maturity-eval --no-rerank --source-filter
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

`docflow eval retrieval --refresh-sources --source-filter --write-results` refreshes the expected public source files, runs `eval/qa_v1.jsonl`, and reports:

- Recall@5
- MRR@5
- nDCG@5
- pass rate
- retrieval latency P50/P95/max

Results are written under `eval/results/`, which is local runtime output and is not committed.

Current committed retrieval set: 84 cases across README, public docs, contribution docs, security policy, roadmap, and changelog. Latest source-filtered local run: 84/84 passed, Recall@5 1.0, MRR@5 1.0, nDCG@5 1.0, retrieval P50 297.48 ms and P95 793.96 ms after model warmup.

## Parsing Regression

`docflow eval parsing --write-results` checks the committed corpus in `eval/parsing_corpus/` against expectations in `eval/parsing_expected/`.

Current committed parsing set: 31 documents covering Markdown tables, Obsidian-flavored Markdown, TXT, mixed-language notes, code-like files, native PDFs, and DOCX. Latest local run: 31/31 passed, 42 chunks checked, 11,351 text characters checked.

## Incremental Indexing

`tests/test_incremental_index.py` covers the add, modify, and delete cycle with deterministic local fakes. It confirms changed files replace old vectors and deleted files are cleaned from SQLite metadata within the five-second regression limit.

## Maturity Report

`docflow maturity-eval --no-rerank --source-filter` combines the dimension scorecard with retrieval and parsing signals. The dimension scores remain a planning view; the measured signals are the part to use for external quality claims.

## Reproducibility

Local answer generation defaults to deterministic settings in `config.example.yaml`:

```yaml
query:
  seed: 42
  temperature: 0.0
```

Cloud LLM backends are reported as not reproducible because DocFlow cannot control their full serving environment.
