# Evaluation

DocFlow currently has several validation paths:

```bash
.venv/bin/python -m pytest
docflow eval public --write-results
docflow eval retrieval --refresh-sources --source-filter --write-results
docflow eval parsing --write-results
docflow browser-acceptance
docflow restore-drill
```

## Current Limitation

DocFlow uses measured checks as external quality evidence:

- Retrieval metrics.
- Citation alignment.
- Parsing regression checks.
- Incremental indexing checks.
- Reproducibility checks.
- Offline privacy checks.

The public-domain smoke set below is reproducible from committed files, but it is intentionally small. It is not a BEIR, MTEB, or C-MTEB score and should not be treated as a broad external benchmark.

## Public Reproducible Smoke Benchmark

`docflow eval public --write-results` refreshes the committed public-domain corpus in `eval/public_corpus/`, runs `eval/public_retrieval_v1.jsonl`, and reports:

- Recall@5
- MRR@5
- nDCG@5
- pass rate
- retrieval latency P50/P95/max

The public smoke set currently has 50 cases across public-domain or US government text excerpts. It always runs without source filtering, so the expected evidence must be found by retrieval rather than pre-limited to a source file. Results are written under `eval/results/public/`, which is local runtime output and is not committed.

Latest local run: 50/50 passed, Recall@5 1.0, MRR@5 1.0, nDCG@5 1.0, retrieval P50 798.84 ms and P95 859.81 ms after model warmup and public corpus refresh.

## Internal Source-Filtered Regression

`docflow eval retrieval --refresh-sources --source-filter --write-results` refreshes the expected project source files, runs `eval/qa_v1.jsonl`, and reports:

- Recall@5
- MRR@5
- nDCG@5
- pass rate
- retrieval latency P50/P95/max

Results are written under `eval/results/`, which is local runtime output and is not committed.

Current committed retrieval set: 84 cases across README, public docs, contribution docs, security policy, roadmap, and changelog. Latest source-filtered local run: 84/84 passed, Recall@5 1.0, MRR@5 1.0, nDCG@5 1.0, retrieval P50 310.27 ms and P95 775.24 ms after model warmup.

This set is useful for project regression only. It uses source filtering for many checks, so it must not be presented as an external benchmark.

## Parsing Regression

`docflow eval parsing --write-results` checks the committed corpus in `eval/parsing_corpus/` against expectations in `eval/parsing_expected/`.

Current committed parsing set: 31 documents covering Markdown tables, Obsidian-flavored Markdown, TXT, mixed-language notes, code-like files, native PDFs, and DOCX. Latest local run: 31/31 passed, 42 chunks checked, 11,351 text characters checked.

## Incremental Indexing

`tests/test_incremental_index.py` covers the add, modify, and delete cycle with deterministic local fakes. It confirms changed files replace old vectors and deleted files are cleaned from SQLite metadata within the five-second regression limit.

## Internal Planning Report

The old maturity scorecard is an internal planning aid only. It is not a public quality claim and should not be used in README, release notes, or status summaries.

## Reproducibility

Local answer generation defaults to deterministic settings in `config.example.yaml`:

```yaml
query:
  seed: 42
  temperature: 0.0
```

Cloud LLM backends are reported as not reproducible because DocFlow cannot control their full serving environment.
