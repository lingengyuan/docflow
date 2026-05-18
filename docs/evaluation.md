# Evaluation

DocFlow currently has several validation paths:

```bash
.venv/bin/python -m pytest
docflow dev eval public --write-results
docflow dev eval retrieval --refresh-sources --source-filter --write-results
docflow dev eval parsing --write-results
docflow dev eval performance --write-results
docflow dev eval faithfulness --json
docflow dev eval large-library --documents 10000 --queries 20 --write-results
docflow dev eval external --json
docflow dev eval external run --query-limit 20 --distractors-per-query 3 --write-results
docflow dev browser-acceptance
docflow admin restore-drill
```

## Current Limitation

DocFlow uses measured checks as external quality evidence:

- Retrieval metrics.
- Citation alignment.
- Parsing regression checks.
- Incremental indexing checks.
- Parser/chunker performance smoke checks.
- Answer faithfulness checks.
- Desktop large-library synthetic benchmarks with explicit stage thresholds.
- Reproducibility checks.
- Offline privacy checks.

The public-domain regression set below is reproducible from committed files and is large enough to catch more routine retrieval regressions across law, NASA, literature, civic history, and the original smoke cases. It is still not a BEIR, MTEB, or C-MTEB score and should not be treated as a broad external benchmark.

## Public Reproducible Retrieval Benchmark

`docflow dev eval public --write-results` refreshes the committed public-domain corpus in `eval/public_corpus/`, runs `eval/public_retrieval_v1.jsonl`, and reports:

- Recall@5
- MRR@5
- nDCG@5
- pass rate
- retrieval latency P50/P95/max

The public regression set currently has 547 cases across public-domain literature, United States government texts, NASA summaries, and civic-history excerpts. It always runs without source filtering, so the expected evidence must be found by retrieval rather than pre-limited to a source file. Results are written under `eval/results/public/`, which is local runtime output and is not committed.

Latest local no-rerank run after public corpus refresh: 547/547 passed, Recall@5 0.9982, MRR@5 0.9145, nDCG@5 0.9357, retrieval P50 213.92 ms and P95 271.26 ms.

## GitHub CI and Scheduled Evaluation

GitHub CI now runs ruff, mypy, pytest, offline doctor, frontend checks, release surface, package smoke, parser/chunker performance smoke, and parsing eval. These checks are small enough to run on normal pushes and pull requests.

The full public retrieval benchmark uses Qdrant plus embedding model downloads, so it runs in a separate weekly evaluation workflow and can also be started manually. That split keeps normal pull-request feedback practical while still making the public retrieval number reproducible outside one developer machine.

## External Benchmarks

DocFlow tracks external benchmark readiness in `eval/external_benchmarks.json`. `docflow dev eval external --json` reports the current status and claim policy.

`docflow dev eval external run --dataset scifact --query-limit 20 --distractors-per-query 3 --write-results` runs the archived BEIR SciFact-lite subset. `docflow dev eval external run --dataset nfcorpus --query-limit 20 --distractors-per-query 3 --max-relevant-per-query 5 --write-results` runs the archived BEIR NFCorpus-lite subset. These commands intentionally download public BEIR datasets when they are not already cached; they are not part of DocFlow's default offline local-use path.

Latest archived BEIR SciFact-lite subset: 20 questions, Recall@5 0.95, MRR@5 0.95, nDCG@5 0.95, retrieval P50 315.66 ms and P95 700.93 ms, with 79 indexed documents and no source filtering.

Latest archived BEIR NFCorpus-lite subset: 20 questions, Recall@5 0.30, MRR@5 0.50, nDCG@5 0.3246, retrieval P50 187.99 ms and P95 412.77 ms, with 139 indexed documents, no source filtering, and max 5 relevant documents per query. The latest archived artifact is `eval/results/external/beir-nfcorpus-lite-08f3965.json`.

These are external subset results, not a full BEIR leaderboard score. The committed public and internal regression results must therefore stay labeled as DocFlow regression checks, not as broad BEIR, MTEB, or C-MTEB results.

Reference baselines:

- BEIR: heterogeneous zero-shot retrieval benchmark across public retrieval tasks and domains. Primary source: <https://arxiv.org/abs/2104.08663>.
- MTEB: embedding and retrieval benchmark with public tooling and leaderboard. Primary implementation: <https://github.com/embeddings-benchmark/mteb>.
- C-MTEB: Chinese embedding benchmark listed in MTEB as `MTEB(cmn, v1)`, covering Chinese tasks and datasets. Benchmark overview: <https://embeddings-benchmark.github.io/mteb/overview/available_benchmarks/#mtebcmn-v1>.

Release notes may quote an external benchmark only when the benchmark name, task split, model config, hardware/runtime summary, and raw result artifact are all listed.

## Internal Source-Filtered Regression

`docflow dev eval retrieval --refresh-sources --source-filter --write-results` refreshes the expected project source files, runs `eval/qa_v1.jsonl`, and reports:

- Recall@5
- MRR@5
- nDCG@5
- pass rate
- retrieval latency P50/P95/max

Results are written under `eval/results/`, which is local runtime output and is not committed.

Current committed retrieval set: 84 cases across README, public docs, contribution docs, security policy, roadmap, and changelog. Latest source-filtered local run: 84/84 passed, Recall@5 1.0, MRR@5 1.0, nDCG@5 1.0, retrieval P50 310.27 ms and P95 775.24 ms after model warmup.

This set is useful for project regression only. It uses source filtering for many checks, so it must not be presented as an external benchmark.

## Parsing Regression

`docflow dev eval parsing --write-results` checks the committed corpus in `eval/parsing_corpus/` against expectations in `eval/parsing_expected/`.

Current committed parsing set: 120 documents covering Markdown tables, long Markdown, wikilink/frontmatter/callout-style Markdown, TXT, mixed-language notes, noisy OCR-like text, code-like files, native PDFs, and DOCX. Latest local run: 120/120 passed, 147 chunks checked, 26,613 text characters checked.

## Incremental Indexing

`tests/test_incremental_index.py` covers the add, modify, and delete cycle with deterministic local fakes. It confirms changed files replace old vectors and deleted files are cleaned from SQLite metadata within the five-second regression limit.

## Performance Smoke

`docflow dev eval performance --write-results` generates synthetic local Markdown files and measures parser/chunker throughput without downloading models or calling external services. The standard local CI script runs this smoke check. It is a regression guard for long-note and many-note parsing, not a full large-library retrieval or model-latency benchmark.

Latest local run: passed; long note 73,947 bytes, 192 chunks, 3.33 ms total; many-note library 80 files, 80 chunks, 9.73 ms total.

## Large-Library Benchmark

`docflow dev eval large-library --documents 10000 --queries 20 --write-results` generates a synthetic desktop-scale Markdown library and measures local indexing, direct SQLite source lookup, full-text retrieval orchestration, citation construction, and deterministic answer assembly as separate stages. Each stage must return the expected synthetic note as the top result to pass.

This check still does not measure embedding, Qdrant vector search, MLX reranking, live LLM generation, or first-token latency. The standard CI path runs a 200-document smoke version with thresholds. The scheduled evaluation workflow runs the larger 10,000-document version and archives the result artifact.

Latest local run: passed; 10,000 documents, 10,000 chunks, all 20 queries returned the expected synthetic note as the top result in direct lookup, retrieval orchestration, and answer-path checks. The 20 query targets are spread across the synthetic library from note 250 through note 9,750. Indexing took 5,726.40 ms total, or 0.5726 ms per document. Stage P95 timings were 11.75 ms for direct lookup, 13.17 ms for retrieval orchestration, and 14.77 ms for deterministic answer assembly. No smoke thresholds failed. The archived artifact is `eval/results/large-library/large-library-5b57389.json`.

## Citation Alignment

`tests/test_citation_alignment.py`, `tests/test_claim_support.py`, `tests/test_query_service.py`, and stream conversation tests cover source coordinates, retrieved chunk validation, inline citation cleanup, structured `[[cite:chunk_id]]` markers, invalid marker rejection, duplicate marker handling, legacy citation compatibility, streamed-answer finalization before history is saved, sentence-level checks that flag answer claims without verified source markers, and deterministic source-content overlap checks that flag cited claims whose cited snippets do not share meaningful content terms.

The claim-support check is deterministic citation and source-content coverage. It proves whether each displayed claim carries a verified source marker and whether the cited snippets share meaningful surface terms with the claim. It does not replace human review, semantic entailment, or a broad factuality benchmark.

`docflow dev eval faithfulness --json` runs a deterministic answer-grounding fixture. It covers supported claims, uncited claims, fabricated source markers, wrong-source citations, no-evidence answers, partial citations, mismatched pages, conflicting sources, stale sources, multi-citation support, and insufficient-evidence answers. Latest local run: 14/14 passed.

## Reproducibility

Local answer generation defaults to deterministic settings in `config.example.yaml`:

```yaml
query:
  seed: 42
  temperature: 0.0
```

Cloud LLM backends are reported as not reproducible because DocFlow cannot control their full serving environment.
