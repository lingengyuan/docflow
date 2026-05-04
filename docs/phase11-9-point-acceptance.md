# Phase 11 9-Point Acceptance Baseline

Date: 2026-05-04

## Purpose

Phase 11 turns the 9-point maturity target into a repeatable baseline:

- A dimension scorecard for the 12 maturity dimensions.
- A fixed retrieval evidence set for key user-facing claims.
- A command that generates a combined report.

The baseline is not expected to pass every target yet. Its job is to make current gaps visible before Phase 12 through Phase 18 improve them.

## Source Files

- `eval/phase11_maturity_dimensions.json`
  - Current score, target score, evidence, gap, and next steps for every maturity dimension.
- `eval/phase11_questions.jsonl`
  - Fixed retrieval evidence cases for supported formats, phase rules, health, OCR/VLM, backup, citations, and the 9-point roadmap.
- `scripts/run_maturity_eval.py`
  - Generates the combined maturity and retrieval evidence report.

## Command

```bash
.venv/bin/python main.py maturity-eval --no-rerank
```

Use `--json` when a machine-readable report is needed:

```bash
.venv/bin/python main.py maturity-eval --no-rerank --json
```

Use `--skip-retrieval` only when Qdrant or the local corpus is unavailable. It still reports dimension scores, but it intentionally omits evidence checks.

## Current Baseline

As of Phase 11 implementation:

- Overall maturity score: `6.96 / 9.0`.
- Dimensions at target: `0 / 12`.
- Dimensions near target: `2 / 12`.
- Dimensions below target: `10 / 12`.
- Largest gaps:
  - 扩展生态: gap `4.0`.
  - 文件管理能力: gap `3.0`.
  - 成熟产品感: gap `3.0`.
  - 界面可用性: gap `2.5`.
  - 搜索和答案可靠性: gap `2.0`.

The first retrieval evidence baseline with `--no-rerank` returned `11 / 20` passing cases after re-ingesting the updated README and Phase 11 acceptance document and repairing the live vector index. The failures are expected at this stage and expose two useful facts:

- The live corpus still contains older DocFlow notes that can outrank current project files.
- Some roadmap claims are still difficult to retrieve with the current corpus and scoring terms.

## Acceptance Rule for Later Phases

Later phases should not treat the report as a release gate until the relevant phase updates the feature area. They should use it as a visible baseline:

- Phase 12 should improve file/library organization dimensions.
- Phase 13 should improve import, VLM, and workflow dimensions.
- Phase 14 should improve reliability and scoped-answer dimensions.
- Phase 15 should improve UI and product-feel dimensions.
- Phase 16 should improve model/runtime, health, setup, and recovery dimensions.
- Phase 17 should improve local knowledge workflow dimensions.
- Phase 18 should bring every dimension to `9 / 9` or document the remaining blocker.
