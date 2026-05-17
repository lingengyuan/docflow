from __future__ import annotations

import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "run_phase110_readiness_check.py"


def load_readiness_module():
    spec = importlib.util.spec_from_file_location("phase110_readiness", SCRIPT)
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_public_docs_do_not_claim_subjective_90_plus_score() -> None:
    readiness = load_readiness_module()
    readiness.check_public_score_claims()


def test_status_records_scorecard_and_benchmark_boundaries() -> None:
    readiness = load_readiness_module()
    readiness.check_status_scorecard_alignment()
