#!/usr/bin/env python
"""Ensemble step. Requires results/rankings/individual_rankings.json."""
import json, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from ranking.ensemble_rankers import run_ensemble_rankers
path = ROOT / "results/rankings/individual_rankings.json"
if not path.is_file():
    raise SystemExit(f"Missing individual rankings: {path}. Run scripts/04_run_feature_ranking.py first.")
rankings = json.loads(path.read_text())
# JSON keys are strings; values must be float-like dicts
rankings = {k: {fk: float(fv) for fk, fv in v.items()} for k, v in rankings.items()}
run_ensemble_rankers(rankings, str(ROOT / "results/ensemble_rankings"))
