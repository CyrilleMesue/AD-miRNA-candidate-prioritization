#!/usr/bin/env python
import json, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))
from scripts.run_pipeline import load_config, stage_prioritize, ROOT as PIPE_ROOT
path = PIPE_ROOT / "results/ensemble_rankings/ensemble_rankings.json"
if not path.is_file():
    raise SystemExit(f"Missing ensemble rankings: {path}")
ensembles = json.loads(path.read_text())
ensembles = {k: {fk: float(fv) for fk, fv in v.items()} for k, v in ensembles.items()}
cfg = load_config(str(PIPE_ROOT / "config/analysis_config.yaml"))
stage_prioritize(cfg, PIPE_ROOT, ensembles)
