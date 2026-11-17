#!/usr/bin/env python
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))
from scripts.run_pipeline import load_config, stage_matrices, stage_rank, ROOT as PIPE_ROOT
cfg = load_config(str(PIPE_ROOT / "config/analysis_config.yaml"))
datasets = stage_matrices(cfg, PIPE_ROOT)
stage_rank(cfg, PIPE_ROOT, datasets)
