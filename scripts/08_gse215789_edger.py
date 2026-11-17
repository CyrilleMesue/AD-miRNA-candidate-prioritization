#!/usr/bin/env python
"""edgeR differential expression for GSE215789 (plasma small-RNA counts), AD vs Control.

Added 2026-11-06 so that miRNAs prioritized from GSE215789 ensemble rankings can be
tested in their own dataset. Uses the project's prepare_for_edger and run_edgeR.R
unchanged (filterByExpr, TMM, QL GLM, reference CTRL). MCI samples are excluded by
prepare_for_edger. Output: data/interim/GSE215789_results_miRNA_edgeR.csv

Usage: python scripts/08_gse215789_edger.py <Rscript path>
"""
import subprocess
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from differential.prepare_for_de import prepare_for_edger  # noqa: E402

rscript = sys.argv[1]
work = ROOT / "data/interim/de_inputs"
work.mkdir(parents=True, exist_ok=True)
df = pd.read_csv(ROOT / "data/processed/GSE215789_plasma_count.csv")
counts, samples = prepare_for_edger(df, ["Sample ID", "Gender", "Age", "Diagnosis"],
                                    work / "GSE215789_counts.csv", work / "GSE215789_samples.csv")
print("samples:", samples.group.value_counts().to_dict(), "features:", counts.shape[0])
out = ROOT / "data/interim/GSE215789_results_miRNA_edgeR.csv"
subprocess.run([rscript, str(ROOT / "src/differential/run_edgeR.R"), str(work / "GSE215789_counts.csv"),
                str(work / "GSE215789_samples.csv"), "group", str(out), "CTRL"], check=True)
res = pd.read_csv(out, index_col=0)
print("tested:", len(res), "| PValue<0.05:", int((res.PValue < 0.05).sum()), "| FDR<0.05:", int((res.FDR < 0.05).sum()))
