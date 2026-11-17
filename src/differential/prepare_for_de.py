"""Reshape processed matrices for limma/edgeR.

From val_notebooks/Prepare Data for Differential Analysis.ipynb.
Control is relabeled CTRL because run_limma.R and run_edgeR.R are
called with reference group CTRL.
"""
import pandas as pd


def prepare_for_edger(df, meta_cols, counts_out, samples_out):
    df = df.copy()
    if "Sample ID" not in df.columns:
        raise ValueError("DataFrame must contain a 'Sample ID' column.")
    if "Diagnosis" not in df.columns:
        raise ValueError("DataFrame must contain a 'Diagnosis' column for group assignment.")
    df = df[df["Diagnosis"].isin(["AD", "Control"])].copy()
    df["group"] = df["Diagnosis"].replace({"Control": "CTRL"})
    feature_cols = [c for c in df.columns if c not in meta_cols + ["group"]]
    counts = df.set_index("Sample ID")[feature_cols].T
    counts.index.name = "feature"
    samples = df.set_index("Sample ID")[["group"] + [c for c in meta_cols if c != "Sample ID"]]
    counts.to_csv(counts_out)
    samples.to_csv(samples_out)
    return counts, samples
