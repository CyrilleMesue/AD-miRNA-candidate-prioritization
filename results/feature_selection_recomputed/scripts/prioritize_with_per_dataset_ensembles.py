#!/usr/bin/env python
"""Candidate prioritisation using the 12 shared-feature (58 miRNA) ensemble
lists plus the 36 per-dataset ensemble lists (12 aggregators x 3 datasets).

The prioritisation functions are the unchanged run_stage1 / attach_logfc /
target_association_pass from src/prioritization/candidate_selection.py.
Outputs go to results/candidates_with_per_dataset_ensembles/ so the
existing results/candidates/ shortlist is not overwritten.

Usage: python scripts/prioritize_with_per_dataset_ensembles.py
"""

from __future__ import annotations

import contextlib
import io
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("MPLBACKEND", "Agg")
ROOT = Path(__file__).resolve().parents[3]
OUT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

import pandas as pd

from prioritization.candidate_selection import (
    attach_logfc,
    load_annotation_tables,
    load_differential_tables,
    run_stage1,
    target_association_pass,
)
from scripts.run_pipeline import load_config, resolve

DEST = ROOT / "results" / "candidates_with_per_dataset_ensembles"
DEST.mkdir(parents=True, exist_ok=True)
CANDIDATES = ["hsa-mir-324-3p", "hsa-mir-574-3p", "hsa-mir-769-3p"]


def ensemble_lists() -> dict:
    lists = {}
    g = pd.read_csv(OUT / "ensembles" / "ensemble_rankings.csv")
    for agg, d in g.groupby("aggregator", sort=False):
        lists[f"shared58|{agg}"] = dict(zip(d.feature_id, d.ensemble_score))
    t = pd.read_csv(OUT / "ensembles_per_dataset" / "ensemble_rankings_by_dataset.csv")
    for (ds, agg), d in t.groupby(["dataset", "aggregator"], sort=False):
        lists[f"{ds}|{agg}"] = dict(zip(d.feature_id, d.ensemble_score))
    return lists


def prioritize(lists: dict, dbs: dict, de: dict) -> dict:
    with contextlib.redirect_stdout(io.StringIO()):
        stage1, keep = run_stage1(lists, dbs)
        stage2 = attach_logfc(stage1, keep, de)
        sel = target_association_pass(stage2, dbs)
    pool = set()
    for agg, frame in keep.items():
        pool.update(frame.MicroRNAs.head(20))
    hmdd_novel = set().union(*(s["miR"] for s in stage1.values()))
    return dict(pool=pool, stage1=hmdd_novel, stage2=stage2, sel=sel)


def main() -> None:
    cfg = load_config(str(ROOT / "config/analysis_config.yaml"))
    dbs = load_annotation_tables({k: resolve(ROOT, v) for k, v in cfg["annotation"].items()})
    de = load_differential_tables({k: resolve(ROOT, v) for k, v in cfg["differential"].items()})
    lists = ensemble_lists()
    configs = {
        "shared58_only": {k: v for k, v in lists.items() if k.startswith("shared58|")},
        "per_dataset_only": {k: v for k, v in lists.items() if not k.startswith("shared58|")},
        "shared58_plus_per_dataset": lists,
    }
    summary = {}
    for name, sub in configs.items():
        r = prioritize(sub, dbs, de)
        final = sorted(r["sel"]["stage2_mirnas"])
        summary[name] = {
            "n_lists": len(sub),
            "top20_pool": len(r["pool"]),
            "after_hmdd_target_filters": len(r["stage1"]),
            "after_logfc_filter": len(final),
            "after_association_pass": len(set(r["sel"]["selected_miRs"])),
            "shortlist": final,
            "candidates_retained": [c for c in CANDIDATES if c in final],
        }
        if name == "shared58_plus_per_dataset":
            r["stage2"].to_csv(DEST / "stage2_logfc_filtered.csv", index=False)
            (DEST / "prioritized_mirnas.txt").write_text("\n".join(final) + "\n")
            r["sel"]["selected_microRNA_ranks"].to_csv(DEST / "selected_microrna_ranks.csv", index=False)
            src = r["stage2"][r["stage2"].FeatureType == "miR"].copy()
            src["source"] = src.FeatureRanker.str.split("|").str[0]
            (src.groupby("Feature").agg(sources=("source", lambda x: ";".join(sorted(set(x)))),
                                        n_lists=("FeatureRanker", "nunique"),
                                        best_rank=("Rank", "min"),
                                        de_datasets=("Dataset", lambda x: ";".join(sorted(set(x)))))
                .sort_values(["n_lists", "best_rank"], ascending=[False, True])
                .to_csv(DEST / "shortlist_sources.csv"))
        print(name, {k: v for k, v in summary[name].items() if k != "shortlist"})
    (DEST / "summary.json").write_text(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
