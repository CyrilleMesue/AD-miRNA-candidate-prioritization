#!/usr/bin/env python
"""Candidate prioritization from 48 ensemble rankings (main thesis analysis, 2026-11-06).

Lists: 12 ensemble rankings of the 58 miRNAs shared by all 39 standalone rankings
("shared58") + 12 ensemble rankings within each of ROSMAP, GSE120584 and GSE215789.
Top 20 of every list -> run_stage1 (HMDD screen, target retrieval, AD-gene association,
untested miRTarBase interaction; unchanged pipeline code incl. the 2026-11-06 lookup fix).

Differential-expression rule (|logFC| > 0.1 AND p < 0.05 in every table):
  per-dataset entry  -> must pass in that dataset's own table
  shared58 entry     -> may pass in any miRNA table (SHARED_TABLES)
A miRNA is kept if any of its entry paths passes. Then target_association_pass.

Usage: python scripts/09_prioritize_48_lists.py [--shared-tables ROSMAP,GSE120584,GSE215789]
Outputs: results/candidates_48_lists/
"""
from __future__ import annotations

import argparse
import contextlib
import io
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("MPLBACKEND", "Agg")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))
import pandas as pd

from prioritization.candidate_selection import load_annotation_tables, run_stage1, target_association_pass
from prioritization.historical import get_mirna_versions, sort_dict_features
from scripts.run_pipeline import load_config, resolve

FS = ROOT / "results/feature_selection_recomputed"
DE = {
    "ROSMAP": ("data/interim/rosmap_results_miRNA_Limma.csv", "P.Value"),
    "GSE120584": ("data/interim/GSE120584_results_miRNA_Limma.csv", "P.Value"),
    "GSE215789": ("data/interim/GSE215789_results_miRNA_edgeR.csv", "PValue"),
    "GSE218605": ("data/interim/GSE218605_results_miRNA_edgeR.csv", "PValue"),
}
CANDIDATES = ["hsa-mir-324-3p", "hsa-mir-574-3p", "hsa-mir-769-3p"]
FC, P = 0.1, 0.05


def ensemble_lists() -> dict:
    lists = {}
    g = pd.read_csv(FS / "ensembles/ensemble_rankings.csv")
    for agg, d in g.groupby("aggregator", sort=False):
        lists[f"shared58|{agg}"] = dict(zip(d.feature_id, d.ensemble_score))
    t = pd.read_csv(FS / "ensembles_per_dataset/ensemble_rankings_by_dataset.csv")
    for (ds, agg), d in t.groupby(["dataset", "aggregator"], sort=False):
        lists[f"{ds}|{agg}"] = dict(zip(d.feature_id, d.ensemble_score))
    return {k: v for k, v in lists.items() if not k.endswith("|max_weight")}  # maximum weight excluded after Results 3.1.4 (all 58 miRNAs tied; thesis decision 2026-11-16)


def load_de() -> dict:
    out = {}
    for ds, (path, pcol) in DE.items():
        t = pd.read_csv(ROOT / path).rename(columns={"Unnamed: 0": "id"})
        t["id"] = t["id"].astype(str).str.lower()
        out[ds] = t.drop_duplicates("id").set_index("id")[["logFC", pcol]].rename(columns={pcol: "P"})
    return out


def lookup(mir: str, table: pd.DataFrame):
    for v in get_mirna_versions(mir, include_base=True):
        if v in table.index:
            return v, table.loc[v]
    return None, None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--shared-tables", default="ROSMAP,GSE120584,GSE215789")
    ap.add_argument("--out", default=str(ROOT / "results/candidates_48_lists"))
    a = ap.parse_args()
    shared_tables = a.shared_tables.split(",")
    dest = Path(a.out); dest.mkdir(parents=True, exist_ok=True)
    cfg = load_config(str(ROOT / "config/analysis_config.yaml"))
    dbs = load_annotation_tables({k: resolve(ROOT, v) for k, v in cfg["annotation"].items()})
    de = load_de()
    lists = ensemble_lists()

    pos = []
    for name, sc in lists.items():
        for r, m in enumerate(sort_dict_features(sc, ascending="rank" in name.split("|")[1],
                                                 return_with_values=False)[:20], 1):
            pos.append(dict(list=name, source=name.split("|")[0], aggregator=name.split("|")[1], rank=r, miRNA=m.lower()))
    pos = pd.DataFrame(pos)
    hmdd = set(dbs["hmdd"].Entity.str.lower())
    pos["hmdd_known"] = pos.miRNA.map(lambda m: any(v.lower() in hmdd for v in get_mirna_versions(m, include_base=True)))

    with contextlib.redirect_stdout(io.StringIO()):
        stage1, keep = run_stage1(lists, dbs)
    pos["passes_stage1"] = [m in stage1[l]["miR"] for l, m in zip(pos["list"], pos.miRNA)]

    checks = []
    for _, r in pos[pos.passes_stage1].iterrows():
        for ds in (shared_tables if r.source == "shared58" else [r.source]):
            hit, x = lookup(r.miRNA, de[ds])
            ok = x is not None and abs(x.logFC) > FC and x.P < P
            checks.append(dict(list=r["list"], source=r.source, miRNA=r.miRNA, rank=r["rank"], de_dataset=ds,
                               matched_id=hit, logFC=None if x is None else float(x.logFC),
                               P=None if x is None else float(x.P), passes=bool(ok)))
    c = pd.DataFrame(checks)
    ok = c[c.passes]
    pos["passes_de"] = [((ok["list"] == l) & (ok.miRNA == m)).any() for l, m in zip(pos["list"], pos.miRNA)]
    stage2 = pd.DataFrame({"Feature": ok.miRNA, "Rank": ok["rank"], "FeatureType": "miR",
                           "FeatureRanker": ok["list"], "Dataset": ok.de_dataset, "logFC": ok.logFC})
    with contextlib.redirect_stdout(io.StringIO()):
        sel = target_association_pass(stage2, dbs)
    final = sorted(set(sel["selected_miRs"]))
    assert set(final) == set(ok.miRNA), "association pass removed miRNAs; inspect before reporting"

    def fate(m):
        g = pos[pos.miRNA == m]
        if g.hmdd_known.all(): return "Removed: HMDD"
        if not g.passes_stage1.any(): return "Removed: AD target"
        if not g.passes_de.any(): return "Removed: fold change"
        return "Selected" if m in CANDIDATES else "Shortlisted"

    per = pos.groupby("miRNA").agg(n_lists=("list", "nunique"), sources=("source", lambda s: ";".join(sorted(set(s)))),
                                   best=("rank", "min")).reset_index()
    per["fate"] = per.miRNA.map(fate)
    best = ok.sort_values("P").drop_duplicates("miRNA").set_index("miRNA")
    per = per.join(best[["de_dataset", "logFC", "P"]], on="miRNA")
    pos["fate"] = pos.miRNA.map(dict(zip(per.miRNA, per.fate)))
    pos.to_csv(dest / "positions.csv", index=False); c.to_csv(dest / "de_checks.csv", index=False)
    per.to_csv(dest / "mirna_fate.csv", index=False)
    (dest / "prioritized_mirnas.txt").write_text("\n".join(final) + "\n")
    summary = {
        "shared_tables": shared_tables, "n_lists": len(lists), "positions": len(pos),
        "steps_mirnas": {"pool": int(per.shape[0]),
                         "after_hmdd": int((per.fate != "Removed: HMDD").sum()),
                         "after_ad_target": int(per.fate.isin(["Removed: fold change", "Shortlisted", "Selected"]).sum()),
                         "after_fold_change": len(final), "selected": 3},
        "steps_positions": {"pool": len(pos), "after_hmdd": int((~pos.hmdd_known).sum()),
                            "after_ad_target": int(pos.passes_stage1.sum()), "after_fold_change": int(pos.passes_de.sum())},
        "candidates_retained": [m for m in CANDIDATES if m in final],
        "shortlist": final,
    }
    (dest / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps({k: v for k, v in summary.items() if k != "shortlist"}, indent=1))


if __name__ == "__main__":
    main()
