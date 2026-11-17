#!/usr/bin/env python
"""Prioritisation of the 48 ensemble lists with a same-dataset DE rule.

A miRNA that enters through a per-dataset ensemble list of dataset X must
pass |logFC| > 0.1 AND P.Value < 0.05 in the differential table of X:
  ROSMAP    -> rosmap_results_miRNA_Limma.csv
  GSE120584 -> GSE120584_results_miRNA_Limma.csv
  GSE215789 -> no differential table exists; such entries cannot pass.
A miRNA that enters through a shared-58 list may pass in any of the three
miRNA DE tables (ROSMAP, GSE120584, GSE218605), as in the original
pipeline. Two variants are written for that path:
  shared_rule="original": GSE120584 without a p-value filter (original)
  shared_rule="p05":      p < 0.05 required in every table
A miRNA is kept if any of its entry paths passes. The HMDD/target filters
(run_stage1) and the association pass are the unchanged pipeline code.
Version matching: first of get_mirna_versions(include_base=True) present
in the unfiltered DE table. Usage: python prioritize_same_dataset_de.py
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

from prioritization.candidate_selection import load_annotation_tables, run_stage1, target_association_pass
from prioritization.original import get_mirna_versions
from scripts.run_pipeline import load_config, resolve

sys.path.insert(0, str(Path(__file__).parent))
from prioritize_with_per_dataset_ensembles import ensemble_lists  # noqa: E402

DEST = ROOT / "results" / "candidates_same_dataset_de"
DEST.mkdir(parents=True, exist_ok=True)
DE = {"GSE120584": ("data/interim/GSE120584_results_miRNA_Limma.csv", "P.Value", False),
      "ROSMAP": ("data/interim/rosmap_results_miRNA_Limma.csv", "P.Value", True),
      "GSE218605": ("data/interim/GSE218605_results_miRNA_edgeR.csv", "PValue", True)}
SAME = {"ROSMAP": ["ROSMAP"], "GSE120584": ["GSE120584"], "GSE215789": []}
CANDIDATES = ["hsa-mir-324-3p", "hsa-mir-574-3p", "hsa-mir-769-3p"]


def load_de() -> dict:
    tables = {}
    for ds, (path, pcol, _) in DE.items():
        t = pd.read_csv(ROOT / path).rename(columns={"Unnamed: 0": "id"})
        t["id"] = t["id"].str.lower()
        tables[ds] = t.set_index("id")[["logFC", pcol]].rename(columns={pcol: "P"})
    return tables


def lookup(mir: str, table: pd.DataFrame):
    for v in get_mirna_versions(mir, include_base=True):
        if v in table.index:
            return table.loc[v]
    return None


def main() -> None:
    cfg = load_config(str(ROOT / "config/analysis_config.yaml"))
    dbs = load_annotation_tables({k: resolve(ROOT, v) for k, v in cfg["annotation"].items()})
    de = load_de()
    lists = ensemble_lists()
    with contextlib.redirect_stdout(io.StringIO()):
        stage1, keep = run_stage1(lists, dbs)
    entries = []  # one row per (list, miRNA, DE dataset tested)
    for lst, d in stage1.items():
        source = lst.split("|")[0]
        ranks = keep[lst]
        for mir in d["miR"]:
            rank = int(ranks[ranks.MicroRNAs == mir].Rank.values[0])
            tested = list(DE) if source == "shared58" else SAME[source]
            if not tested:
                entries.append(dict(list=lst, source=source, miRNA=mir, rank=rank, de_dataset=None,
                                    logFC=None, P=None, pass_original=False, pass_p05=False))
            for ds in tested:
                x = lookup(mir, de[ds])
                lfc, p = (None, None) if x is None else (float(x.logFC), float(x.P))
                ok_fc = lfc is not None and abs(lfc) > 0.1
                ok_p = p is not None and p < 0.05
                p_required = DE[ds][2] or source != "shared58"
                entries.append(dict(list=lst, source=source, miRNA=mir, rank=rank, de_dataset=ds, logFC=lfc, P=p,
                                    pass_original=ok_fc and (ok_p or not p_required), pass_p05=ok_fc and ok_p))
    e = pd.DataFrame(entries)
    e.to_csv(DEST / "de_checks_by_entry.csv", index=False)
    old11 = set((ROOT / "results/candidates/prioritized_mirnas.txt").read_text().split())
    summary = {"stage1_hmdd_target_pool": int(e.miRNA.nunique())}
    for rule in ("original", "p05"):
        ok = e[e[f"pass_{rule}"]]
        stage2 = pd.DataFrame({"Feature": ok.miRNA, "Rank": ok["rank"], "FeatureType": "miR",
                               "FeatureRanker": ok["list"], "Dataset": ok.de_dataset, "logFC": ok.logFC})
        with contextlib.redirect_stdout(io.StringIO()):
            sel = target_association_pass(stage2, dbs)
        final = sorted(set(sel["selected_miRs"]))
        per = (ok.groupby("miRNA").agg(entry_sources=("source", lambda s: ";".join(sorted(set(s)))),
                                       passing_de=("de_dataset", lambda s: ";".join(sorted(set(s)))),
                                       best_rank=("rank", "min")).reindex(final))
        per["in_current_11"] = per.index.isin(old11)
        per.to_csv(DEST / f"shortlist_shared_{rule}.csv")
        summary[f"shared_rule_{rule}"] = {
            "after_de_filter": int(ok.miRNA.nunique()), "after_association_pass": len(final),
            "candidates_retained": [c for c in CANDIDATES if c in final], "shortlist": final}
    failed = e.groupby("miRNA").filter(lambda g: not g.pass_original.any())
    summary["dropped_gse215789_untestable"] = sorted(
        failed.groupby("miRNA").filter(lambda g: set(g.source) == {"GSE215789"}).miRNA.unique())
    (DEST / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps({k: (v if not isinstance(v, dict) else {a: b for a, b in v.items() if a != "shortlist"})
                      for k, v in summary.items()}, indent=1))


if __name__ == "__main__":
    main()
