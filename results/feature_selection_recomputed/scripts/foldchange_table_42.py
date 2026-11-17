"""logFC, p-value and |logFC| rank of the 42 shortlisted miRNAs in the three miRNA DE tables.
Name matching follows attach_logfc (first of get_mirna_versions(include_base=True) present).
Ranks are over the full, unfiltered DE table. Usage: python foldchange_table_42.py"""
import sys
from pathlib import Path
import pandas as pd
ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))
from prioritization.original import get_mirna_versions
TABLES = {"GSE120584": ("data/interim/GSE120584_results_miRNA_Limma.csv", "P.Value", "adj.P.Val", False),
          "ROSMAP": ("data/interim/rosmap_results_miRNA_Limma.csv", "P.Value", "adj.P.Val", True),
          "GSE218605": ("data/interim/GSE218605_results_miRNA_edgeR.csv", "PValue", "FDR", True)}
D = ROOT / "results/candidates_with_per_dataset_ensembles"
mirs = D.joinpath("prioritized_mirnas.txt").read_text().split()
src = pd.read_csv(D / "shortlist_sources.csv").set_index("Feature")
old = set((ROOT / "results/candidates/prioritized_mirnas.txt").read_text().split())
rows = []
for m in mirs:
    r = {"miRNA": m, "in_current_11": m in old, "ensemble_sources": src.loc[m, "sources"], "best_ensemble_rank": int(src.loc[m, "best_rank"])}
    for ds, (path, pc, fc, pfilt) in TABLES.items():
        t = pd.read_csv(ROOT / path).rename(columns={"Unnamed: 0": "id"})
        t["id"] = t["id"].str.lower(); t["absrank"] = t.logFC.abs().rank(ascending=False, method="min").astype(int)
        t = t.set_index("id")
        hit = next((v for v in get_mirna_versions(m, include_base=True) if v in t.index), None)
        if hit is None:
            continue
        x = t.loc[hit]
        r |= {f"{ds}_logFC": round(x.logFC, 3), f"{ds}_P": x[pc], f"{ds}_FDR": x[fc],
              f"{ds}_absrank": f"{x.absrank}/{len(t)}",
              f"{ds}_passes": bool(abs(x.logFC) > 0.1 and (x[pc] < 0.05 or not pfilt))}
    rows.append(r)
out = pd.DataFrame(rows).sort_values("GSE120584_logFC", key=abs, ascending=False)
out.to_csv(D / "foldchange_42.csv", index=False)
