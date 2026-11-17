"""Table for Results 3.1.6: target retrieval and AD-gene interactions of the 25 miRNAs retained
after the HMDD screen in the ORIGINAL prioritization (12 rankings of the 58 shared miRNAs).
Retrieval uses mature + stem identifiers (include_base=True); the interaction step uses the
original lookup (-3p/-5p forms only). Untested = no reporter-assay, Western-blot or qPCR evidence.
Usage: python make_target_table.py <thesis_tables dir>"""
import os
import sys
from pathlib import Path

os.environ.setdefault("MPLBACKEND", "Agg")
ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "src"))
import pandas as pd

from prioritization.candidate_selection import load_annotation_tables
from prioritization.original import get_mirna_versions, sort_dict_features
from scripts.run_pipeline import load_config, resolve

out = Path(sys.argv[1])
cfg = load_config(str(ROOT / "config/analysis_config.yaml"))
db = load_annotation_tables({k: resolve(ROOT, v) for k, v in cfg["annotation"].items()})
AD = set(db["ctd"].Entity.to_list()[:100]) | set(db["genecards"].Entity.to_list()[:100])
ts, mt = db["targetscan"], db["mirtarbase"]
known = set(db["hmdd"].Entity.str.lower())
CAND = {"hsa-mir-324-3p", "hsa-mir-574-3p", "hsa-mir-769-3p"}

e = pd.read_csv(ROOT / "results/feature_selection_recomputed/ensembles/ensemble_rankings.csv")
e = e[e.aggregator != "max_weight"]  # maximum weight excluded after Results 3.1.4 (all 58 miRNAs tied; thesis decision 2026-11-16)
pool = set()
for a, g in e.groupby("aggregator", sort=False):
    pool |= {m.lower() for m in sort_dict_features(dict(zip(g.feature_id, g.ensemble_score)),
                                                   ascending="rank" in a, return_with_values=False)[:20]}
retained = sorted(m for m in pool if not any(v.lower() in known for v in get_mirna_versions(m, include_base=True)))
assert len(pool) == 38 and len(retained) == 25

rows = []
for m in retained:
    vb = [v.lower() for v in get_mirna_versions(m, include_base=True)]
    T = set(ts[ts["miR Family"].isin(vb)]["Gene Symbol"]); M = set(mt[mt.miRNA.isin(vb)].Gene)
    vn = [v.lower() for v in get_mirna_versions(m)]
    sub = mt[mt.miRNA.isin(vn) & mt.Gene.isin(AD)].drop_duplicates()
    E = sub[["Reporter assay", "Western blot", "qPCR"]].sum(axis=1)
    rows.append(dict(miRNA=m, ts=len(T), mt=len(M), union=len(T | M), ad=len((T | M) & AD),
                     untested=sub[E == 0].Gene.nunique()))
t = pd.DataFrame(rows)
t["kept"] = t.untested > 0
assert t.kept.sum() == 20
t = t.sort_values(["kept", "ad", "union"], ascending=[False, False, False])
t.to_csv(out / "target_analysis_original.csv", index=False)


def name(m):
    s = m.replace("hsa-mir-", "miR-")
    return r"\textbf{%s}" % s if m in CAND else s


L = [r"""\begin{table}[htbp]
\centering
\caption{Target genes and AD-associated interactions of the 25 miRNAs retained after the HMDD screen.}
\label{tab:target_analysis}
{\footnotesize
\begin{tabular}{l r r r r r l}
\hline
 & \multicolumn{3}{c}{\textbf{Target genes}} & \textbf{AD-assoc.} & \textbf{AD-assoc. genes with} & \\
\cline{2-4}
\textbf{miRNA} & \textbf{TargetScan} & \textbf{miRTarBase} & \textbf{Combined} & \textbf{targets} & \textbf{qualifying interactions} & \textbf{Outcome} \\
\hline"""]
for _, r in t.iterrows():
    fmt = lambda x: f"{x:,}"
    L.append(f"{name(r.miRNA)} & {fmt(r.ts)} & {fmt(r.mt)} & {fmt(r.union)} & {r.ad} & {r.untested} & "
             + ("Retained" if r.kept else r"Removed$^{a}$") + r" \\")
L.append(r"""\hline
\end{tabular}}
\par\smallskip
\parbox{\textwidth}{\footnotesize ``Combined'' counts distinct genes in the union of the TargetScan and miRTarBase target sets. ``AD-associated targets'' counts genes in that union that appear in the 176-gene CTD and GeneCards reference set. ``AD-associated genes with qualifying interactions'' counts genes with at least one retained miRTarBase record lacking reporter assay, Western blot, and qPCR evidence. A miRNA passed the target screen if it had at least one such gene. The \textit{hsa-} prefix is omitted. Bold indicates the three miRNAs selected for experimental investigation. $^{a}$For miRNAs without an arm designation, the original interaction lookup queried only the 3p and 5p forms (Section~\ref{subsec:mirna_target_analysis}).}
\end{table}""")
(out / "target_analysis_table.tex").write_text("\n".join(L) + "\n")
print(t.to_string(index=False))
