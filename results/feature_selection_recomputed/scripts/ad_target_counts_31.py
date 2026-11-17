"""Rank the 31 same-dataset-DE candidates by number of AD-associated gene targets.
AD-associated genes = first 100 CTD AD rows UNION first 100 GeneCards Alzheimer rows
(the pipeline's rule). Targets: TargetScan family + miRTarBase, matched with
get_mirna_versions(include_base=True) as in ad_mirna_target_pipeline step 3.
miRTarBase pairs are split into strongly validated (reporter/Western/qPCR > 0)
and unvalidated (sum 0; the pairs the pipeline keeps). Usage: python ad_target_counts_31.py"""
import sys
from pathlib import Path
import pandas as pd
ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "src"))
from prioritization.candidate_selection import load_annotation_tables
from prioritization.historical import get_mirna_versions
from scripts.run_pipeline import load_config, resolve
cfg = load_config(str(ROOT / "config/analysis_config.yaml"))
db = load_annotation_tables({k: resolve(ROOT, v) for k, v in cfg["annotation"].items()})
ad_genes = set(db["ctd"].Entity.to_list()[:100]) | set(db["genecards"].Entity.to_list()[:100])
ts, mt = db["targetscan"], db["mirtarbase"].drop_duplicates()
D = ROOT / "results/candidates_same_dataset_de"
short = pd.read_csv(D / "shortlist_shared_original.csv", index_col=0)
rows = []
for m in short.index:
    v = [x.lower() for x in get_mirna_versions(m, include_base=True)]
    ts_t = set(ts[ts["miR Family"].isin(v)]["Gene Symbol"])
    sub = mt[mt.miRNA.isin(v)]
    strong = sub[["Reporter assay", "Western blot", "qPCR"]].sum(axis=1) > 0
    mt_all, mt_strong, mt_weak = set(sub.Gene), set(sub[strong].Gene), set(sub[~strong].Gene)
    ad_ts, ad_mt, ad_s, ad_w = (s & ad_genes for s in (ts_t, mt_all, mt_strong, mt_weak))
    rows.append(dict(miRNA=m, in_current_11=short.loc[m, "in_current_11"], entered_via=short.loc[m, "entry_sources"],
                     total_targets=len(ts_t | mt_all), ad_targets=len(ad_ts | ad_mt), ad_targetscan=len(ad_ts),
                     ad_mirtarbase=len(ad_mt), ad_mirtarbase_strong=len(ad_s), ad_mirtarbase_unvalidated_only=len(ad_w - ad_s),
                     ad_genes=";".join(sorted(ad_ts | ad_mt))))
out = pd.DataFrame(rows).sort_values(["ad_targets", "ad_mirtarbase"], ascending=False)
out["rank"] = out.ad_targets.rank(ascending=False, method="min").astype(int)
out.to_csv(D / "ad_target_counts_31.csv", index=False)
print(len(ad_genes)); pd.set_option("display.width", 250); pd.set_option("display.max_colwidth", 90)
print(out.drop(columns=["entered_via"]).to_string(index=False))
