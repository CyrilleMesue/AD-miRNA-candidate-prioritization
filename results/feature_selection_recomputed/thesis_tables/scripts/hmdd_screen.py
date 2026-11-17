"""HMDD screen of the 12 ensemble top-20 lists (thesis Results, candidate prioritization).
Uses the pipeline's rule from ad_mirna_target_pipeline step 2: a miRNA is a previously
reported AD miRNA if any of get_mirna_versions(include_base=True) matches an HMDD
Entity (case-insensitive) among AD records with Score > 1.
Usage: python -I hmdd_screen.py <repo root> <out dir>"""
import sys, json
from pathlib import Path
import pandas as pd
root, out = Path(sys.argv[1]), Path(sys.argv[2]); sys.path.insert(0, str(root / "src"))
from prioritization.original import get_mirna_versions, sort_dict_features
h = pd.read_csv(root / "data/external/HMDD_val_data_miRNA.csv"); h = h[h.Disease == "AD"]
allscore = dict(zip(h.Entity.str.lower(), h.Score)); known = {k for k, v in allscore.items() if v > 1}
e = pd.read_csv(root / "results/feature_selection_recomputed/ensembles/ensemble_rankings.csv")
e = e[e.aggregator != "max_weight"]  # maximum weight excluded after Results 3.1.4 (all 58 miRNAs tied; thesis decision 2026-11-16)
rows = []
for agg, g in e.groupby("aggregator", sort=False):
    top = [f.lower() for f in sort_dict_features(dict(zip(g.feature_id, g.ensemble_score)), ascending="rank" in agg, return_with_values=False)[:20]]
    for r, m in enumerate(top, 1):
        v = [x.lower() for x in get_mirna_versions(m, include_base=True)]
        rows.append(dict(aggregator=agg, rank=r, miRNA=m, hmdd_known=any(x in known for x in v),
                         hmdd_score=max([allscore.get(x, 0) for x in v]), hmdd_entity=";".join(x for x in v if x in allscore)))
d = pd.DataFrame(rows); d.to_csv(out / "hmdd_screen_positions.csv", index=False)
per = d.groupby("miRNA").agg(n_lists=("aggregator", "nunique"), best=("rank", "min"), median=("rank", "median"),
                             hmdd_known=("hmdd_known", "first"), hmdd_score=("hmdd_score", "first"), entity=("hmdd_entity", "first"))
per.sort_values(["hmdd_known", "n_lists", "best"], ascending=[True, False, True]).to_csv(out / "hmdd_screen_mirnas.csv")
s = dict(positions=len(d), distinct=int(per.shape[0]),
         removed_positions=int(d.hmdd_known.sum()), removed_distinct=int(per.hmdd_known.sum()),
         retained_positions=int((~d.hmdd_known).sum()), retained_distinct=int((~per.hmdd_known).sum()),
         retained_with_single_record=int(((~per.hmdd_known) & (per.hmdd_score == 1)).sum()),
         retained_per_list={a: int((~g.hmdd_known).sum()) for a, g in d.groupby("aggregator", sort=False)},
         hmdd_ad_records=int(len(h)), hmdd_ad_score_gt1=len(known))
(out / "hmdd_screen_summary.json").write_text(json.dumps(s, indent=2)); print(json.dumps(s, indent=1))
pd.set_option("display.width", 200); print(per.sort_values(["hmdd_known", "n_lists", "best"], ascending=[True, False, True]).to_string())
