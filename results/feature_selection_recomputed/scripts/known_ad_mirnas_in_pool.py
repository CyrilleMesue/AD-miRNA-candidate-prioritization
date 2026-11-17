"""Known AD miRNAs (HMDD, Disease == AD) in the 48-list top-20 pool, with the
same-dataset DE rule. HMDD matching uses the pipeline's rule (any of
get_mirna_versions(include_base=True) in HMDD Entity, case-insensitive).
DE: per-dataset entry -> that dataset's table (|logFC| > 0.1 and p < 0.05;
GSE215789 has none); shared-58 entry -> any of the three tables, reported
both with the original GSE120584 rule (no p filter) and with p < 0.05.
Usage: python known_ad_mirnas_in_pool.py"""
import sys
from pathlib import Path
import pandas as pd
ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "src")); sys.path.insert(0, str(Path(__file__).parent))
from prioritization.original import get_mirna_versions
from prioritize_with_per_dataset_ensembles import ensemble_lists
from prioritize_same_dataset_de import DE, SAME, load_de, lookup
from prioritization.original import sort_dict_features

hmdd = pd.read_csv(ROOT / "data/external/HMDD_val_data_miRNA.csv")
hmdd = hmdd[hmdd.Disease == "AD"]; score = dict(zip(hmdd.Entity.str.lower(), hmdd.Score))
de = load_de()
rows = []
for lst, scores in ensemble_lists().items():
    source = lst.split("|")[0]
    top = [f.lower() for f in sort_dict_features(scores, ascending="rank" in lst.split("|")[1], return_with_values=False)[:20]]
    for rank, mir in enumerate(top, 1):
        hits = [v.lower() for v in get_mirna_versions(mir, include_base=True) if v.lower() in score]
        rows.append(dict(list=lst, source=source, miRNA=mir, rank=rank,
                         hmdd_entity=";".join(hits), hmdd_score=max((score[h] for h in hits), default=0)))
pool = pd.DataFrame(rows)
checks = []
for _, r in pool[pool.hmdd_score > 0].iterrows():
    for ds in (list(DE) if r.source == "shared58" else SAME[r.source]):
        x = lookup(r.miRNA, de[ds])
        if x is None: continue
        fc, p = abs(x.logFC) > 0.1, x.P < 0.05
        checks.append(dict(miRNA=r.miRNA, source=r.source, de_dataset=ds, logFC=float(x.logFC), P=float(x.P),
                           pass_original=fc and (p or (r.source == "shared58" and not DE[ds][2])), pass_p05=fc and p))
c = pd.DataFrame(checks)
out = (pool[pool.hmdd_score > 0].groupby("miRNA")
       .agg(hmdd_entity=("hmdd_entity", "first"), hmdd_score=("hmdd_score", "max"),
            sources=("source", lambda s: ";".join(sorted(set(s)))), n_lists=("list", "nunique"), best_rank=("rank", "min")))
for rule in ("original", "p05"):
    ok = c[c[f"pass_{rule}"]]
    out[f"pass_{rule}"] = out.index.isin(ok.miRNA)
best = c[c.pass_original].sort_values("P").drop_duplicates("miRNA").set_index("miRNA")
out = out.join(best[["de_dataset", "logFC", "P"]])
out = out.sort_values(["hmdd_score", "n_lists"], ascending=False)
dest = ROOT / "results/candidates_same_dataset_de"; out.to_csv(dest / "known_ad_mirnas_in_pool.csv")
print("pool", pool.miRNA.nunique(), "| in HMDD-AD any", len(out), "| score>1", (out.hmdd_score > 1).sum())
for t in (1, 0):
    s = out[out.hmdd_score > t]
    print(f"score>{t}: pass_original {s.pass_original.sum()} pass_p05 {s.pass_p05.sum()}")
pd.set_option("display.width", 220); print(out.to_string())
