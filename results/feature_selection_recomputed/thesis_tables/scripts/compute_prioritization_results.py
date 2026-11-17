"""Numbers for Results 3.1.6-3.1.8: original fold-change filter (20 miRNAs; ROSMAP, GSE120584, GSE215789,
|log2FC| > 0.1 and p < 0.05 in at least one), revised target step
(144 -> 110, with removal reasons) and revised fold-change step by entry ranking. Asserts agreement with
results/candidates_uniform_de/ (original) and results/candidates_48_lists/ (revised). Writes prioritization_results.json."""
import json, os, sys
from collections import Counter
from pathlib import Path
os.environ.setdefault("MPLBACKEND", "Agg")
ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "src"))
import pandas as pd
from prioritization.candidate_selection import load_annotation_tables, load_differential_tables
from prioritization.original import get_mirna_versions, mirtarbase_query_names
from scripts.run_pipeline import load_config, resolve

cfg = load_config(str(ROOT / "config/analysis_config.yaml"))
db = load_annotation_tables({k: resolve(ROOT, v) for k, v in cfg["annotation"].items()})
AD = set(db["ctd"].Entity.to_list()[:100]) | set(db["genecards"].Entity.to_list()[:100])
ts, mt = db["targetscan"], db["mirtarbase"]
ev = mt[["Reporter assay", "Western blot", "qPCR"]].sum(axis=1)

RAW = {"ROSMAP": ("data/interim/rosmap_results_miRNA_Limma.csv", "P.Value", "adj.P.Val"),
       "GSE120584": ("data/interim/GSE120584_results_miRNA_Limma.csv", "P.Value", "adj.P.Val"),
       "GSE215789": ("data/interim/GSE215789_results_miRNA_edgeR.csv", "PValue", "FDR")}
raw = {}
for k, (p, pc, fc) in RAW.items():
    t = pd.read_csv(ROOT / p).rename(columns={"Unnamed: 0": "id"}); t["id"] = t.id.astype(str).str.lower()
    raw[k] = t.drop_duplicates("id").set_index("id")[["logFC", pc, fc]].set_axis(["logFC", "P", "FDR"], axis=1)
out = {"prefilter": {k: {"n": len(v), "p05": int((v.P < 0.05).sum())} for k, v in raw.items()}}


def look(m, k):
    for v in get_mirna_versions(m, include_base=True):
        if v in raw[k].index:
            return v, raw[k].loc[v]
    return None, None


# ---------------- A. original DE for the 20 miRNAs
orig = pd.read_csv(ROOT / "results/feature_selection_recomputed/thesis_tables/target_analysis_original.csv")
twenty = orig[orig.kept].miRNA.tolist(); assert len(twenty) == 20
rowsA = []
for m in twenty:
    r = {"miRNA": m}; ok = False
    for k in ("ROSMAP", "GSE120584", "GSE215789"):
        hit, x = look(m, k)
        if x is None:
            r[k] = None; continue
        avail = x.P < 0.05   # uniform rule: p < 0.05 in every dataset
        passed = avail and abs(x.logFC) > 0.1
        ok |= passed
        r[k] = dict(id=hit, logFC=round(float(x.logFC), 3), P=float(x.P), FDR=float(x.FDR), avail=bool(avail), passed=bool(passed))
    r["kept"] = ok; rowsA.append(r)
saved = set((ROOT / "results/candidates_uniform_de/prioritized_mirnas.txt").read_text().split())
assert {r["miRNA"] for r in rowsA if r["kept"]} == saved
out["A"] = rowsA

# ---------------- B. revised target step (fixed lookup), by entry source
pos = pd.read_csv(ROOT / "results/candidates_48_lists/positions.csv")
fate = pd.read_csv(ROOT / "results/candidates_48_lists/mirna_fate.csv")
after_h = sorted(set(pos[~pos.hmdd_known].miRNA)); assert len(after_h) == 127
pooled, reasons, pairs_un, pairs_val_only, recs, recs_val = set(), {}, set(), set(), 0, 0
genes_un = Counter()
for m in after_h:
    vb = [v.lower() for v in get_mirna_versions(m, include_base=True)]
    T = set(ts[ts["miR Family"].isin(vb)]["Gene Symbol"]); M = set(mt[mt.miRNA.isin(vb)].Gene); G = T | M; pooled |= G
    vq = [v.lower() for v in mirtarbase_query_names(m)]
    sub = mt[mt.miRNA.isin(vq) & mt.Gene.isin(AD)].drop_duplicates()
    e_ = sub[["Reporter assay", "Western blot", "qPCR"]].sum(axis=1)
    recs += len(sub); recs_val += int((e_ > 0).sum())
    un = set(sub[e_ == 0].Gene); va = set(sub[e_ > 0].Gene)
    for g in un: pairs_un.add((m, g)); genes_un[g] += 1
    for g in va - un: pairs_val_only.add((m, g))
    if un and "_" in m: reasons[m] = "compound identifier not mapped back"
    elif un: reasons[m] = "kept"
    elif not G: reasons[m] = "no targets"
    elif not (G & AD): reasons[m] = "no AD-associated target"
    elif not len(sub): reasons[m] = "AD targets only without a miRTarBase record in queried form" if (M & AD) else "AD targets from TargetScan only"
    else: reasons[m] = "only validated interactions"
kept = [m for m, r in reasons.items() if r == "kept"]
assert len(kept) == 96, len(kept)
assert set(kept) == set(pos[pos.passes_stage1].miRNA)
out["B"] = dict(pooled=len(pooled), ad=len(pooled & AD), records=recs, records_validated=recs_val,
                untested_pairs=len(pairs_un), untested_genes=len(genes_un), val_only_pairs=len(pairs_val_only),
                top_genes=genes_un.most_common(10), reasons=Counter(r for r in reasons.values()))
srcs = ["shared58", "ROSMAP", "GSE120584", "GSE215789"]
tab = []
for s in srcs:
    ps = pos[pos.source == s]
    pool = set(ps.miRNA); h = set(ps[~ps.hmdd_known].miRNA); t = set(ps[ps.passes_stage1].miRNA); d = set(ps[ps.passes_de].miRNA)
    tab.append(dict(source=s, positions=len(ps), pool=len(pool), after_hmdd=len(h), after_target=len(t), after_de=len(d),
                    pos_hmdd=int((~ps.hmdd_known).sum()), pos_target=int(ps.passes_stage1.sum()), pos_de=int(ps.passes_de.sum())))
tab.append(dict(source="all", positions=len(pos), pool=pos.miRNA.nunique(), after_hmdd=127, after_target=96, after_de=44,
                pos_hmdd=int((~pos.hmdd_known).sum()), pos_target=int(pos.passes_stage1.sum()), pos_de=int(pos.passes_de.sum())))
out["source_table"] = tab
out["B_removed_reason_by_mirna"] = {m: r for m, r in reasons.items() if r != "kept"}

# ---------------- C. revised DE: reasons per (source, miRNA) for the 110
chk = pd.read_csv(ROOT / "results/candidates_48_lists/de_checks.csv")
c2 = chk.drop_duplicates(["source", "miRNA", "de_dataset"])
def why(r):
    if pd.isna(r.logFC): return "not tested"
    if abs(r.logFC) <= 0.1 and r.P >= 0.05: return "fc and p"
    if abs(r.logFC) <= 0.1: return "fc"
    if r.P >= 0.05: return "p"
    return "pass"
c2 = c2.assign(why=c2.apply(why, axis=1))
per = c2.groupby(["source", "miRNA"]).why.apply(lambda w: "pass" if "pass" in set(w) else ("not tested" if set(w) == {"not tested"} else "/".join(sorted(set(w) - {"not tested"})))).reset_index()
out["C_by_source"] = per.groupby(["source", "why"]).size().unstack(fill_value=0).to_dict(orient="index")
fails = set(fate[fate.fate == "Removed: fold change"].miRNA)
out["C_n_removed"] = len(fails)
sh = fate[fate.fate.isin(["Shortlisted", "Selected"])]
out["C_shortlist_direction"] = {"up": int((sh.logFC > 0).sum()), "down": int((sh.logFC < 0).sum())}
out["C_215789_tested_of_entered"] = per[per.source == "GSE215789"].why.value_counts().to_dict()
json.dump(out, open(str(ROOT / "results/feature_selection_recomputed/thesis_tables/prioritization_results.json"), "w"), indent=1, default=str)
print(json.dumps({k: v for k, v in out.items() if k not in ("A", "B_removed_reason_by_mirna")}, indent=1, default=str))
for r in rowsA:
    print(r["miRNA"], r["kept"], {k: (None if r[k] is None else (r[k]["logFC"], f'{r[k]["P"]:.2g}', r[k]["avail"], r[k]["passed"])) for k in ("ROSMAP", "GSE120584", "GSE215789")})
