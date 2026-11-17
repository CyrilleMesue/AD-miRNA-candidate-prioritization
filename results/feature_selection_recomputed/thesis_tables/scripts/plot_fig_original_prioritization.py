"""Figure for Results 3.1.5: the 41 miRNAs of the original prioritization (12 ensemble
rankings of the 58 shared miRNAs), placed by recurrence (number of top-20 lists) and median
rank, coloured by how far each got in the ORIGINAL procedure.

Original rules (before the 2026-11-06 revision): HMDD score > 1 removes; an untested miRTarBase
interaction with a CTD/GeneCards AD gene is required, looked up with -3p/-5p forms only
(get_mirna_versions without the plain name); |logFC| > 0.1 and p < 0.05 in at least one of ROSMAP, GSE120584 or GSE215789
(uniform rule, thesis decision 2026-11-07). Asserts the result equals the saved
original shortlist results/candidates_uniform_de/prioritized_mirnas.txt.

Usage: python plot_fig_original_prioritization.py <figure dir>
"""
import os
import sys
from pathlib import Path

os.environ.setdefault("MPLBACKEND", "Agg")
ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "src"))
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.transforms import Bbox

from prioritization.candidate_selection import load_annotation_tables, load_differential_tables
from prioritization.historical import get_mirna_versions, sort_dict_features
from scripts.run_pipeline import load_config, resolve

out = Path(sys.argv[1])
cfg = load_config(str(ROOT / "config/analysis_config.yaml"))
db = load_annotation_tables({k: resolve(ROOT, v) for k, v in cfg["annotation"].items()})
de = load_differential_tables({k: resolve(ROOT, v) for k, v in cfg["differential"].items()})
e = pd.read_csv(ROOT / "results/feature_selection_recomputed/ensembles/ensemble_rankings.csv")
e = e[e.aggregator != "max_weight"]  # maximum weight excluded after Results 3.1.4 (all 58 miRNAs tied; thesis decision 2026-11-16)
CAND = ["hsa-mir-324-3p", "hsa-mir-574-3p", "hsa-mir-769-3p"]

rows = []
for a, g in e.groupby("aggregator", sort=False):
    top = sort_dict_features(dict(zip(g.feature_id, g.ensemble_score)), ascending="rank" in a,
                             return_with_values=False)[:20]
    rows += [(a, r, m.lower()) for r, m in enumerate(top, 1)]
pos = pd.DataFrame(rows, columns=["agg", "rank", "miRNA"])
per = pos.groupby("miRNA").agg(n=("agg", "nunique"), med=("rank", "median"), best=("rank", "min")).reset_index()

known = set(db["hmdd"].Entity.str.lower())
ad = set(db["ctd"].Entity.to_list()[:100]) | set(db["genecards"].Entity.to_list()[:100])
mt = db["mirtarbase"]; val = mt.iloc[:, 2:5].sum(axis=1)
RAWDE = {"ROSMAP": ("rosmap_results_miRNA_Limma.csv", "P.Value"), "GSE120584": ("GSE120584_results_miRNA_Limma.csv", "P.Value"),
         "GSE215789": ("GSE215789_results_miRNA_edgeR.csv", "PValue")}
fc = {}
for k, (f_, pc) in RAWDE.items():
    t = pd.read_csv(ROOT / "data/interim" / f_).rename(columns={"Unnamed: 0": "id"})
    t["id"] = t.id.astype(str).str.lower(); t = t.drop_duplicates("id")
    fc[k] = {i: (lf, pv) for i, lf, pv in zip(t.id, t.logFC, t[pc])}


def fate(m):
    vb = [v.lower() for v in get_mirna_versions(m, include_base=True)]
    if any(v in known for v in vb):
        return "hmdd"
    vn = [v.lower() for v in get_mirna_versions(m)]  # original lookup: -3p/-5p forms only
    if not ((mt.miRNA.isin(vn)) & (mt.Gene.isin(ad)) & (val == 0)).any():
        return "target"
    ok = False
    for t in fc.values():
        hit = next((v for v in vb if v in t), None)
        ok |= hit is not None and abs(t[hit][0]) > 0.1 and t[hit][1] < 0.05
    if not ok:
        return "fc"
    return "sel" if m in CAND else "short"


per["fate"] = per.miRNA.map(fate)
saved = set((ROOT / "results/candidates_uniform_de/prioritized_mirnas.txt").read_text().split())
assert set(per[per.fate.isin(["short", "sel"])].miRNA) == saved, "does not reproduce the original shortlist"
cnt = per.fate.value_counts().to_dict()
assert cnt == {"hmdd": 13, "fc": 12, "short": 5, "target": 5, "sel": 3}, cnt

plt.rcParams.update({"font.family": "STIXGeneral", "mathtext.fontset": "stix", "font.size": 8,
                     "pdf.fonttype": 42, "savefig.dpi": 300, "axes.linewidth": 0.6})
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"
COL = {"hmdd": "#86b6ef", "target": "#3987e5", "fc": "#1c5cab", "short": "#0d366b", "sel": "#eb6834"}
LAB = {"hmdd": "Removed: more than one HMDD AD record", "target": "Removed: no untested interaction with an AD gene",
       "fc": "Removed: no dataset with $|\\log_2\\mathrm{FC}| > 0.1$ and $p < 0.05$", "short": "Shortlisted, not selected",
       "sel": "Selected for experiments"}

# nudge coincident points apart horizontally
per["x"] = per.med.astype(float)
for (_, _), g in per.groupby(["n", "med"]):
    if len(g) > 1:
        per.loc[g.index, "x"] += np.linspace(-0.32, 0.32, len(g))
per["y"] = per.n.astype(float)

fig, ax = plt.subplots(figsize=(6.3, 4.5))
ax.set_xlim(0.2, 21); ax.set_ylim(0.2, 13.2)
ax.set_xticks([1, 5, 10, 15, 20]); ax.set_yticks(range(1, 13))
ax.tick_params(labelsize=7, length=2.5, width=0.6, color=INK2)
ax.grid(color=GRID, lw=0.5); ax.set_axisbelow(True)
for s in ("top", "right"): ax.spines[s].set_visible(False)
ax.set_xlabel("Median rank within the top-20 lists containing the miRNA (1 = highest)", size=7.6)
ax.set_ylabel("Number of ensemble top-20 lists (of 11)", size=7.6)
order = ["hmdd", "target", "fc", "short", "sel"]
for k in order:
    d = per[per.fate == k]
    ax.scatter(d.x, d.y, s=58 if k == "sel" else 44, c=COL[k], edgecolors=INK if k == "sel" else "white",
               linewidths=0.9 if k == "sel" else 0.6, zorder=3 + order.index(k))

# greedy label placement (display coordinates)
fig.canvas.draw(); rend = fig.canvas.get_renderer()
pts = ax.transData.transform(per[["x", "y"]].values)
pt_boxes = [Bbox.from_bounds(px - 4, py - 4, 8, 8) for px, py in pts]
placed = []
offsets = [(5, 0, "left", "center"), (-5, 0, "right", "center"), (0, 6, "center", "bottom"),
           (0, -6, "center", "top"), (5, 5, "left", "bottom"), (-5, 5, "right", "bottom"),
           (5, -5, "left", "top"), (-5, -5, "right", "top"), (9, 0, "left", "center"), (-9, 0, "right", "center"),
           (0, 11, "center", "bottom"), (0, -11, "center", "top")]
prio = per.assign(c=per.miRNA.isin(CAND)).sort_values(["c", "n"], ascending=False).index
for i in prio:
    r = per.loc[i]; cand = r.miRNA in CAND
    txt = r.miRNA.replace("hsa-mir-", "miR-")
    best, best_ov = None, None
    MANUAL = {"hsa-mir-324-5p": [(0, -6, "center", "top")]}  # avoids touching the miR-543 label
    for dx, dy, ha, va in MANUAL.get(r.miRNA, offsets):
        t = ax.annotate(txt, (r.x, r.y), xytext=(dx, dy), textcoords="offset points", ha=ha, va=va,
                        size=6.6 if cand else 5.6, color=INK if cand else INK2,
                        weight="bold" if cand else "normal", zorder=10)
        bb = t.get_window_extent(rend).expanded(1.16, 1.25)
        inside = ax.get_window_extent(rend).contains(bb.x0, bb.y0) and ax.get_window_extent(rend).contains(bb.x1, bb.y1)
        ov = sum(bb.overlaps(b) for b in placed) + sum(bb.overlaps(b) for j, b in enumerate(pt_boxes) if j != list(per.index).index(i)) + (0 if inside else 5)
        if best_ov is None or ov < best_ov:
            if best is not None: best[0].remove()
            best, best_ov = (t, bb), ov
        else:
            t.remove()
        if ov == 0:
            break
    placed.append(best[1])

hs = [Line2D([], [], marker="o", ls="", ms=6, mfc=COL[k], mec=INK if k == "sel" else "white", mew=0.8,
             label=f"{LAB[k]} ({int((per.fate == k).sum())})") for k in order]
ax.legend(handles=hs, loc="upper center", bbox_to_anchor=(0.5, -0.13), ncol=2, frameon=False, fontsize=6.6,
          handletextpad=0.3, columnspacing=1.6)
fig.savefig(out / "fig_original_prioritization.pdf", bbox_inches="tight", pad_inches=0.03)
fig.savefig(out / "fig_original_prioritization.png", bbox_inches="tight", pad_inches=0.03)
per.to_csv(out / "original_prioritization_mirnas.csv", index=False)
print(cnt)
