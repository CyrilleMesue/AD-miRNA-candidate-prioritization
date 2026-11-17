"""Results 3.1.4: positions of the 41 pooled miRNAs in the 12 ensemble rankings
of the 58 shared miRNAs. Usage: python plot_fig_ensemble_heatmap.py <out.pdf>"""
import sys
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, Normalize
import numpy as np
import pandas as pd

R = Path(__file__).resolve().parents[2]
e = pd.read_csv(R / "ensembles" / "ensemble_rankings.csv")
RANK = ["mean_rank", "geom.mean_rank", "stuart_rank", "mra_rank", "min_rank", "median_rank", "rra_rank"]
WT = ["mean_weight", "median_weight", "geom.mean_weight", "ta_weight", "max_weight"]
COLS = RANK + WT
TIED = {"min_rank", "median_rank", "rra_rank", "max_weight"}
NAME = {"mean_rank": "Mean rank", "geom.mean_rank": "Geometric-mean rank", "stuart_rank": "Stuart",
        "mra_rank": "MRA", "min_rank": "Minimum rank", "median_rank": "Median rank", "rra_rank": "RRA",
        "mean_weight": "Mean weight", "median_weight": "Median weight",
        "geom.mean_weight": "Geometric-mean weight", "ta_weight": "Threshold algorithm",
        "max_weight": "Maximum weight"}
CAND = {"hsa-miR-324-3p", "hsa-miR-574-3p", "hsa-miR-769-3p"}

P = e.pivot(index="feature_id", columns="aggregator", values="ensemble_rank")[COLS]
top = P <= 20
P = P[top.any(axis=1)]
top = top.loc[P.index]
n = top.sum(axis=1)
med = P.where(top).median(axis=1)
order = pd.DataFrame({"n": n, "med": med}).sort_values(["n", "med"], ascending=[False, True]).index
P, top, n = P.loc[order], top.loc[order], n.loc[order]

ramp = ["#0d366b", "#184f95", "#256abf", "#3987e5", "#6da7ec", "#9ec5f4", "#cde2fb"]
cmap = LinearSegmentedColormap.from_list("blue", ramp)
norm = Normalize(1, 20)
plt.rcParams.update({"font.family": "serif", "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
                     "font.size": 8})
fig, ax = plt.subplots(figsize=(6.3, 8.6))
nr, nc = P.shape
for i in range(nr):
    for j, c in enumerate(COLS):
        v = int(P.iat[i, j])
        if top.iat[i, j]:
            fc = cmap(norm(v)); tc = "white" if v <= 9 else "#1a1a1a"
        else:
            fc = "#f3f3f1"; tc = "#9a9a96"
        ax.add_patch(plt.Rectangle((j + 0.04, i + 0.04), 0.92, 0.92, facecolor=fc, edgecolor="none"))
        ax.text(j + 0.5, i + 0.52, str(v), ha="center", va="center", fontsize=6.3, color=tc)
    ax.text(nc + 0.35, i + 0.52, str(int(n.iat[i])), ha="left", va="center", fontsize=7, color="#1a1a1a")
ax.axvline(len(RANK), color="#1a1a1a", lw=0.8)
ax.set_xlim(0, nc + 1); ax.set_ylim(nr, 0)
ax.set_xticks(np.arange(nc) + 0.5)
ax.set_xticklabels([NAME[c] + ("$^{\\dagger}$" if c in TIED else "") for c in COLS], rotation=55,
                   ha="left", rotation_mode="anchor")
ax.xaxis.tick_top()
ax.set_yticks(np.arange(nr) + 0.5)
ax.set_yticklabels([s.replace("hsa-", "") for s in P.index])
for t in ax.get_yticklabels():
    if "hsa-" + t.get_text() in CAND:
        t.set_fontweight("bold")
ax.text(nc + 0.35, -0.4, "Lists", ha="left", va="bottom", fontsize=7, fontweight="bold")
ax.tick_params(length=0)
for s in ax.spines.values():
    s.set_visible(False)
cax = fig.add_axes([0.30, 0.045, 0.40, 0.012])
cb = fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), cax=cax, orientation="horizontal",
                  ticks=[1, 5, 10, 15, 20])
cb.outline.set_visible(False); cb.ax.tick_params(length=2, labelsize=7)
cb.set_label("Rank within the top 20 (grey cells: rank 21–58)", fontsize=7)
fig.subplots_adjust(left=0.17, right=0.95, top=0.80, bottom=0.08)
fig.savefig(sys.argv[1]); fig.savefig(sys.argv[1].replace(".pdf", ".png"), dpi=150)
print("saved", sys.argv[1])
