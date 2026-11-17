"""Method-agreement figures for Results 3.1.2 and 3.1.3.
A: Spearman correlation between the 13 standalone rankings in each dataset
   (validation/individual_method_agreement.csv, spearman_average_ties).
B: Spearman correlation between the 12 ensemble rankings of the 58 shared miRNAs
   (ensemble scores, oriented so larger = higher priority, ties averaged) and the
   number of shared top-20 miRNAs. Reproduces the values quoted in ch3.
Usage: python plot_fig_method_agreement.py <out dir>"""
import itertools, sys
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm, Normalize
import numpy as np
import pandas as pd

R = Path(__file__).resolve().parents[2]
OUT = Path(sys.argv[1])
plt.rcParams.update({"font.family": "serif", "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
                     "font.size": 7.5, "pdf.fonttype": 42})
# diverging blue <-> red, grey midpoint (dataviz reference ramps)
DIV = LinearSegmentedColormap.from_list("div", ["#104281", "#3987e5", "#9ec5f4", "#f0efec",
                                                 "#f4a99b", "#e0533d", "#9a2a1a"])
SEQ = LinearSegmentedColormap.from_list("seq", ["#f0efec", "#cde2fb", "#6da7ec", "#256abf", "#104281"])
INK, MUTED = "#1a1a1a", "#8a8a86"


def draw(ax, M, labels, cmap, norm, fmt, lower=True, na=None, show_y=True):
    n = len(labels)
    for i in range(n):
        for j in range(n):
            if lower and j > i:
                continue
            v = M[i, j]
            if na is not None and (labels[i] in na or labels[j] in na):
                ax.add_patch(plt.Rectangle((j + .04, i + .04), .92, .92, fc="#ffffff", ec="#c9c9c5",
                                           lw=.4, hatch="////"))
                continue
            if i == j and lower:
                ax.add_patch(plt.Rectangle((j + .04, i + .04), .92, .92, fc="#e4e4e0", ec="none"))
                continue
            c = cmap(norm(v))
            ax.add_patch(plt.Rectangle((j + .04, i + .04), .92, .92, fc=c, ec="none"))
            lum = 0.299 * c[0] + 0.587 * c[1] + 0.114 * c[2]
            ax.text(j + .5, i + .53, fmt(v), ha="center", va="center", fontsize=5.6,
                    color="white" if lum < 0.5 else INK)
    # method names on the diagonal: row i ends at column i, so one label names both
    for i, l in enumerate(labels):
        ax.text(i + 1.15, i + .53, l, ha="left", va="center", fontsize=7, color=INK)
    ax.set_xlim(0, n + (7.6 if n == 12 else 5.2)); ax.set_ylim(n, 0)
    ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)
    ax.set_aspect("equal")


# ---------------- A: standalone ----------------
M13 = ["lasso", "ridge", "elasticnet", "shap", "lime", "xgb_feature_importance",
       "randomforest_feature_importance", "rf_permutation_feature_importance",
       "xgb_permutation_feature_importance", "t_test", "boruta", "mannwhitneyu", "svm_rfe"]
N13 = {"lasso": "LASSO", "ridge": "Ridge", "elasticnet": "Elastic net", "shap": "SHAP", "lime": "LIME",
       "xgb_feature_importance": "XGBoost importance", "randomforest_feature_importance": "RF importance",
       "rf_permutation_feature_importance": "RF permutation",
       "xgb_permutation_feature_importance": "XGBoost permutation", "t_test": "$t$-test",
       "boruta": "Boruta", "mannwhitneyu": "Mann–Whitney U", "svm_rfe": "SVM-RFE"}
a = pd.read_csv(R / "validation" / "individual_method_agreement.csv")
DS = ["ROSMAP", "GSE120584", "GSE215789"]
norm = TwoSlopeNorm(vcenter=0, vmin=-1, vmax=1)
fig, axes = plt.subplots(1, 3, figsize=(6.3, 3.75))
labs = [N13[m] for m in M13]
fmt13 = lambda v: f"{v:.2f}".replace("0.", ".").replace("-.", "\u2212.")
for k, (ax, d) in enumerate(zip(axes, DS)):
    g = a[a.dataset == d]
    M = pd.DataFrame(np.eye(13), index=M13, columns=M13)
    for _, r in g.iterrows():
        M.loc[r.method_a, r.method_b] = M.loc[r.method_b, r.method_a] = r.spearman_average_ties
    s = g.spearman_average_ties
    for i in range(13):
        for j in range(i + 1):
            if i == j:
                ax.add_patch(plt.Rectangle((j + .05, i + .05), .9, .9, fc="#e4e4e0", ec="none"))
                continue
            v = M.values[i, j]; c = DIV(norm(v))
            ax.add_patch(plt.Rectangle((j + .05, i + .05), .9, .9, fc=c, ec="none"))
            lum = 0.299 * c[0] + 0.587 * c[1] + 0.114 * c[2]
            ax.text(j + .5, i + .54, fmt13(v), ha="center", va="center", fontsize=3.9,
                    color="white" if lum < 0.5 else INK)
    ax.set_xlim(0, 13); ax.set_ylim(13, 0); ax.set_aspect("equal")
    ax.set_xticks(np.arange(13) + .5); ax.set_xticklabels(labs, rotation=90, fontsize=5.6)
    ax.set_yticks(np.arange(13) + .5)
    ax.set_yticklabels(labs if k == 0 else [], fontsize=5.6)
    ax.tick_params(length=0, pad=1.5)
    for sp in ax.spines.values():
        sp.set_visible(False)
    ax.set_title(f"{'ABC'[k]}   {d}\n(median $\\rho$ = {s.median():.2f})", loc="left", fontsize=7.5,
                 fontweight="bold", pad=3)
fig.subplots_adjust(left=0.165, right=0.995, top=0.91, bottom=0.36, wspace=0.06)
cax = fig.add_axes([0.36, 0.105, 0.42, 0.018])
cb = fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=DIV), cax=cax, orientation="horizontal",
                  ticks=[-1, -.5, 0, .5, 1])
cb.outline.set_visible(False); cb.ax.tick_params(length=2, labelsize=6.5)
cb.set_label("Spearman correlation ($\\rho$) between full rankings", fontsize=7)
fig.savefig(OUT / "fs_method_agreement.pdf", bbox_inches="tight", pad_inches=0.03); fig.savefig(OUT / "fs_method_agreement.png", dpi=150, bbox_inches="tight", pad_inches=0.03)
plt.close(fig)

# ---------------- B: ensembles ----------------
E12 = ["mean_rank", "min_rank", "median_rank", "rra_rank", "geom.mean_rank", "stuart_rank", "mra_rank",
       "mean_weight", "max_weight", "median_weight", "geom.mean_weight", "ta_weight"]
N12 = {"mean_rank": "Mean rank", "min_rank": "Minimum rank", "median_rank": "Median rank", "rra_rank": "RRA",
       "geom.mean_rank": "Geometric-mean rank", "stuart_rank": "Stuart", "mra_rank": "MRA",
       "mean_weight": "Mean weight", "max_weight": "Maximum weight", "median_weight": "Median weight",
       "geom.mean_weight": "Geometric-mean weight", "ta_weight": "Threshold algorithm"}
e = pd.read_csv(R / "ensembles" / "ensemble_rankings.csv")
S = e.pivot(index="feature_id", columns="aggregator", values="ensemble_score")[E12]
asc = e.groupby("aggregator").sort_ascending.first()
for c in E12:
    if asc[c]:
        S[c] = -S[c]
C = S.corr("spearman").values
K = e.pivot(index="feature_id", columns="aggregator", values="ensemble_rank")[E12] <= 20
O = np.array([[(K[x] & K[y]).sum() for y in E12] for x in E12], dtype=float)
lab = [N12[m] for m in E12]
fig, (a1, a2) = plt.subplots(1, 2, figsize=(6.3, 3.0))
draw(a1, C, lab, DIV, norm, lambda v: f"{v:.2f}".replace("0.", ".").replace("-.", "−."), na={"Maximum weight"})
draw(a2, O, lab, SEQ, Normalize(0, 20), lambda v: f"{int(v)}", na={"Maximum weight"}, show_y=False)
for ax in (a1, a2):
    ax.plot([0, 7], [7, 7], color=INK, lw=.7); ax.plot([7, 7], [7, 12], color=INK, lw=.7)
a1.set_title("A   Spearman $\\rho$ (58 miRNAs)", loc="left", fontsize=8.5, fontweight="bold")
a2.set_title("B   Shared top-20 miRNAs", loc="left", fontsize=8.5, fontweight="bold")
fig.subplots_adjust(left=0.01, right=0.99, top=0.92, bottom=0.20, wspace=0.03)
for ax, cm, nm, ticks, lbl in [(a1, DIV, norm, [-1, 0, 1], "$\\rho$"), (a2, SEQ, Normalize(0, 20), [0, 10, 20], "Shared miRNAs")]:
    p = ax.get_position()
    cax = fig.add_axes([p.x0 + p.width * .05, 0.13, p.width * .40, 0.022])
    cb = fig.colorbar(plt.cm.ScalarMappable(norm=nm, cmap=cm), cax=cax, orientation="horizontal", ticks=ticks)
    cb.outline.set_visible(False); cb.ax.tick_params(length=2, labelsize=7); cb.set_label(lbl, fontsize=7.5)
fig.savefig(OUT / "ensemble_method_agreement.pdf"); fig.savefig(OUT / "ensemble_method_agreement.png", dpi=150)
print("saved")
