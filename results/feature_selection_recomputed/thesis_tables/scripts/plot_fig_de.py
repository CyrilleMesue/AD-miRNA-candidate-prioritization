"""Results 3.1.7 figures: (A) volcano plots of the four miRNA differential-expression
analyses, (B) AD vs control expression of the three experimental candidates.
Usage: python plot_fig_de.py <thesis fig dir>"""
import sys
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[4]
OUT = Path(sys.argv[1])
C = ["hsa-miR-324-3p", "hsa-miR-574-3p", "hsa-miR-769-3p"]
ACC, GREY, INK, MUTED = "#e8762b", "#cfcfcb", "#1a1a1a", "#6b6b67"
UP, DN, CTRLC = "#c2416b", "#3987e5", "#a3a39f"
plt.rcParams.update({"font.family": "serif", "font.serif": ["Nimbus Roman", "Times New Roman", "Liberation Serif", "DejaVu Serif"],
                     "font.size": 8, "mathtext.fontset": "stix", "axes.linewidth": 0.6, "axes.edgecolor": MUTED,
                     "xtick.color": MUTED, "ytick.color": MUTED, "xtick.major.width": 0.6,
                     "ytick.major.width": 0.6, "axes.labelcolor": INK, "pdf.fonttype": 42})

I = ROOT / "data" / "interim"
DE = [("ROSMAP", "brain", "rosmap_results_miRNA_Limma.csv", "P.Value"),
      ("GSE120584", "serum", "GSE120584_results_miRNA_Limma.csv", "P.Value"),
      ("GSE215789", "plasma", "GSE215789_results_miRNA_edgeR.csv", "PValue")]
tabs = {}
for name, tissue, f, pc in DE:
    d = pd.read_csv(I / f, index_col=0)
    tabs[name] = pd.DataFrame({"logFC": d["logFC"], "p": d[pc]})

# ---------------- A: volcano ----------------
fig, axes = plt.subplots(1, 3, figsize=(6.3, 2.7))
LETTERS = "ABC"
LAB = {  # label offsets in points, per panel
    "ROSMAP": {"hsa-miR-324-3p": (-14, 14), "hsa-miR-574-3p": (22, 26), "hsa-miR-769-3p": (24, 10)},
    "GSE120584": {"hsa-miR-324-3p": (26, -6), "hsa-miR-574-3p": (-14, 22), "hsa-miR-769-3p": (24, 14)},
    "GSE215789": {"hsa-miR-574-3p": (18, 16)}}
for k, (ax, (name, tissue, f, pc)) in enumerate(zip(axes.flat, DE)):
    t = tabs[name]
    y = -np.log10(t["p"])
    sig = (t["p"] < 0.05) & (t["logFC"].abs() > 0.1)
    ax.scatter(t["logFC"][~sig], y[~sig], s=4, c=GREY, lw=0, alpha=0.6, rasterized=True)
    up, dn = sig & (t["logFC"] > 0), sig & (t["logFC"] < 0)
    ax.scatter(t["logFC"][up], y[up], s=5, c=UP, lw=0, alpha=0.8, rasterized=True)
    ax.scatter(t["logFC"][dn], y[dn], s=5, c=DN, lw=0, alpha=0.8, rasterized=True)
    for x in (-0.1, 0.1):
        ax.axvline(x, color=MUTED, lw=0.6, ls=(0, (3, 2)))
    ax.axhline(-np.log10(0.05), color=MUTED, lw=0.6, ls=(0, (3, 2)))
    xl = np.nanmax(np.abs(t["logFC"])) * 1.08
    ax.set_xlim(-xl, xl)
    ymax = y.max() * 1.1
    ax.set_ylim(0, ymax)
    present = [c for c in C if c in t.index]
    for c in present:
        xx, yy = t.at[c, "logFC"], -np.log10(t.at[c, "p"])
        ax.scatter([xx], [yy], s=34, c=ACC, edgecolor=INK, lw=0.8, zorder=5)
        ox, oy = LAB[name][c]
        ax.annotate(c.replace("hsa-", ""), (xx, yy), xytext=(ox, oy), textcoords="offset points",
                    fontsize=7, fontweight="bold", color=INK, ha="left" if ox > 0 else "right",
                    va="center", arrowprops=dict(arrowstyle="-", lw=0.5, color=MUTED,
                                                 shrinkA=0, shrinkB=3))
    missing = [c for c in C if c not in t.index]
    if missing:
        ax.text(0.03, 0.97, "Not tested (expression filter):\n" +
                "\n".join(m.replace("hsa-", "") for m in missing),
                transform=ax.transAxes, fontsize=6.5, color=MUTED, va="top", ha="left",
                bbox=dict(facecolor="white", edgecolor="none", pad=1.5), zorder=6)
    lab = f"{name} ({tissue})" if tissue else name
    ax.set_title(f"{LETTERS[k]}   {lab}\n{len(t):,} miRNAs", loc="left", fontsize=7.5, color=INK)
    ax.set_xlabel(r"$\log_2$ fold change (AD vs control)")
    ax.set_ylabel(r"$-\log_{10}\,p$")
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
fig.tight_layout(w_pad=1.0)
fig.savefig(OUT / "de_volcano.pdf", dpi=300)
fig.savefig(OUT / "de_volcano.png", dpi=150)
plt.close(fig)

# ---------------- B: candidate expression ----------------
P = ROOT / "data" / "processed"
ros = pd.read_csv(P / "ROSMAP_brain_norm.csv")
ros["Diagnosis"] = ros["Diagnosis"].apply(lambda x: "AD" if "AD" in str(x) else x)  # AD+ counted as AD, as in ranking and DE
gse = pd.read_csv(P / "GSE120584_serum_norm.csv")
cnt = pd.read_csv(I / "de_inputs" / "GSE215789_counts.csv", index_col=0)
smp = pd.read_csv(I / "de_inputs" / "GSE215789_samples.csv").set_index("Sample ID")
lib = cnt.sum(axis=0)
lcpm = np.log2((cnt + 0.5).div(lib + 1, axis=1) * 1e6).T
lcpm["Diagnosis"] = smp.loc[lcpm.index, "group"].replace({"CTRL": "Control"}).values

def groups(df, c):
    return [df.loc[df["Diagnosis"] == g, c].dropna().values for g in ("Control", "AD")]

DATA = [("ROSMAP", "brain, normalized log$_2$", ros, "ROSMAP"),
        ("GSE120584", "serum, normalized log$_2$", gse, "GSE120584"),
        ("GSE215789", "plasma, log$_2$ CPM", lcpm, "GSE215789")]
rng = np.random.default_rng(1)
fig, axes = plt.subplots(3, 3, figsize=(6.3, 6.2))
for i, c in enumerate(C):
    for j, (name, unit, df, key) in enumerate(DATA):
        ax = axes[i, j]
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        if c not in df.columns or c not in tabs[key].index:
            ax.text(0.5, 0.5, "Not tested\n(expression filter)", transform=ax.transAxes,
                    ha="center", va="center", fontsize=7, color=MUTED)
            ax.set_xticks([]); ax.set_yticks([])
            for s in ("left", "bottom"):
                ax.spines[s].set_visible(False)
        else:
            g = groups(df, c)
            for x, v in enumerate(g):
                a = 0.25 if len(v) > 300 else 0.6
                ax.scatter(x + rng.uniform(-0.22, 0.22, len(v)), v, s=2.5 if len(v) > 300 else 5,
                           c=CTRLC if x == 0 else ACC, alpha=a, lw=0, rasterized=True, zorder=1)
            bp = ax.boxplot(g, positions=[0, 1], widths=0.5, showfliers=False, patch_artist=True,
                            medianprops=dict(color=INK, lw=1.0), boxprops=dict(lw=0.7, color=INK),
                            whiskerprops=dict(lw=0.7, color=INK), capprops=dict(lw=0.7, color=INK))
            for b in bp["boxes"]:
                b.set_facecolor("none")
            ax.set_xticks([0, 1])
            ax.set_xticklabels([f"Control\n({len(g[0])})", f"AD\n({len(g[1])})"], fontsize=7)
            r = tabs[key].loc[c]
            p = r["p"]
            if p >= 0.1:
                ps = f"{p:.2f}"
            elif p >= 0.001:
                ps = f"{p:.2g}"
            else:
                m, ex = f"{p:.1e}".split("e")
                ps = m + r"$\times10^{" + str(int(ex)) + "}$"
            ax.set_title(f"logFC {r['logFC']:+.3f}".replace("-", "\u2212") + f", $p$ = {ps}", fontsize=7, color=INK, loc="center")
            ax.tick_params(axis="y", labelsize=7)
            ax.set_xlim(-0.6, 1.6)
        if j == 0:
            ax.set_ylabel(c.replace("hsa-", ""), fontweight="bold", fontsize=8)
        if i == 0:
            ax.annotate(f"{name}\n{unit}", xy=(0.5, 1.28), xycoords="axes fraction", ha="center",
                        va="bottom", fontsize=8, color=INK)
fig.tight_layout(h_pad=1.4, w_pad=1.2, rect=(0, 0, 1, 0.96))
fig.savefig(OUT / "de_candidate_boxplots.pdf", dpi=300)
fig.savefig(OUT / "de_candidate_boxplots.png", dpi=150)
print("saved")
