"""Figure B: flow of miRNAs through candidate prioritization (48 ensemble rankings).
Input: results/candidates_48_lists/mirna_fate.csv (scripts/09_prioritize_48_lists.py).
Usage: python -I plot_fig_b_sankey_48.py <candidates_48_lists dir> <figure dir>"""
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.patches import PathPatch, Rectangle
from matplotlib.path import Path as MPath

src, out = Path(sys.argv[1]), Path(sys.argv[2])
f = pd.read_csv(src / "mirna_fate.csv")
plt.rcParams.update({"font.family": "STIXGeneral", "mathtext.fontset": "stix", "font.size": 8,
                     "pdf.fonttype": 42, "savefig.dpi": 300})
INK, INK2, EXIT = "#0b0b0b", "#52514e", "#c9c8c3"
CAND = ["hsa-mir-324-3p", "hsa-mir-574-3p", "hsa-mir-769-3p"]


def group(row):
    if row.miRNA in CAND:
        return "cand"
    s = set(row.sources.split(";"))
    if "shared58" in s:
        return "shared58"
    # miR-1229-3p is in GSE120584 and GSE215789 lists; it passed in GSE120584.
    return "GSE120584" if "GSE120584" in s else s.pop()


f["group"] = f.apply(group, axis=1)
# stack order top -> bottom; colours validated as an adjacent categorical set (blue, aqua, violet, yellow)
GROUPS = ["cand", "shared58", "GSE120584", "ROSMAP", "GSE215789"]
COL = {"cand": "#eb6834", "shared58": "#2a78d6", "GSE120584": "#1baf7a", "ROSMAP": "#6250d6", "GSE215789": "#eda100"}
NAME = {"cand": "Selected candidates", "shared58": "58 shared miRNAs (11 lists)",
        "GSE120584": "GSE120584 serum (11 lists)", "ROSMAP": "ROSMAP brain (11 lists)",
        "GSE215789": "GSE215789 plasma (11 lists)"}
alive = [
    lambda x: True,
    lambda x: x != "Removed: HMDD",
    lambda x: x in ("Removed: fold change", "Shortlisted", "Selected"),
    lambda x: x in ("Shortlisted", "Selected"),
    lambda x: x == "Selected",
]
n = [{g: int(((f.group == g) & f.fate.map(a)).sum()) for g in GROUPS} for a in alive]
tot = [sum(d.values()) for d in n]
STAGE = ["Top-20 pool", "After HMDD\nscreen", "After AD-target\nscreen", "After fold-change\nfilter",
         "Selected"]
EXITS = ["more than one\nHMDD AD record", "no untested\ninteraction with\nan AD gene",
         "$|\\log_2\\mathrm{FC}| \\leq 0.1$ or\n$p \\geq 0.05$", "literature review"]

fig, ax = plt.subplots(figsize=(6.3, 4.6))
xs = [0.0, 1.45, 2.9, 4.35, 5.8]
nw, TOP, gap = 0.09, 166.0, 1.2
EXIT_BASE = -78


from matplotlib.colors import to_rgb


def tint(c, t=0.45):
    r, g, b = to_rgb(c)
    return (1 - t + t * r, 1 - t + t * g, 1 - t + t * b)


def band(x0, a0, b0, x1, a1, b1, color, alpha, z=1):
    xm = (x0 + x1) / 2
    v = [(x0, a0), (xm, a0), (xm, a1), (x1, a1), (x1, b1), (xm, b1), (xm, b0), (x0, b0), (x0, a0)]
    c = [MPath.MOVETO, MPath.CURVE4, MPath.CURVE4, MPath.CURVE4, MPath.LINETO,
         MPath.CURVE4, MPath.CURVE4, MPath.CURVE4, MPath.CLOSEPOLY]
    ax.add_patch(PathPatch(MPath(v, c), fc=color, ec="none", alpha=alpha, lw=0, zorder=z))


# segment positions: groups stacked from TOP downward with a small gap between groups
seg = []
for i, d in enumerate(n):
    y, s = TOP, {}
    for g in GROUPS:
        h = d[g]
        s[g] = (y, y - h)
        if h:
            y -= h + gap
    seg.append(s)

for i in range(4):
    x0, x1 = xs[i] + nw / 2, xs[i + 1] - nw / 2
    xe = xs[i] + 0.62 * (xs[i + 1] - xs[i])
    ey = EXIT_BASE + (tot[i] - tot[i + 1])  # top of exit stack
    for g in GROUPS:
        a0, b0 = seg[i][g]
        keep, lost = n[i + 1][g], n[i][g] - n[i + 1][g]
        if keep:
            band(x0, a0, a0 - keep, x1, seg[i + 1][g][0], seg[i + 1][g][1],
                 COL[g] if g == "cand" else tint(COL[g]), 1.0, 3 if g == "cand" else 2)
        if lost:
            band(x0, a0 - keep, b0, xe, ey, ey - lost, EXIT, 0.6, 1)
            ey -= lost
    ne = tot[i] - tot[i + 1]
    ax.add_patch(Rectangle((xe, EXIT_BASE), 0.05, ne, fc="#9a9994", ec="none"))
    ax.text(xe + 0.025, EXIT_BASE - 3, f"{ne} " + ("not selected" if i == 3 else "removed"), ha="center", va="top", size=7.4, weight="bold", color=INK)
    ax.text(xe + 0.025, EXIT_BASE - 13, EXITS[i], ha="center", va="top", size=6.6, color=INK2, linespacing=1.1)

for i, s in enumerate(seg):
    for g in GROUPS:
        a, b = s[g]
        if a > b:
            ax.add_patch(Rectangle((xs[i] - nw / 2, b), nw, a - b, fc=COL[g], ec="white", lw=0.5, zorder=3))
    ax.text(xs[i], TOP + 4, f"{tot[i]}", ha="center", va="bottom", size=10, weight="bold", color=INK)
    ax.text(xs[i], TOP + 19, STAGE[i], ha="center", va="bottom", size=7.6, color=INK2, linespacing=1.05)

# direct labels for the sources (left) with pool counts
for g in GROUPS:
    a, b = seg[0][g]
    lab = NAME[g] if g != "cand" else "Selected candidates (3)"
    cnt = f"{n[0][g]} miRNAs" + (" + 3 candidates" if g == "shared58" else "")
    ax.text(xs[0] - nw / 2 - 0.06, (a + b) / 2, f"{lab}\n{cnt}" if g != "cand" else lab,
            ha="right", va="center", size=6.8, color=INK, linespacing=1.05,
            weight="bold" if g == "cand" else "normal")
    ax.plot([xs[0] - nw / 2 - 0.05, xs[0] - nw / 2], [(a + b) / 2] * 2, color=COL[g], lw=1.2)
# survivors per source after the fold-change filter (right of column 4)
last = None
for g in GROUPS[1:]:
    a, b = seg[3][g]
    if a > b:
        y = (a + b) / 2 if last is None else min((a + b) / 2, last - 7)
        ax.text(xs[3] + nw / 2 + 0.05, y, str(n[3][g]), ha="left", va="center", size=6.6, color=INK2)
        last = y
ax.text(xs[4] + nw / 2 + 0.07, TOP - 1.5, "miR-324-3p\nmiR-574-3p\nmiR-769-3p", ha="left", va="top",
        size=7.4, weight="bold", color=INK, linespacing=1.15)

ax.set_xlim(-1.55, 6.65)
ax.set_ylim(EXIT_BASE - 36, TOP + 40)
ax.axis("off")
fig.savefig(out / "fig_B_sankey_48.pdf", bbox_inches="tight", pad_inches=0.03)
fig.savefig(out / "fig_B_sankey_48.png", bbox_inches="tight", pad_inches=0.03)
print(n)
