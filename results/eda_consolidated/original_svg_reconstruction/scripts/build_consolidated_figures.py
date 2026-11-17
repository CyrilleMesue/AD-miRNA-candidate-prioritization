#!/usr/bin/env python
"""Redraw the consolidated EDA figures from saved results.

PCA coordinates are read from the original SVG files. Other panels read
saved tables. This script does not refit PCA, rank features, or edit
original outputs.

python results/eda_consolidated/scripts/build_consolidated_figures.py
"""
from __future__ import annotations

import json
import re
import shutil
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
matplotlib.rcParams["pdf.fonttype"] = 42
matplotlib.rcParams["ps.fonttype"] = 42
matplotlib.rcParams["svg.fonttype"] = "none"
matplotlib.rcParams["font.size"] = 9

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
EDA = ROOT / "results/eda"
BIO = ROOT / "results/biological_eda"
OUT = ROOT / "results/eda_consolidated"
FINAL = OUT / "figures/final"
PREVIEW = OUT / "figures/previews"
PLOTS = OUT / "plotting_data"
TABLES = OUT / "tables"
REPORTS = OUT / "reports"

CONTROL = "#0072B2"
AD = "#E69F00"
DATASET_COLOR = {
    "ROSMAP": "#0072B2",
    "GSE120584": "#E69F00",
    "GSE215789": "#009E73",
}
CLASS_COLOR = {
    "Resolved mature": "#009E73",
    "Ambiguous": "#CC79A7",
    "Unresolved": "#56B4E9",
    "Non-human": "#666666",
}
DATASETS = ("ROSMAP", "GSE120584", "GSE215789")
DISPLAY = {
    "ROSMAP": "ROSMAP DLPFC",
    "GSE120584": "GSE120584 serum",
    "GSE215789": "GSE215789 plasma",
}
SVG = {
    "ROSMAP": EDA / "figures/pca_rosmap.svg",
    "GSE120584": EDA / "figures/pca_gse120584.svg",
    "GSE215789": EDA / "figures/pca_gse215789.svg",
}


def dress(ax, xlabel: str, ylabel: str) -> None:
    ax.set_xlabel(xlabel, fontsize=10, fontweight="bold")
    ax.set_ylabel(ylabel, fontsize=10, fontweight="bold")
    ax.tick_params(labelsize=8)
    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set_fontweight("bold")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def title(ax, letter: str, text: str) -> None:
    ax.set_title(f"{letter}   {text}", loc="left", fontsize=11, fontweight="bold", pad=8)


def save(fig, stem: str) -> None:
    FINAL.mkdir(parents=True, exist_ok=True)
    PREVIEW.mkdir(parents=True, exist_ok=True)
    for suffix, dpi in (("png", 300), ("svg", None), ("pdf", None)):
        path = FINAL / f"{stem}.{suffix}"
        fig.savefig(path, dpi=dpi, bbox_inches="tight", facecolor="white")
    shutil.copyfile(FINAL / f"{stem}.png", PREVIEW / f"{stem}.png")
    plt.close(fig)


def parse_number(token: str) -> float:
    return float(token.replace("−", "-").replace("–", "-").strip())


def parse_pca_svg(path: Path) -> pd.DataFrame:
    text = path.read_text(encoding="utf-8")
    body = text.split('<g id="matplotlib.axis_1">', 1)[0]

    def ticks(kind: str) -> list[tuple[float, float]]:
        found = []
        for match in re.finditer(rf'<g id="{kind}tick_\d+">.*?<text[^>]*>([^<]+)</text>', text, flags=re.S):
            chunk = match.group(0)
            use = re.search(r'x="([0-9.]+)" y="([0-9.]+)"', chunk)
            if not use:
                continue
            pixel = float(use.group(1) if kind == "x" else use.group(2))
            found.append((pixel, parse_number(match.group(1))))
        return found

    xticks = ticks("x")
    yticks = ticks("y")
    if len(xticks) < 2 or len(yticks) < 2:
        raise RuntimeError(f"Could not read axis ticks from {path}")
    x_slope, x_intercept = np.polyfit([item[0] for item in xticks], [item[1] for item in xticks], 1)
    y_slope, y_intercept = np.polyfit([item[0] for item in yticks], [item[1] for item in yticks], 1)

    rows = []
    for match in re.finditer(r'<use [^>]*x="([0-9.]+)" y="([0-9.]+)" style="([^"]+)"', body):
        style = match.group(3).lower()
        pc1 = x_slope * float(match.group(1)) + x_intercept
        pc2 = y_slope * float(match.group(2)) + y_intercept
        if "fill: #0072b2" in style:
            rows.append({"pc1": pc1, "pc2": pc2, "diagnosis": "Control", "flagged": 0})
        elif "fill: #e69f00" in style:
            rows.append({"pc1": pc1, "pc2": pc2, "diagnosis": "AD", "flagged": 0})
        elif "fill: none" in style and "stroke: #000000" in style:
            rows.append({"pc1": pc1, "pc2": pc2, "diagnosis": "", "flagged": 1})
    frame = pd.DataFrame(rows)
    if frame.empty:
        raise RuntimeError(f"No PCA points in {path}")
    return frame


def tick_residual(path: Path) -> float:
    text = path.read_text(encoding="utf-8")

    def ticks(kind: str) -> list[tuple[float, float]]:
        found = []
        for match in re.finditer(rf'<g id="{kind}tick_\d+">.*?<text[^>]*>([^<]+)</text>', text, flags=re.S):
            chunk = match.group(0)
            use = re.search(r'x="([0-9.]+)" y="([0-9.]+)"', chunk)
            if use:
                pixel = float(use.group(1) if kind == "x" else use.group(2))
                found.append((pixel, parse_number(match.group(1))))
        return found

    residuals = []
    for kind in ("x", "y"):
        pairs = ticks(kind)
        slope, intercept = np.polyfit([item[0] for item in pairs], [item[1] for item in pairs], 1)
        for pixel, value in pairs:
            residuals.append(abs(slope * pixel + intercept - value))
    return float(max(residuals))


def mark_flagged(points: pd.DataFrame, outliers: pd.DataFrame, dataset: str) -> tuple[pd.DataFrame, float]:
    rings = points[points["flagged"] == 1][["pc1", "pc2"]].to_numpy(dtype=float)
    filled = points[points["flagged"] == 0].copy()
    saved = outliers[(outliers["row_type"] == "sample") & (outliers["dataset"] == dataset)]
    saved_xy = saved[["pc1", "pc2"]].to_numpy(dtype=float)
    if len(rings) != len(saved):
        raise RuntimeError(f"{dataset} SVG has {len(rings)} rings and the table has {len(saved)} flagged samples")
    errors = []
    nearest = np.full(len(filled), 0, dtype=int)
    for x_value, y_value in saved_xy:
        distance = np.hypot(rings[:, 0] - x_value, rings[:, 1] - y_value)
        errors.append(float(distance.min()))
        filled_distance = np.hypot(filled["pc1"] - x_value, filled["pc2"] - y_value)
        nearest[int(np.argmin(filled_distance))] = 1
    filled["flagged"] = nearest
    if int(filled["flagged"].sum()) != len(saved):
        raise RuntimeError(f"{dataset} flagged rings did not match distinct filled points")
    return filled, max(errors)


def quartile_boxes(ax, groups: list[np.ndarray], labels: list[str], colors: list[str]) -> None:
    drawn = ax.boxplot(
        groups,
        tick_labels=labels,
        patch_artist=True,
        widths=0.55,
        showfliers=True,
        flierprops={"marker": "o", "markersize": 3.5, "markerfacecolor": "white", "markeredgecolor": "#222222", "markeredgewidth": 0.6},
    )
    for patch, color in zip(drawn["boxes"], colors, strict=True):
        patch.set_facecolor("white")
        patch.set_edgecolor(color)
        patch.set_linewidth(1.5)
    for element in ("whiskers", "caps"):
        for line in drawn[element]:
            line.set_color("#222222")
            line.set_linewidth(1.0)
    for line in drawn["medians"]:
        line.set_color("#000000")
        line.set_linewidth(1.3)


def figure_pca(points: dict[str, pd.DataFrame], summary: pd.DataFrame) -> None:
    figure, axes = plt.subplots(1, 3, figsize=(11.6, 4.15), dpi=120)
    figure.patch.set_facecolor("white")
    letters = ("A", "B", "C")
    for ax, dataset, letter in zip(axes, DATASETS, letters, strict=True):
        ax.set_facecolor("white")
        frame = points[dataset]
        row = summary.loc[summary["dataset"] == dataset].iloc[0]
        n = len(frame)
        size = 36 if n < 100 else 22 if n < 500 else 12
        alpha = 1.0 if n < 100 else 0.85 if n < 500 else 0.45
        for diagnosis, color in (("Control", CONTROL), ("AD", AD)):
            chosen = frame["diagnosis"] == diagnosis
            ax.scatter(
                frame.loc[chosen, "pc1"],
                frame.loc[chosen, "pc2"],
                c=color,
                s=size,
                alpha=alpha,
                linewidths=0,
                label=diagnosis,
                rasterized=False,
            )
        flagged = frame["flagged"] == 1
        if flagged.any():
            ax.scatter(
                frame.loc[flagged, "pc1"],
                frame.loc[flagged, "pc2"],
                facecolors="none",
                edgecolors="#000000",
                s=size + 26,
                linewidths=0.8,
                label="Flagged",
            )
        ax.set_aspect("equal", adjustable="datalim")
        title(ax, letter, f"{DISPLAY[dataset]} (n = {int(row['n_samples'])})")
        dress(
            ax,
            f"PC1 ({100 * row['variance_pc1']:.1f}% variance)",
            f"PC2 ({100 * row['variance_pc2']:.1f}% variance)",
        )
    handles, labels = axes[0].get_legend_handles_labels()
    figure.legend(handles, labels, frameon=False, ncol=3, loc="lower center", bbox_to_anchor=(0.5, -0.02), fontsize=9)
    figure.tight_layout(rect=(0, 0.04, 1, 1))
    save(figure, "figure1_sample_structure_pca")


def figure_expression(sample_mean: pd.DataFrame, zero_fraction: pd.DataFrame) -> None:
    figure, axes = plt.subplots(1, 2, figsize=(10.6, 4.35), dpi=120)
    figure.patch.set_facecolor("white")
    tick_labels = ["ROSMAP\nDLPFC", "GSE120584\nserum", "GSE215789\nplasma"]
    panels = (
        (axes[0], sample_mean, "A", "Sample mean", "Mean of min-max scaled values"),
        (axes[1], zero_fraction, "B", "Exact-zero proportion", "Proportion of exact zeros"),
    )
    for ax, frame, letter, heading, ylabel in panels:
        ax.set_facecolor("white")
        groups = [frame.loc[frame["group"] == name, "value"].to_numpy(dtype=float) for name in DATASETS]
        quartile_boxes(ax, groups, tick_labels, [DATASET_COLOR[name] for name in DATASETS])
        ax.set_title(heading, fontsize=11, fontweight="bold", pad=12)
        ax.text(-0.16, 1.04, letter, transform=ax.transAxes, fontsize=14, fontweight="bold", ha="left", va="bottom", clip_on=False)
        dress(ax, "Dataset", ylabel)
    figure.tight_layout(w_pad=2.4)
    save(figure, "figure2_scaled_matrix_summaries")


def figure_coverage(identity: pd.DataFrame, intersections: pd.DataFrame, prevalence: pd.DataFrame) -> None:
    figure = plt.figure(figsize=(12.8, 5.5), dpi=120)
    figure.patch.set_facecolor("white")
    outer = figure.add_gridspec(
        1, 3, width_ratios=[1.12, 1.55, 1.0], wspace=0.42, left=0.07, right=0.985, bottom=0.16, top=0.90
    )
    ax = figure.add_subplot(outer[0])
    upset = outer[1].subgridspec(2, 2, height_ratios=[2.5, 1.25], width_ratios=[1.15, 2.2], hspace=0.05, wspace=0.08)
    observation = figure.add_subplot(outer[2])
    ax.set_facecolor("white")
    classes = ["Resolved mature", "Ambiguous", "Unresolved", "Non-human"]
    columns = {
        "Resolved mature": "resolved_mature",
        "Ambiguous": "ambiguous",
        "Unresolved": "unresolved",
        "Non-human": "non_mirna_control",
    }
    y = np.arange(len(DATASETS))
    left = np.zeros(len(DATASETS))
    lookup = identity.set_index("dataset")
    totals = np.array([float(lookup.loc[dataset, "n_features"]) for dataset in DATASETS])
    for class_name in classes:
        widths = np.array([float(lookup.loc[dataset, columns[class_name]]) for dataset in DATASETS]) / totals
        ax.barh(y, widths, left=left, color=CLASS_COLOR[class_name], height=0.62, label=class_name)
        left += widths
    for ypos, total in enumerate(totals):
        ax.text(1.02, ypos, f"n = {int(total)}", va="center", fontsize=8)
    ax.set_yticks(y, [DISPLAY[name] for name in DATASETS])
    ax.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.set_xlim(0, 1.45)
    ax.invert_yaxis()
    class_handles, class_labels = ax.get_legend_handles_labels()
    title(ax, "A", "Feature-identifier class")
    dress(ax, "Fraction of retained features", "")

    labels = {
        "GSE120584": "Serum only",
        "GSE120584+GSE215789": "Serum and plasma",
        "ROSMAP+GSE120584+GSE215789": "All three",
        "ROSMAP+GSE120584": "ROSMAP and serum",
        "GSE215789": "Plasma only",
        "ROSMAP": "ROSMAP only",
        "ROSMAP+GSE215789": "ROSMAP and plasma",
    }
    exclusive = intersections[intersections["membership"].isin(labels)].copy()
    exclusive = exclusive.sort_values(["n_resolved_mature", "membership"], ascending=[False, True])
    dot_names = ["ROSMAP", "GSE120584", "GSE215789"]
    set_y = np.array([2, 1, 0])
    positions = np.arange(len(exclusive))
    bar_ax = figure.add_subplot(upset[0, 1])
    dot_ax = figure.add_subplot(upset[1, 1], sharex=bar_ax)
    size_ax = figure.add_subplot(upset[1, 0], sharey=dot_ax)
    figure.add_subplot(upset[0, 0]).axis("off")
    for panel in (bar_ax, dot_ax, size_ax):
        panel.set_facecolor("white")

    heights = exclusive["n_resolved_mature"].to_numpy(dtype=float)
    bar_ax.bar(positions, heights, color="#009E73", width=0.72)
    bar_ax.set_ylim(0, max(heights.max() * 1.2, 1))
    for xpos, height in zip(positions, heights):
        bar_ax.text(xpos, height + heights.max() * 0.03, str(int(height)), ha="center", va="bottom", fontsize=7)
    bar_ax.tick_params(axis="x", labelbottom=False, length=0)
    bar_ax.set_ylabel("Resolved miRNAs", fontsize=9, fontweight="bold")
    title(bar_ax, "B", "Resolved-name intersections")
    bar_ax.spines["top"].set_visible(False)
    bar_ax.spines["right"].set_visible(False)
    bar_ax.tick_params(labelsize=8)
    for label in bar_ax.get_yticklabels():
        label.set_fontweight("bold")

    for column, (_, row) in enumerate(exclusive.iterrows()):
        active = []
        for index, dataset in enumerate(DATASETS):
            present = int(row[dataset]) == 1
            if present:
                active.append(set_y[index])
            dot_ax.scatter(
                column,
                set_y[index],
                s=32,
                color="#009E73" if present else "white",
                edgecolor="#333333",
                linewidth=0.6,
                zorder=3,
            )
        if len(active) > 1:
            dot_ax.plot([column, column], [min(active), max(active)], color="#333333", linewidth=1.0, zorder=2)
    dot_ax.set_xlim(-0.65, len(exclusive) - 0.35)
    dot_ax.set_ylim(-0.55, 2.55)
    dot_ax.tick_params(labelleft=False, labelbottom=False, length=0)
    for spine in dot_ax.spines.values():
        spine.set_visible(False)

    set_sizes = [int(lookup.loc[name, "resolved_mature"]) for name in DATASETS]
    size_ax.barh(set_y, set_sizes, color="#009E73", height=0.52)
    size_ax.set_yticks(set_y, dot_names)
    size_ax.tick_params(axis="y", labelsize=7, pad=2, length=0)
    size_ax.tick_params(axis="x", labelsize=7)
    for label in size_ax.get_yticklabels() + size_ax.get_xticklabels():
        label.set_fontweight("bold")
    size_ax.set_xlabel("Set size", fontsize=8, fontweight="bold")
    size_ax.set_xlim(0, max(set_sizes) * 1.42)
    for y_pos, size in zip(set_y, set_sizes):
        size_ax.text(size + max(set_sizes) * 0.04, y_pos, str(size), va="center", ha="left", fontsize=7)
    size_ax.spines["top"].set_visible(False)
    size_ax.spines["right"].set_visible(False)

    ax = observation
    ax.set_facecolor("white")
    for diagnosis, color in (("Control", CONTROL), ("AD", AD)):
        values = np.sort(
            prevalence.loc[prevalence["diagnosis"] == diagnosis, "read_observation_prevalence"].to_numpy(dtype=float)
        )
        fraction = np.arange(1, values.size + 1) / values.size
        ax.step(values, fraction, where="post", color=color, linewidth=1.7, label=diagnosis)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.legend(frameon=False, fontsize=8, loc="lower right")
    title(ax, "C", "GSE215789 read observation")
    dress(ax, "Fraction of samples with count > 0", "Fraction of retained features")
    figure.legend(
        class_handles,
        class_labels,
        frameon=False,
        fontsize=7.5,
        ncol=2,
        loc="lower left",
        bbox_to_anchor=(0.05, 0.01),
    )
    save(figure, "figure3_mature_coverage_and_observation")


def figure_heterogeneity(scores: pd.DataFrame, summary: pd.DataFrame) -> None:
    figure, axes = plt.subplots(1, 2, figsize=(8.6, 4.3), dpi=120)
    figure.patch.set_facecolor("white")
    for ax, dataset, letter in zip(axes, ("ROSMAP", "GSE120584"), ("A", "B"), strict=True):
        ax.set_facecolor("white")
        subset = scores[scores["dataset"] == dataset]
        groups = []
        lines = []
        for diagnosis in ("Control", "AD"):
            values = subset.loc[subset["diagnosis"] == diagnosis, "heterogeneity"].to_numpy(dtype=float)
            groups.append(values)
            row = summary[(summary["dataset"] == dataset) & (summary["diagnosis"] == diagnosis)].iloc[0]
            lines.append(f"{diagnosis} n = {int(row['n'])}; median {row['median']:.3f} (IQR {row['q1']:.3f}–{row['q3']:.3f})")
        quartile_boxes(ax, groups, ["Control", "AD"], [CONTROL, AD])
        upper = float(np.nanmax(np.concatenate(groups))) * 1.55
        ax.set_ylim(0, upper)
        ax.text(0.03, 0.97, "\n".join(lines), transform=ax.transAxes, va="top", ha="left", fontsize=7.5)
        title(ax, letter, DISPLAY[dataset])
        dress(ax, "Diagnosis", "Profile dissimilarity\nmedian (1 − Spearman ρ)")
    figure.tight_layout()
    save(figure, "figure4_within_group_dissimilarity")


def write_captions(candidates: pd.DataFrame) -> None:
    candidate_bits = []
    for feature in ("hsa-miR-324-3p", "hsa-miR-574-3p", "hsa-miR-769-3p"):
        parts = []
        for diagnosis in ("AD", "Control"):
            row = candidates[(candidates["feature_id"] == feature) & (candidates["diagnosis"] == diagnosis)].iloc[0]
            parts.append(f"{diagnosis} {int(row['n_observed'])}/{int(row['n_samples'])}")
        candidate_bits.append(f"{feature}: " + ", ".join(parts))
    captions = {
        "figure1_sample_structure_pca": """Figure 1. Unsupervised structure of the three original ranking matrices.

The question is whether AD and Control samples occupy different regions of each dataset after a column-centered principal component analysis. Each panel is one dataset. The matrices were not combined. Diagnosis was not used to fit the components; it only colours the points after the scores were obtained.

A. ROSMAP dorsolateral prefrontal cortex, NanoString, 378 samples (209 AD, 169 Control) and 309 miRNAs. PC1 explains 14.5% of the variance and PC2 explains 7.6%.
B. GSE120584 serum, microarray, 1,309 samples (1,021 AD, 288 Control) and 2,521 miRNAs. PC1 explains 55.4% of the variance and PC2 explains 6.8%.
C. GSE215789 plasma, miRNA sequencing, 44 samples (23 AD, 21 Control) and 1,137 miRNAs. PC1 explains 42.1% of the variance and PC2 explains 7.7%.

Point coordinates were read from the geometry of the original PCA figures. Those SVG files do not store sample identifiers, so this figure is not a recovered sample-level score matrix. Axes use an equal data aspect, so one unit on PC1 has the same length as one unit on PC2 within a panel. Axis ranges are not shared across panels. Blue is Control and orange is AD, using the colours stored in the original figures. Open circles mark the points whose coordinates match the previously flagged PC1–PC2 positions. The original screen used a squared robust distance above the chi-square 0.999 quantile (threshold 13.816 on 2 degrees of freedom): 3 in ROSMAP, 8 in GSE120584, and 6 in GSE215789. No sample was removed. Separation in these plots is not evidence of differential expression or of classification performance. Component signs are arbitrary.
""",
        "figure2_scaled_matrix_summaries": """Figure 2. Sample-level summaries of the feature-wise min-max scaled ranking matrices.

The question is how the scaled values are distributed inside each original matrix. Each matrix was scaled separately to its own unit interval before ranking. The plotted numbers are not raw expression abundance, and they are not comparable across NanoString, microarray, and sequencing.

A. Mean of the min-max scaled feature values within each sample. ROSMAP n = 378, GSE120584 n = 1,309, GSE215789 n = 44.
B. Proportion of retained features equal to exact zero in each sample. Sample sizes match panel A. The matrix-wide zero proportions are 0.0020 in ROSMAP, 0.0007 in GSE120584, and 0.675 in GSE215789.

Boxes show the median and quartiles. Whiskers extend to 1.5 times the interquartile range, and points beyond the whiskers are plotted individually. Box edges use the dataset colours blue, orange, and green in the same order as the x-axis labels. No legend is drawn. No diagnosis split and no significance test are shown. None of the three matrices has missing values.
""",
        "figure3_mature_coverage_and_observation": """Figure 3. Mature-miRNA identifiers in the retained feature lists, and read observation in the plasma sequencing matrix.

The question is which original feature names correspond to one mature miRNA, which resolved names are shared across datasets, and how often the retained plasma features have a nonzero count. A name is resolved only when it is a unique product name in the project accession map. Arm labels were not added or removed. Identifiers that join more than one miRNA name were not split. The resolved counts depend on that map. They are not a census of every mature miRNA measured by the assay.

A. Class of every retained feature, drawn as a fraction of that dataset's feature list. The count at the right is the number of retained features. ROSMAP has 95 resolved, 8 ambiguous, 189 unresolved, and 17 non-human features (309 total). GSE120584 has 2,521 resolved features. GSE215789 has 1,017 resolved, 41 ambiguous, and 79 unresolved features (1,137 total). The plasma count matrix contains 2,362 resolved names before the original all-zero filter.
B. UpSet plot of resolved mature identities only. Ambiguous, unresolved, and non-human features were excluded from every intersection. Horizontal bars at the left are the resolved-set sizes: ROSMAP 95, GSE120584 2,521, and GSE215789 1,017. Vertical bars are exclusive intersection sizes, and a filled dot means that dataset belongs to the intersection. The exclusive sizes are serum only 1,485, serum and plasma 942, all three datasets 58, ROSMAP and serum 36, plasma only 17, ROSMAP only 1, and ROSMAP and plasma outside serum 0. Pairwise totals, counting the shared three-way names, are ROSMAP ∩ serum = 94 (Jaccard 0.037), ROSMAP ∩ plasma = 58 (Jaccard 0.055), and serum ∩ plasma = 1,000 (Jaccard 0.394). These counts describe the project's accession map, not the complete biological content of each assay. ROSMAP contributes only 95 resolved names because 189 of its 309 retained probe names have no accession in that map. The small overlap is therefore an identifier-resolution limit. It does not mean the unresolved miRNAs were absent from DLPFC. Arm labels were preserved, and the three candidates hsa-miR-324-3p, hsa-miR-574-3p, and hsa-miR-769-3p are retained in every dataset.
C. Empirical distribution of read-observation prevalence for the 1,137 retained GSE215789 features. A feature is observed in a sample when its original count is greater than 0. This is not an analytical detection call. Curves are AD (n = 23) and Control (n = 21). The median fraction of features observed in a sample is 0.296 in AD (IQR 0.231–0.351) and 0.343 in Control (IQR 0.218–0.441). Candidate counts are """
        + "; ".join(candidate_bits)
        + ". Assay-supported detection prevalence could not be assessed for the NanoString and microarray datasets because validated detection calls or corresponding background-control information were unavailable.\n",
        "figure4_within_group_dissimilarity": """Figure 4. Concordance of normalized expression profiles within diagnosis groups.

The question is how similar the pre-scaling expression profiles are among samples that share a diagnosis. For each sample, eligible features were ranked with average ranks for ties. Dissimilarity to another sample in the same group is 1 minus Spearman ρ, and the plotted score is the median of those dissimilarities. All retained features were eligible: 309 in ROSMAP and 2,521 in GSE120584. No significance test was applied to the dependent pairwise correlations.

A. ROSMAP normalized NanoString values. Control n = 169, median 0.069 (IQR 0.065–0.077). AD n = 209, median 0.071 (IQR 0.065–0.077).
B. GSE120584 normalized microarray values. Control n = 288, median 0.168 (IQR 0.152–0.212). AD n = 1,021, median 0.166 (IQR 0.148–0.205).

Boxes show the median and quartiles of those per-sample scores. Whiskers extend to 1.5 times the interquartile range, and points beyond the whiskers are plotted individually. Blue is Control and orange is AD. The y-axes are separate because each score describes concordance inside one normalized matrix. Within each dataset the AD and Control medians are nearly the same. That descriptive pattern is not evidence of disease-specific heterogeneity. GSE215789 is not shown because its pre-scaling matrix is a read-count matrix rather than documented normalized expression.
""",
    }
    for stem, text in captions.items():
        (FINAL / f"{stem}_caption.txt").write_text(text, encoding="utf-8")


def main() -> None:
    for folder in (FINAL, PREVIEW, PLOTS, TABLES, REPORTS):
        folder.mkdir(parents=True, exist_ok=True)
    pca_summary = pd.read_csv(EDA / "pca_summary.csv")
    dataset_summary = pd.read_csv(EDA / "dataset_summary.csv")
    outliers = pd.read_csv(EDA / "outlier_summary.csv")
    sample_mean = pd.read_csv(EDA / "skill_audits/sample_mean/plotting_data.csv")
    zero_fraction = pd.read_csv(EDA / "skill_audits/zero_fraction/plotting_data.csv")
    identity = pd.read_csv(BIO / "tables/mature_identity_summary.tsv", sep="\t")
    intersections = pd.read_csv(BIO / "tables/coverage_intersections.tsv", sep="\t")
    prevalence = pd.read_csv(BIO / "supporting_data/feature_observation_prevalence.tsv", sep="\t")
    candidates = pd.read_csv(BIO / "tables/candidate_read_observation.tsv", sep="\t")
    heterogeneity = pd.read_csv(BIO / "supporting_data/sample_profile_heterogeneity.tsv", sep="\t")
    heterogeneity_summary = pd.read_csv(BIO / "tables/profile_heterogeneity_group_summary.tsv", sep="\t")

    recovered = {}
    errors = {}
    provenance = []
    expected_variance = {"ROSMAP": (14.5, 7.6), "GSE120584": (55.4, 6.8), "GSE215789": (42.1, 7.7)}
    for dataset in DATASETS:
        points, error = mark_flagged(parse_pca_svg(SVG[dataset]), outliers, dataset)
        expected = int(pca_summary.loc[pca_summary["dataset"] == dataset, "n_samples"].iloc[0])
        if len(points) != expected:
            raise RuntimeError(f"{dataset} recovered {len(points)} points; the PCA table has {expected}")
        counts = points["diagnosis"].value_counts().to_dict()
        summary_row = dataset_summary.loc[dataset_summary["dataset"] == dataset].iloc[0]
        if counts.get("AD") != int(summary_row["n_ad"]) or counts.get("Control") != int(summary_row["n_control"]):
            raise RuntimeError(f"{dataset} diagnosis counts in the SVG do not match dataset_summary.csv")
        if error > 0.03:
            raise RuntimeError(f"{dataset} flagged-point recovery error is {error}")
        recovered[dataset] = points
        errors[dataset] = error
        exported = points.assign(dataset=dataset)
        exported.insert(0, "point_index", np.arange(len(exported)))
        exported.to_csv(PLOTS / f"figure1_{dataset.lower()}_pca_points.tsv", sep="\t", index=False)
        if "sample_id" in exported.columns:
            raise RuntimeError("Recovered PCA tables must not invent sample identifiers")
        variance_row = pca_summary.loc[pca_summary["dataset"] == dataset].iloc[0]
        shown = (round(100 * float(variance_row["variance_pc1"]), 1), round(100 * float(variance_row["variance_pc2"]), 1))
        if shown != expected_variance[dataset]:
            raise RuntimeError(f"{dataset} displayed variance {shown} does not match the original one-decimal report")
        residual = tick_residual(SVG[dataset])
        if residual > 1e-6:
            raise RuntimeError(f"{dataset} SVG tick transform is not linear")
        provenance.append(
            {
                "dataset": dataset,
                "source_svg": str(SVG[dataset]),
                "n_points": int(len(points)),
                "n_control": int((points["diagnosis"] == "Control").sum()),
                "n_ad": int((points["diagnosis"] == "AD").sum()),
                "n_flagged": int(points["flagged"].sum()),
                "max_flagged_coordinate_error": error,
                "max_tick_transform_residual": residual,
                "variance_pc1_percent_1dp": shown[0],
                "variance_pc2_percent_1dp": shown[1],
                "sample_ids_in_svg": False,
                "coordinate_table": str(PLOTS / f"figure1_{dataset.lower()}_pca_points.tsv"),
            }
        )

    for frame, expected in ((sample_mean, dataset_summary), (zero_fraction, dataset_summary)):
        for dataset in DATASETS:
            n_rows = int((frame["group"] == dataset).sum())
            n_samples = int(expected.loc[expected["dataset"] == dataset, "n_samples"].iloc[0])
            if n_rows != n_samples:
                raise RuntimeError(f"{dataset} summary rows {n_rows} do not match {n_samples} samples")
    sample_mean.to_csv(PLOTS / "figure2_sample_mean.csv", index=False)
    zero_fraction.to_csv(PLOTS / "figure2_zero_fraction.csv", index=False)

    identity_check = identity.copy()
    identity_check["sum_classes"] = (
        identity_check["resolved_mature"]
        + identity_check["ambiguous"]
        + identity_check["unresolved"]
        + identity_check["non_mirna_control"]
    )
    if not identity_check["sum_classes"].equals(identity_check["n_features"]):
        raise RuntimeError("Identifier classes do not add to the retained feature count")
    identity.to_csv(PLOTS / "figure3_identifier_classes.tsv", sep="\t", index=False)
    intersections.to_csv(PLOTS / "figure3_intersections.tsv", sep="\t", index=False)
    prevalence.to_csv(PLOTS / "figure3_read_observation_prevalence.tsv", sep="\t", index=False)

    for dataset in ("ROSMAP", "GSE120584"):
        for diagnosis in ("AD", "Control"):
            values = heterogeneity.loc[
                (heterogeneity["dataset"] == dataset) & (heterogeneity["diagnosis"] == diagnosis),
                "heterogeneity",
            ].to_numpy(dtype=float)
            saved = heterogeneity_summary[
                (heterogeneity_summary["dataset"] == dataset) & (heterogeneity_summary["diagnosis"] == diagnosis)
            ].iloc[0]
            if len(values) != int(saved["n"]):
                raise RuntimeError(f"{dataset} {diagnosis} heterogeneity count changed")
            if abs(float(np.quantile(values, 0.5, method="linear")) - float(saved["median"])) > 1e-9:
                raise RuntimeError(f"{dataset} {diagnosis} median does not match the saved summary")
    heterogeneity.to_csv(PLOTS / "figure4_sample_heterogeneity.tsv", sep="\t", index=False)

    dataset_summary.to_csv(TABLES / "sample_composition.tsv", sep="\t", index=False)
    pca_summary.to_csv(TABLES / "pca_variance.tsv", sep="\t", index=False)
    shutil.copyfile(BIO / "tables/biological_eda_summary.tsv", TABLES / "biological_eda_summary.tsv")
    shutil.copyfile(BIO / "tables/metadata_group_summary.tsv", TABLES / "metadata_group_summary.tsv")
    shutil.copyfile(BIO / "tables/batch_diagnosis_counts.tsv", TABLES / "batch_diagnosis_counts.tsv")

    figure_pca(recovered, pca_summary)
    figure_expression(sample_mean, zero_fraction)
    figure_coverage(identity, intersections, prevalence)
    figure_heterogeneity(heterogeneity, heterogeneity_summary)
    write_captions(candidates)

    provenance_record = {
        "method": "Pixel positions of filled markers and open rings were read from the original SVG. Axis ticks supplied a linear map from pixels to PC1 and PC2. PCA was not refit.",
        "sample_identity": "The SVG geometry does not contain sample identifiers. Coordinate tables therefore have a point index, diagnosis colour, and flagged status, but no sample ID.",
        "flagged_status": "An open ring was accepted when its recovered coordinates matched one row of outlier_summary.csv. The nearest filled marker was then marked flagged. This is a geometric match, not an identifier stored in the SVG.",
        "datasets": provenance,
    }
    (REPORTS / "pca_coordinate_provenance.json").write_text(json.dumps(provenance_record, indent=2) + "\n", encoding="utf-8")
    audit = {
        "pca_source": "Coordinates recovered from the original SVG files. PCA was not refit.",
        "pca_provenance": str(REPORTS / "pca_coordinate_provenance.json"),
        "flagged_point_max_absolute_error": errors,
        "skill_renderers_used": [],
        "skill_renderer_not_used": {
            "pca_scatter": "That renderer refits PCA from a long expression table.",
            "grouped_boxes": "The quartile rule is followed, but the renderer cannot place these panels in the composite figures.",
            "grouped_bars": "It always runs a Welch test. Class counts are a census and are not plotted.",
        },
        "colours": {"Control": CONTROL, "AD": AD, "datasets": DATASET_COLOR, "identifier_classes": CLASS_COLOR},
        "exports": ["png-300dpi", "svg", "pdf"],
        "pdf_fonttype": 42,
        "omitted_panels": [
            "NanoString detection prevalence",
            "Microarray detection prevalence",
            "PCA-metadata associations",
            "GSE215789 within-group heterogeneity",
            "Class-count bars",
            "Assay-plate bars",
        ],
    }
    (FINAL / "figure_audit.json").write_text(json.dumps(audit, indent=2) + "\n", encoding="utf-8")
    (REPORTS / "figure_validation.json").write_text(json.dumps(audit, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"flagged_point_max_absolute_error": errors}, indent=2))


if __name__ == "__main__":
    main()
