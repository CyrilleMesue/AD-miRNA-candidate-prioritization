#!/usr/bin/env python
"""Compose the four validated EDA figures into two five-panel figures.

Reads the saved plotting tables only. Does not refit PCA, rescale matrices,
impute values, or run statistical tests.

python results/eda_consolidated/composite_figures/scripts/compose_eda_figures.py
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
matplotlib.rcParams.update(
    {
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "svg.fonttype": "none",
        "font.family": "DejaVu Sans",
        "font.size": 8,
        "axes.labelsize": 8,
        "xtick.labelsize": 7.5,
        "ytick.labelsize": 7.5,
        "legend.fontsize": 7.5,
        "axes.linewidth": 0.8,
        "xtick.major.width": 0.6,
        "ytick.major.width": 0.6,
        "xtick.major.size": 3,
        "ytick.major.size": 3,
    }
)

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator, ScalarFormatter
from PIL import Image

ROOT = Path(__file__).resolve().parents[4]
EDA = ROOT / "results/eda"
BIO = ROOT / "results/biological_eda"
CONSOLIDATED = ROOT / "results/eda_consolidated"
PLOTS = CONSOLIDATED / "plotting_data"
OUT = CONSOLIDATED / "composite_figures"
PREVIEW = OUT / "previews"
REPORTS = OUT / "reports"

CONTROL = "#a3a39f"
AD = "#e8762b"
DATASET_COLOR = {
    "ROSMAP": "#8E5BA8",
    "GSE120584": "#C2873A",
    "GSE215789": "#B5473F",
}
CLASS_COLOR = {
    "Resolved mature": "#009E73",
    "Ambiguous": "#CC79A7",
    "Unresolved": "#56B4E9",
    "Non-human": "#666666",
}
DATASETS = ("ROSMAP", "GSE120584", "GSE215789")
WIDTH_IN = 180 / 25.4
EXCLUSIVE_ORDER = (
    "GSE120584",
    "GSE120584+GSE215789",
    "ROSMAP+GSE120584+GSE215789",
    "ROSMAP+GSE120584",
    "GSE215789",
    "ROSMAP",
    "ROSMAP+GSE215789",
)


def dress(ax, xlabel: str, ylabel: str) -> None:
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.tick_params(labelsize=7.5, width=0.6, length=3)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(False)
    ax.set_facecolor("white")


def panel_title(ax, letter: str, text: str) -> None:
    ax.set_title(f"{letter}  {text}", loc="left", fontsize=9, fontweight="bold", pad=4)


def quartile_boxes(ax, groups: list[np.ndarray], labels: list[str], colors: list[str]) -> None:
    drawn = ax.boxplot(
        groups,
        tick_labels=labels,
        patch_artist=True,
        widths=0.58,
        showfliers=True,
        flierprops={
            "marker": "o",
            "markersize": 3.2,
            "markerfacecolor": "white",
            "markeredgecolor": "#222222",
            "markeredgewidth": 0.5,
        },
    )
    for patch, color in zip(drawn["boxes"], colors, strict=True):
        patch.set_facecolor("white")
        patch.set_edgecolor(color)
        patch.set_linewidth(1.4)
    for element in ("whiskers", "caps"):
        for line in drawn[element]:
            line.set_color("#222222")
            line.set_linewidth(0.9)
    for line in drawn["medians"]:
        line.set_color("#000000")
        line.set_linewidth(1.2)


def diagnosis_handles(include_flagged: bool) -> list[Line2D]:
    handles = [
        Line2D([0], [0], marker="o", linestyle="none", markerfacecolor=CONTROL, markeredgecolor="none", markersize=6, label="Control"),
        Line2D([0], [0], marker="o", linestyle="none", markerfacecolor=AD, markeredgecolor="none", markersize=6, label="AD"),
    ]
    if include_flagged:
        handles.append(
            Line2D(
                [0],
                [0],
                marker="o",
                linestyle="none",
                markerfacecolor="none",
                markeredgecolor="#000000",
                markeredgewidth=0.9,
                markersize=6,
                label="Flagged",
            )
        )
    return handles


def draw_pca(ax, frame: pd.DataFrame, pc1: float, pc2: float, letter: str, heading: str) -> None:
    n = len(frame)
    size = 30 if n < 100 else 14 if n < 500 else 8
    alpha = 1.0 if n < 100 else 0.85 if n < 500 else 0.55
    for diagnosis, color in (("Control", CONTROL), ("AD", AD)):
        chosen = frame["diagnosis"] == diagnosis
        ax.scatter(
            frame.loc[chosen, "pc1"],
            frame.loc[chosen, "pc2"],
            c=color,
            s=size,
            alpha=alpha,
            linewidths=0,
            zorder=2,
        )
    flagged = frame["flagged"].astype(int) == 1
    if flagged.any():
        ax.scatter(
            frame.loc[flagged, "pc1"],
            frame.loc[flagged, "pc2"],
            facecolors="none",
            edgecolors="#000000",
            s=size + 22,
            linewidths=0.7,
            zorder=3,
        )
    ax.margins(0.06)
    ax.set_aspect("equal", adjustable="datalim")
    formatter = ScalarFormatter()
    formatter.set_scientific(False)
    formatter.set_useOffset(False)
    for axis in (ax.xaxis, ax.yaxis):
        axis.set_major_locator(MaxNLocator(nbins=4, min_n_ticks=2, prune="both"))
        axis.set_major_formatter(formatter)
    panel_title(ax, letter, heading)
    dress(ax, f"PC1 ({pc1:.1f}%)", f"PC2 ({pc2:.1f}%)")


def draw_identifier(ax, identity: pd.DataFrame) -> tuple[list, list]:
    classes = ["Resolved mature", "Ambiguous", "Unresolved", "Non-human"]
    columns = {
        "Resolved mature": "resolved_mature",
        "Ambiguous": "ambiguous",
        "Unresolved": "unresolved",
        "Non-human": "non_mirna_control",
    }
    lookup = identity.set_index("dataset")
    y = np.arange(len(DATASETS))
    left = np.zeros(len(DATASETS))
    totals = np.array([float(lookup.loc[name, "n_features"]) for name in DATASETS])
    for class_name in classes:
        widths = np.array([float(lookup.loc[name, columns[class_name]]) for name in DATASETS]) / totals
        ax.barh(y, widths, left=left, color=CLASS_COLOR[class_name], height=0.62, label=class_name)
        left += widths
    for ypos, total in enumerate(totals):
        ax.text(1.03, ypos, f"n={int(total)}", va="center", ha="left", fontsize=7)
    ax.set_yticks(y, list(DATASETS))
    ax.set_xticks([0, 0.5, 1.0])
    ax.set_xlim(0, 1.48)
    ax.invert_yaxis()
    panel_title(ax, "A", "Identifier class")
    dress(ax, "Fraction of retained features", "")
    ax.tick_params(axis="y", length=0, pad=2)
    return ax.get_legend_handles_labels()


def draw_upset(fig, cell, identity: pd.DataFrame, intersections: pd.DataFrame) -> None:
    upset = cell.subgridspec(2, 2, height_ratios=[2.35, 1.15], width_ratios=[1.35, 1.7], hspace=0.06, wspace=0.05)
    bar_ax = fig.add_subplot(upset[0, 1])
    dot_ax = fig.add_subplot(upset[1, 1], sharex=bar_ax)
    size_ax = fig.add_subplot(upset[1, 0], sharey=dot_ax)
    fig.add_subplot(upset[0, 0]).axis("off")
    exclusive = intersections[intersections["membership"].isin(EXCLUSIVE_ORDER)].copy()
    exclusive["membership"] = pd.Categorical(exclusive["membership"], EXCLUSIVE_ORDER, ordered=True)
    exclusive = exclusive.sort_values("membership")
    lookup = identity.set_index("dataset")
    heights = exclusive["n_resolved_mature"].to_numpy(dtype=float)
    positions = np.arange(len(exclusive))
    bar_ax.bar(positions, heights, color="#009E73", width=0.68)
    ceiling = max(float(heights.max()) * 1.42, 1)
    bar_ax.set_ylim(0, ceiling)
    bar_ax.set_yticks([0, 500, 1000, 1500, 2000])
    for xpos, height in zip(positions, heights):
        bar_ax.text(
            xpos,
            height + heights.max() * 0.025,
            str(int(height)),
            ha="center",
            va="bottom",
            rotation=90,
            fontsize=6.5,
            clip_on=False,
        )
    bar_ax.tick_params(axis="x", labelbottom=False, length=0)
    bar_ax.set_ylabel("Resolved miRNAs")
    panel_title(bar_ax, "B", "Resolved overlap")
    bar_ax.spines["top"].set_visible(False)
    bar_ax.spines["right"].set_visible(False)
    bar_ax.tick_params(axis="y", labelsize=7)
    bar_ax.set_facecolor("white")
    bar_ax.grid(False)

    set_y = np.array([2, 1, 0])
    for column, (_, row) in enumerate(exclusive.iterrows()):
        active = []
        for index, dataset in enumerate(DATASETS):
            present = int(row[dataset]) == 1
            if present:
                active.append(set_y[index])
            dot_ax.scatter(
                column,
                set_y[index],
                s=26,
                color="#009E73" if present else "white",
                edgecolor="#333333",
                linewidth=0.55,
                zorder=3,
            )
        if len(active) > 1:
            dot_ax.plot([column, column], [min(active), max(active)], color="#333333", linewidth=0.9, zorder=2)
    dot_ax.set_xlim(-0.6, len(exclusive) - 0.4)
    dot_ax.set_ylim(-0.55, 2.55)
    dot_ax.tick_params(labelleft=False, labelbottom=False, length=0)
    for spine in dot_ax.spines.values():
        spine.set_visible(False)
    dot_ax.set_facecolor("white")
    dot_ax.grid(False)

    set_sizes = [int(lookup.loc[name, "resolved_mature"]) for name in DATASETS]
    size_ax.barh(set_y, set_sizes, color="#009E73", height=0.48)
    size_ax.set_yticks(set_y, ["ROSMAP", "GSE120584", "GSE215789"])
    size_ax.tick_params(axis="y", labelsize=6.5, pad=1, length=0)
    size_ax.tick_params(axis="x", labelsize=6.5)
    size_ax.set_xlabel("Set size", fontsize=7.5)
    size_ax.set_xlim(0, max(set_sizes) * 1.28)
    size_ax.set_xticks([0])
    for y_pos, size in zip(set_y, set_sizes):
        size_ax.text(size + max(set_sizes) * 0.03, y_pos, str(size), va="center", ha="left", color="#222222", fontsize=6.5)
    size_ax.spines["top"].set_visible(False)
    size_ax.spines["right"].set_visible(False)
    size_ax.set_facecolor("white")
    size_ax.grid(False)


def draw_ecdf(ax, prevalence: pd.DataFrame) -> None:
    for diagnosis, color in (("Control", CONTROL), ("AD", AD)):
        values = np.sort(prevalence.loc[prevalence["diagnosis"] == diagnosis, "read_observation_prevalence"].to_numpy(dtype=float))
        fraction = np.arange(1, values.size + 1) / values.size
        ax.step(values, fraction, where="post", color=color, linewidth=1.5)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.02)
    panel_title(ax, "C", "Read observation")
    dress(ax, "Fraction of samples\nwith count > 0", "Fraction of features")
    ax.set_xticks([0, 0.5, 1])
    ax.set_yticks([0, 0.5, 1])


def draw_dissimilarity(ax, scores: pd.DataFrame, summary: pd.DataFrame, dataset: str, letter: str) -> None:
    subset = scores[scores["dataset"] == dataset]
    groups = [subset.loc[subset["diagnosis"] == diagnosis, "heterogeneity"].to_numpy(dtype=float) for diagnosis in ("Control", "AD")]
    quartile_boxes(ax, groups, ["Control", "AD"], [CONTROL, AD])
    upper = float(np.nanmax(np.concatenate(groups)))
    ax.set_ylim(0, upper * 1.48)
    lines = []
    for diagnosis in ("Control", "AD"):
        row = summary[(summary["dataset"] == dataset) & (summary["diagnosis"] == diagnosis)].iloc[0]
        lines.append(f"{diagnosis} n={int(row['n'])}, median {row['median']:.3f}")
    ax.text(0.03, 0.98, "\n".join(lines), transform=ax.transAxes, va="top", ha="left", fontsize=6.5, linespacing=1.25)
    panel_title(ax, letter, dataset)
    dress(ax, "Diagnosis", "Dissimilarity\nmedian (1 − Spearman ρ)")


def figure_structure(points: dict[str, pd.DataFrame], summary: pd.DataFrame, sample_mean: pd.DataFrame, zero_fraction: pd.DataFrame) -> plt.Figure:
    fig = plt.figure(figsize=(WIDTH_IN, 6.85 * 0.65))
    fig.patch.set_facecolor("white")
    outer = fig.add_gridspec(2, 1, height_ratios=[0.92, 1.0], left=0.10, right=0.985, top=0.94, bottom=0.16, hspace=0.42)
    top = outer[0].subgridspec(1, 3, wspace=0.55)
    bottom = outer[1].subgridspec(1, 2, wspace=0.32)
    variance = summary.set_index("dataset")
    headings = {"ROSMAP": "ROSMAP", "GSE120584": "GSE120584", "GSE215789": "GSE215789"}
    for index, dataset in enumerate(DATASETS):
        ax = fig.add_subplot(top[index])
        row = variance.loc[dataset]
        draw_pca(
            ax,
            points[dataset],
            100 * float(row["variance_pc1"]),
            100 * float(row["variance_pc2"]),
            "ABC"[index],
            headings[dataset],
        )
    tick_labels = ["ROSMAP", "GSE120584", "GSE215789"]
    panels = (
        (fig.add_subplot(bottom[0]), sample_mean, "D", "Mean scaled expression", "Mean of feature-wise\nmin-max values"),
        (fig.add_subplot(bottom[1]), zero_fraction, "E", "Exact-zero fraction", "Exact-zero fraction"),
    )
    for ax, frame, letter, heading, ylabel in panels:
        groups = [frame.loc[frame["group"] == name, "value"].to_numpy(dtype=float) for name in DATASETS]
        quartile_boxes(ax, groups, tick_labels, [DATASET_COLOR[name] for name in DATASETS])
        panel_title(ax, letter, heading)
        dress(ax, "Dataset", ylabel)
        ax.tick_params(axis="x", labelsize=7)
        low, high = ax.get_ylim()
        ax.set_ylim(low, high + (high - low) * 0.08)
    fig.legend(
        handles=diagnosis_handles(True),
        frameon=False,
        ncol=3,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.005),
        handletextpad=0.35,
        columnspacing=1.0,
    )
    return fig


def figure_coverage(identity: pd.DataFrame, intersections: pd.DataFrame, prevalence: pd.DataFrame, scores: pd.DataFrame, summary: pd.DataFrame) -> plt.Figure:
    fig = plt.figure(figsize=(WIDTH_IN, 8.05 * 0.65))
    fig.patch.set_facecolor("white")
    outer = fig.add_gridspec(2, 1, height_ratios=[1.38, 1.0], left=0.13, right=0.99, top=0.955, bottom=0.145, hspace=0.36)
    top = outer[0].subgridspec(1, 3, width_ratios=[0.98, 1.72, 1.18], wspace=0.40)
    bottom = outer[1].subgridspec(1, 2, wspace=0.36)
    class_ax = fig.add_subplot(top[0])
    handles, labels = draw_identifier(class_ax, identity)
    draw_upset(fig, top[1], identity, intersections)
    draw_ecdf(fig.add_subplot(top[2]), prevalence)
    draw_dissimilarity(fig.add_subplot(bottom[0]), scores, summary, "ROSMAP", "D")
    draw_dissimilarity(fig.add_subplot(bottom[1]), scores, summary, "GSE120584", "E")
    fig.legend(
        handles,
        labels,
        frameon=False,
        ncol=2,
        loc="lower left",
        bbox_to_anchor=(0.02, 0.004),
        handlelength=1.1,
        handletextpad=0.35,
        columnspacing=0.8,
        borderaxespad=0,
    )
    fig.legend(
        handles=diagnosis_handles(False),
        frameon=False,
        ncol=2,
        loc="lower right",
        bbox_to_anchor=(0.98, 0.004),
        handletextpad=0.35,
        columnspacing=0.8,
        borderaxespad=0,
    )
    return fig


def export(fig, stem: str) -> Path:
    OUT.mkdir(parents=True, exist_ok=True)
    png = OUT / f"{stem}.png"
    fig.savefig(png, dpi=300, bbox_inches="tight", pad_inches=0.03, facecolor="white")
    fig.savefig(OUT / f"{stem}.svg", bbox_inches="tight", pad_inches=0.03, facecolor="white")
    fig.savefig(OUT / f"{stem}.pdf", bbox_inches="tight", pad_inches=0.03, facecolor="white")
    plt.close(fig)
    return png


def write_previews(png: Path, stem: str) -> dict[str, str]:
    PREVIEW.mkdir(parents=True, exist_ok=True)
    image = Image.open(png)
    sizes = {}
    for mm in (160, 180):
        width = round(mm / 25.4 * 300)
        height = round(image.height * width / image.width)
        dest = PREVIEW / f"{stem}_{mm}mm.png"
        image.resize((width, height), Image.Resampling.LANCZOS).save(dest)
        sizes[str(mm)] = f"{width} x {height} px"
    sizes["source_px"] = f"{image.width} x {image.height}"
    sizes["source_mm_at_300dpi"] = f"{image.width / 300 * 25.4:.1f} x {image.height / 300 * 25.4:.1f}"
    return sizes


def layout_problems(fig) -> list[str]:
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    texts = []
    for artist in fig.findobj(plt.Text):
        content = artist.get_text().strip()
        if not content or not artist.get_visible():
            continue
        box = artist.get_window_extent(renderer).expanded(1.0, 1.0)
        if box.width <= 0 or box.height <= 0:
            continue
        texts.append((content.replace("\n", " "), box))
    problems = []
    width, height = fig.canvas.get_width_height()
    for content, box in texts:
        if box.x0 < -2 or box.y0 < -2 or box.x1 > width + 2 or box.y1 > height + 2:
            problems.append(f"outside figure: {content}")
    for index, (left_text, left_box) in enumerate(texts):
        for right_text, right_box in texts[index + 1 :]:
            x0 = max(left_box.x0, right_box.x0)
            y0 = max(left_box.y0, right_box.y0)
            x1 = min(left_box.x1, right_box.x1)
            y1 = min(left_box.y1, right_box.y1)
            if x1 - x0 < 3 or y1 - y0 < 3:
                continue
            area = (x1 - x0) * (y1 - y0)
            smaller = min(left_box.width * left_box.height, right_box.width * right_box.height)
            if area / smaller > 0.25:
                problems.append(f"overlap ({area:.0f}px): '{left_text}' | '{right_text}'")
    return problems


def fmt_iqr(row) -> str:
    return f"{row['median']:.3f} (IQR {row['q1']:.3f}–{row['q3']:.3f})"


def nfmt(value) -> str:
    return f"{int(value):,}"


def write_captions(bundle: dict) -> None:
    variance = bundle["pca_summary"].set_index("dataset")
    dataset = bundle["dataset_summary"].set_index("dataset")
    identity = bundle["identity"].set_index("dataset")
    hetero = bundle["heterogeneity_summary"]
    breadth = bundle["breadth"].set_index("diagnosis")
    candidates = []
    for feature in ("hsa-miR-324-3p", "hsa-miR-574-3p", "hsa-miR-769-3p"):
        parts = []
        for diagnosis in ("AD", "Control"):
            row = bundle["candidates"][(bundle["candidates"]["feature_id"] == feature) & (bundle["candidates"]["diagnosis"] == diagnosis)].iloc[0]
            parts.append(f"{diagnosis} {nfmt(row['n_observed'])}/{nfmt(row['n_samples'])}")
        candidates.append(f"{feature}: " + ", ".join(parts))

    def pc(name: str) -> str:
        row = variance.loc[name]
        return f"PC1 explains {100 * row['variance_pc1']:.1f}% of the variance and PC2 explains {100 * row['variance_pc2']:.1f}%"

    def group(name: str, diagnosis: str) -> str:
        row = hetero[(hetero["dataset"] == name) & (hetero["diagnosis"] == diagnosis)].iloc[0]
        return f"{diagnosis} n = {nfmt(row['n'])}, median {fmt_iqr(row)}"

    zeros = dataset["zero_proportion"]
    caption_1 = f"""Dataset characteristics and sample-level expression structure.

The question is what the expression characteristics and the major sample-level variation patterns are in the three Alzheimer's disease miRNA datasets. The datasets were measured with different technologies and were analysed separately: ROSMAP dorsolateral prefrontal cortex (NanoString nCounter), GSE120584 serum (miRNA microarray), and GSE215789 plasma (miRNA sequencing).

A. ROSMAP principal component plot. All {nfmt(variance.loc['ROSMAP', 'n_samples'])} observations are shown ({nfmt(dataset.loc['ROSMAP', 'n_ad'])} AD, {nfmt(dataset.loc['ROSMAP', 'n_control'])} Control; {nfmt(variance.loc['ROSMAP', 'n_mirnas'])} features). {pc('ROSMAP')}.
B. GSE120584 principal component plot. All {nfmt(variance.loc['GSE120584', 'n_samples'])} observations are shown ({nfmt(dataset.loc['GSE120584', 'n_ad'])} AD, {nfmt(dataset.loc['GSE120584', 'n_control'])} Control; {nfmt(variance.loc['GSE120584', 'n_mirnas'])} features). {pc('GSE120584')}.
C. GSE215789 principal component plot. All {nfmt(variance.loc['GSE215789', 'n_samples'])} observations are shown ({nfmt(dataset.loc['GSE215789', 'n_ad'])} AD, {nfmt(dataset.loc['GSE215789', 'n_control'])} Control; {nfmt(variance.loc['GSE215789', 'n_mirnas'])} features). {pc('GSE215789')}.

Coordinates in A–C are the reproduced column-centered SVD scores of the original ranking matrices. Each point keeps the sample identifier and diagnosis from that matrix. The scores match the original explained-variance ratios, the original outlier sample identifiers, and the SVG-derived point cloud after component signs are aligned to the original outlier coordinates. Within each panel the axes have an equal data aspect, so one unit of PC1 has the same length as one unit of PC2. Axis ranges and orientations are those of the source figures and are not shared across datasets. Grey is Control and orange is AD. Open circles mark coordinates that match the previously flagged PC1–PC2 positions: 3 in ROSMAP, 8 in GSE120584, and 6 in GSE215789. The original screen was a squared robust distance on unsupervised PC1 and PC2 above the chi-square 0.999 quantile (threshold 13.816 on 2 degrees of freedom). Flagged observations were not removed and were not independently confirmed as technical failures. Component signs are arbitrary. Separation in these panels is not evidence of differential expression or of classification performance.

D. Sample-level mean of the feature-wise min-max-scaled values. Every sample in the three ranking matrices is shown (n = {nfmt(dataset.loc['ROSMAP', 'n_samples'])}, {nfmt(dataset.loc['GSE120584', 'n_samples'])}, and {nfmt(dataset.loc['GSE215789', 'n_samples'])}).
E. Sample-level fraction of retained features equal to exact zero, for the same samples. The matrix-wide exact-zero proportions are {zeros['ROSMAP']:.4f} in ROSMAP, {zeros['GSE120584']:.4f} in GSE120584, and {zeros['GSE215789']:.3f} in GSE215789. An exact zero in these scaled matrices is not an analytical non-detection call.

Each matrix was scaled separately onto its own unit interval before ranking. The values in D and E are therefore not raw expression abundances, and they are not directly comparable across NanoString, microarray, and sequencing. Boxes show the median and quartiles. Whiskers extend to 1.5 times the interquartile range, and points beyond the whiskers are individual samples. Box edges identify the dataset (ROSMAP blue, GSE120584 orange, GSE215789 green) and do not encode diagnosis. The diagnosis legend applies to A–C only. No diagnosis split and no significance test are shown in D or E. None of the three matrices has missing values.
"""
    caption_2 = f"""Mature-miRNA coverage, read observation, and within-group profile variation.

The question is how the datasets differ in resolved mature-miRNA representation and sequencing read observation, and what within-group expression-profile variation is present in the datasets for which a normalised pre-scaling matrix is available.

A name is resolved only when it is a unique product name in the project accession map. Arm labels were not added or removed, and identifiers that join more than one miRNA name were not split. Resolved counts depend on that map. They are not a census of every mature miRNA measured by the assay.

A. Identifier class as a fraction of each retained feature list. ROSMAP has {nfmt(identity.loc['ROSMAP', 'resolved_mature'])} resolved, {nfmt(identity.loc['ROSMAP', 'ambiguous'])} ambiguous, {nfmt(identity.loc['ROSMAP', 'unresolved'])} unresolved, and {nfmt(identity.loc['ROSMAP', 'non_mirna_control'])} non-human features ({nfmt(identity.loc['ROSMAP', 'n_features'])} total). GSE120584 has {nfmt(identity.loc['GSE120584', 'resolved_mature'])} resolved features of {nfmt(identity.loc['GSE120584', 'n_features'])}. GSE215789 has {nfmt(identity.loc['GSE215789', 'resolved_mature'])} resolved, {nfmt(identity.loc['GSE215789', 'ambiguous'])} ambiguous, and {nfmt(identity.loc['GSE215789', 'unresolved'])} unresolved features ({nfmt(identity.loc['GSE215789', 'n_features'])} total). The plasma count matrix contains {nfmt(identity.loc['GSE215789', 'resolved_in_original_assay_matrix'])} resolved names before the original all-zero filter.
B. UpSet display of resolved mature identifiers only. Ambiguous, unresolved, and non-human names are excluded. Row labels are ROSMAP, GSE120584 (serum), and GSE215789 (plasma). Horizontal bars are the resolved-set sizes ({nfmt(identity.loc['ROSMAP', 'resolved_mature'])}, {nfmt(identity.loc['GSE120584', 'resolved_mature'])}, and {nfmt(identity.loc['GSE215789', 'resolved_mature'])}). Vertical bars are exclusive intersection sizes, and a filled dot means that dataset belongs to the intersection. From largest to smallest the exclusive sizes are serum only 1,485, serum and plasma 942, all three datasets 58, ROSMAP and serum 36, plasma only 17, ROSMAP only 1, and ROSMAP and plasma outside serum 0. Pairwise totals, which include the three-way names, are ROSMAP ∩ serum = 94 (Jaccard 0.037), ROSMAP ∩ plasma = 58 (Jaccard 0.055), and serum ∩ plasma = 1,000 (Jaccard 0.394). ROSMAP contributes only 95 resolved names because 189 of its 309 retained probe names have no accession in the map. The small overlap is an identifier-resolution limit. It does not mean that unresolved ROSMAP features were biologically absent, and it is not a complete biological overlap across measurement technologies. The candidates hsa-miR-324-3p, hsa-miR-574-3p, and hsa-miR-769-3p are retained in every dataset.
C. Empirical distribution of read-observation prevalence for the {nfmt(identity.loc['GSE215789', 'n_features'])} retained GSE215789 features. A feature is observed in a sample when its original count is greater than 0. This is not an analytical detection call. Curves are AD (n = 23) and Control (n = 21). The median fraction of retained features observed in a sample is {breadth.loc['AD', 'median']:.3f} in AD (IQR {breadth.loc['AD', 'q1']:.3f}–{breadth.loc['AD', 'q3']:.3f}) and {breadth.loc['Control', 'median']:.3f} in Control (IQR {breadth.loc['Control', 'q1']:.3f}–{breadth.loc['Control', 'q3']:.3f}). Candidate observation counts are {'; '.join(candidates)}. Assay-supported detection calls were not available for the NanoString or microarray datasets, so no detection panel is shown for those technologies.

D. Within-diagnosis expression-profile dissimilarity in ROSMAP. For each sample the score is the median of 1 minus Spearman ρ against the other samples with the same diagnosis, using average ranks for ties. All {nfmt(hetero.loc[hetero['dataset'].eq('ROSMAP'), 'n_eligible_features'].iloc[0])} retained features were eligible. {group('ROSMAP', 'Control')}. {group('ROSMAP', 'AD')}.
E. The same score in GSE120584. All {nfmt(hetero.loc[hetero['dataset'].eq('GSE120584'), 'n_eligible_features'].iloc[0])} retained features were eligible. {group('GSE120584', 'Control')}. {group('GSE120584', 'AD')}.

Boxes in D and E show the median and quartiles of the per-sample scores. Whiskers extend to 1.5 times the interquartile range, and points beyond the whiskers are individual samples. Grey is Control and orange is AD, as in panel C. The y-axes are separate because each score describes concordance inside one normalised matrix. No significance markers are shown. Within each dataset the AD and Control medians are nearly the same. That descriptive pattern does not establish disease-specific biological heterogeneity. GSE215789 is omitted because its pre-scaling matrix stores read counts rather than documented normalised expression.
"""
    (OUT / "figure_eda_01_caption.txt").write_text(caption_1, encoding="utf-8")
    (OUT / "figure_eda_02_caption.txt").write_text(caption_2, encoding="utf-8")


def load() -> dict:
    points = {}
    for dataset in DATASETS:
        frame = pd.read_csv(PLOTS / f"figure1_{dataset.lower()}_pca_points.tsv", sep="\t")
        if "sample_id" in frame.columns and (frame["sample_id"].isna().any() or frame["sample_id"].duplicated().any()):
            raise RuntimeError(f"{dataset} PCA table has a missing or duplicated sample_id")
        points[dataset] = frame
    sample_mean = pd.read_csv(PLOTS / "figure2_sample_mean.csv")
    zero_fraction = pd.read_csv(PLOTS / "figure2_zero_fraction.csv")
    identity = pd.read_csv(PLOTS / "figure3_identifier_classes.tsv", sep="\t")
    intersections = pd.read_csv(PLOTS / "figure3_intersections.tsv", sep="\t")
    prevalence = pd.read_csv(PLOTS / "figure3_read_observation_prevalence.tsv", sep="\t")
    scores = pd.read_csv(PLOTS / "figure4_sample_heterogeneity.tsv", sep="\t")
    pca_summary = pd.read_csv(EDA / "pca_summary.csv")
    dataset_summary = pd.read_csv(EDA / "dataset_summary.csv")
    outliers = pd.read_csv(EDA / "outlier_summary.csv")
    heterogeneity_summary = pd.read_csv(BIO / "tables/profile_heterogeneity_group_summary.tsv", sep="\t")
    candidates = pd.read_csv(BIO / "tables/candidate_read_observation.tsv", sep="\t")
    breadth = pd.read_csv(BIO / "tables/sample_observation_breadth_summary.tsv", sep="\t")
    return {
        "points": points,
        "sample_mean": sample_mean,
        "zero_fraction": zero_fraction,
        "identity": identity,
        "intersections": intersections,
        "prevalence": prevalence,
        "scores": scores,
        "pca_summary": pca_summary,
        "dataset_summary": dataset_summary,
        "outliers": outliers,
        "heterogeneity_summary": heterogeneity_summary,
        "candidates": candidates,
        "breadth": breadth,
    }


def validate(bundle: dict) -> dict:
    expected_n = {"ROSMAP": (378, 209, 169, 3, 14.5, 7.6), "GSE120584": (1309, 1021, 288, 8, 55.4, 6.8), "GSE215789": (44, 23, 21, 6, 42.1, 7.7)}
    checks = {"completed": [], "not_completed": []}
    pca_summary = bundle["pca_summary"].set_index("dataset")
    dataset_summary = bundle["dataset_summary"].set_index("dataset")
    for dataset, (n, n_ad, n_control, n_flagged, pc1, pc2) in expected_n.items():
        frame = bundle["points"][dataset]
        if len(frame) != n:
            raise RuntimeError(f"{dataset} has {len(frame)} PCA points")
        if int((frame["diagnosis"] == "AD").sum()) != n_ad or int((frame["diagnosis"] == "Control").sum()) != n_control:
            raise RuntimeError(f"{dataset} diagnosis counts changed")
        if int(frame["flagged"].sum()) != n_flagged:
            raise RuntimeError(f"{dataset} flagged count changed")
        shown = (round(100 * float(pca_summary.loc[dataset, "variance_pc1"]), 1), round(100 * float(pca_summary.loc[dataset, "variance_pc2"]), 1))
        if shown != (pc1, pc2):
            raise RuntimeError(f"{dataset} variance labels {shown} do not match the original one-decimal values")
        saved = bundle["outliers"][(bundle["outliers"]["row_type"] == "sample") & (bundle["outliers"]["dataset"] == dataset)]
        flagged = frame[frame["flagged"].astype(int) == 1][["pc1", "pc2"]].to_numpy(dtype=float)
        errors = []
        for x_value, y_value in saved[["pc1", "pc2"]].to_numpy(dtype=float):
            errors.append(float(np.min(np.hypot(flagged[:, 0] - x_value, flagged[:, 1] - y_value))))
        if max(errors) > 1e-6:
            raise RuntimeError(f"{dataset} flagged coordinates disagree with outlier_summary.csv")
        for column, expected in (("n_samples", n), ("n_ad", n_ad), ("n_control", n_control)):
            if int(dataset_summary.loc[dataset, column]) != expected:
                raise RuntimeError(f"{dataset} {column} disagrees with the plotting cohort")
        checks["completed"].append(
            {
                "dataset": dataset,
                "pca_points": n,
                "ad": n_ad,
                "control": n_control,
                "flagged": n_flagged,
                "pc1_percent": pc1,
                "pc2_percent": pc2,
                "max_flagged_coordinate_error": max(errors),
                "sample_id_column": "sample_id" in frame.columns,
            }
        )
        if "sample_id" not in frame.columns:
            checks["not_completed"].append(
                f"{dataset} sample-level PCA identity cannot be confirmed from this plotting table."
            )
    for frame, label in ((bundle["sample_mean"], "sample mean"), (bundle["zero_fraction"], "zero fraction")):
        counts = frame["group"].value_counts().to_dict()
        if counts != {"ROSMAP": 378, "GSE120584": 1309, "GSE215789": 44}:
            raise RuntimeError(f"{label} sample counts changed: {counts}")
        if not np.isfinite(frame["value"]).all():
            raise RuntimeError(f"{label} contains non-finite values")
    checks["completed"].append({"sample_mean_n": 378 + 1309 + 44, "zero_fraction_n": 378 + 1309 + 44})

    identity = bundle["identity"].set_index("dataset")
    expected_identity = {
        "ROSMAP": (309, 95, 8, 189, 17),
        "GSE120584": (2521, 2521, 0, 0, 0),
        "GSE215789": (1137, 1017, 41, 79, 0),
    }
    for dataset, values in expected_identity.items():
        got = tuple(int(identity.loc[dataset, column]) for column in ("n_features", "resolved_mature", "ambiguous", "unresolved", "non_mirna_control"))
        if got != values or sum(got[1:]) != got[0]:
            raise RuntimeError(f"{dataset} identifier classes {got} do not match the validated counts")
    checks["completed"].append({"identifier_classes": {key: values for key, values in expected_identity.items()}})

    exclusive = bundle["intersections"][bundle["intersections"]["membership"].isin(EXCLUSIVE_ORDER)]
    counts = dict(zip(exclusive["membership"], exclusive["n_resolved_mature"].astype(int)))
    expected_exclusive = {
        "GSE120584": 1485,
        "GSE120584+GSE215789": 942,
        "ROSMAP+GSE120584+GSE215789": 58,
        "ROSMAP+GSE120584": 36,
        "GSE215789": 17,
        "ROSMAP": 1,
        "ROSMAP+GSE215789": 0,
    }
    if counts != expected_exclusive:
        raise RuntimeError(f"exclusive intersections changed: {counts}")
    if counts["ROSMAP"] + counts["ROSMAP+GSE120584"] + counts["ROSMAP+GSE215789"] + counts["ROSMAP+GSE120584+GSE215789"] != 95:
        raise RuntimeError("ROSMAP exclusive intersections do not sum to 95")
    if counts["GSE120584"] + counts["GSE120584+GSE215789"] + counts["ROSMAP+GSE120584"] + counts["ROSMAP+GSE120584+GSE215789"] != 2521:
        raise RuntimeError("serum exclusive intersections do not sum to 2521")
    if counts["GSE215789"] + counts["GSE120584+GSE215789"] + counts["ROSMAP+GSE215789"] + counts["ROSMAP+GSE120584+GSE215789"] != 1017:
        raise RuntimeError("plasma exclusive intersections do not sum to 1017")
    pairwise = bundle["intersections"][bundle["intersections"]["membership"].str.startswith("pairwise")]
    pairwise_got = {row.membership: (int(row.n_resolved_mature), round(float(row.jaccard), 3)) for row in pairwise.itertuples()}
    expected_pairwise = {
        "pairwise ROSMAP ∩ GSE120584": (94, 0.037),
        "pairwise ROSMAP ∩ GSE215789": (58, 0.055),
        "pairwise GSE120584 ∩ GSE215789": (1000, 0.394),
    }
    if pairwise_got != expected_pairwise:
        raise RuntimeError(f"pairwise intersections changed: {pairwise_got}")
    checks["completed"].append({"exclusive_intersections": expected_exclusive, "pairwise": expected_pairwise})

    prevalence = bundle["prevalence"]
    if set(prevalence["dataset"]) != {"GSE215789"}:
        raise RuntimeError("read-observation table is not restricted to GSE215789")
    if not prevalence["definition"].str.contains("count > 0").all():
        raise RuntimeError("read-observation definition changed")
    if prevalence.groupby("diagnosis").size().to_dict() != {"AD": 1137, "Control": 1137}:
        raise RuntimeError("read-observation feature counts changed")
    checks["completed"].append({"read_observation_features_per_diagnosis": 1137, "definition": "count > 0"})

    for dataset, expected_features in (("ROSMAP", 309), ("GSE120584", 2521)):
        subset = bundle["scores"][bundle["scores"]["dataset"] == dataset]
        if subset.groupby("diagnosis").size().to_dict() != ({"ROSMAP": {"AD": 209, "Control": 169}, "GSE120584": {"AD": 1021, "Control": 288}}[dataset]):
            raise RuntimeError(f"{dataset} dissimilarity sample counts changed")
        if not (subset["n_eligible_features"] == expected_features).all():
            raise RuntimeError(f"{dataset} eligible-feature count changed")
        for diagnosis in ("Control", "AD"):
            values = subset.loc[subset["diagnosis"] == diagnosis, "heterogeneity"].to_numpy(dtype=float)
            saved = bundle["heterogeneity_summary"]
            row = saved[(saved["dataset"] == dataset) & (saved["diagnosis"] == diagnosis)].iloc[0]
            if abs(float(np.median(values)) - float(row["median"])) > 1e-9:
                raise RuntimeError(f"{dataset} {diagnosis} plotted median disagrees with the saved summary")
            if len(values) != int(row["n"]):
                raise RuntimeError(f"{dataset} {diagnosis} plotted n disagrees with the saved summary")
    if (bundle["scores"]["dataset"] == "GSE215789").any():
        raise RuntimeError("GSE215789 dissimilarity scores must not be plotted")
    checks["completed"].append({"dissimilarity": "ROSMAP 169/209 and GSE120584 288/1021 match saved medians within 1e-9"})
    return checks


def main() -> None:
    REPORTS.mkdir(parents=True, exist_ok=True)
    bundle = load()
    checks = validate(bundle)
    structure = figure_structure(bundle["points"], bundle["pca_summary"], bundle["sample_mean"], bundle["zero_fraction"])
    structure_problems = layout_problems(structure)
    coverage = figure_coverage(bundle["identity"], bundle["intersections"], bundle["prevalence"], bundle["scores"], bundle["heterogeneity_summary"])
    coverage_problems = layout_problems(coverage)
    structure_png = export(structure, "figure_eda_01_dataset_structure")
    coverage_png = export(coverage, "figure_eda_02_coverage_heterogeneity")
    write_captions(bundle)
    preview_sizes = {
        "figure_eda_01": write_previews(structure_png, "figure_eda_01"),
        "figure_eda_02": write_previews(coverage_png, "figure_eda_02"),
    }
    payload = {"checks": checks, "layout_problems": {"figure_eda_01": structure_problems, "figure_eda_02": coverage_problems}, "preview_sizes": preview_sizes}
    (REPORTS / "composite_numeric_checks.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
