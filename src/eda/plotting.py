"""Thesis figures drawn with the visualization skill's palette, axes, and exporters.

Class counts are not sent to grouped_bars. That renderer always runs a Welch
test, and a census of sample counts is not a set of replicate measurements.
PCA is drawn from a column-centered SVD on the full matrix. The skill's
pca_scatter renderer expects one row per sample-feature pair, which is about
3.3 million rows for GSE120584.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from bioinformatics_visual_evidence_atlas.visualization.figures import RENDERERS, _panel_letter
from bioinformatics_visual_evidence_atlas.visualization.style import legend, palette, save_figure, style_axes

DIAGNOSIS_ORDER = ["Control", "AD"]


def plot_class_counts(summary, folder: Path, stem: str = "class_distribution") -> dict:
    order = ["ROSMAP", "GSE120584", "GSE215789"]
    colors = palette(DIAGNOSIS_ORDER)
    lookup = summary.set_index("dataset")
    fig, ax = plt.subplots(figsize=(7.2, 4.6), dpi=120)
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")
    positions = np.arange(len(order))
    width = 0.36
    for index, diagnosis in enumerate(DIAGNOSIS_ORDER):
        column = "n_control" if diagnosis == "Control" else "n_ad"
        heights = [int(lookup.loc[name, column]) for name in order]
        offset = (index - 0.5) * width
        bars = ax.bar(positions + offset, heights, width=width, color=colors[diagnosis], label=diagnosis)
        for bar, height in zip(bars, heights):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                height,
                str(height),
                ha="center",
                va="bottom",
                fontsize=9,
                clip_on=False,
            )
    ax.set_xticks(positions, order)
    ax.set_ylim(0, float(lookup[["n_ad", "n_control"]].to_numpy().max()) * 1.14)
    style_axes(ax, "AD and Control sample counts", "Dataset", "Samples")
    legend(ax)
    paths = save_figure(fig, folder, stem)
    audit = {
        "renderer": "count bars using the skill palette; grouped_bars was not used",
        "method": "Direct sample counts. No mean, SEM, or hypothesis test.",
        "synthetic": False,
        "files": paths,
    }
    _write_audit(folder, stem, audit)
    return audit


def plot_expression(sample_mean: list[dict], zero_fraction: list[dict], folder: Path, stem: str = "expression_distribution") -> dict:
    fig, axes = plt.subplots(2, 1, figsize=(7.2, 6.8), dpi=120)
    fig.patch.set_facecolor("white")
    panels = [
        (axes[0], sample_mean, "Sample mean of min-max scaled values", "Sample mean", "A"),
        (axes[1], zero_fraction, "Per-sample fraction of exact zeros", "Fraction of zeros", "B"),
    ]
    audits = []
    for ax, rows, title, ylabel, letter in panels:
        ax.set_facecolor("white")
        _fig, audit = RENDERERS["grouped_boxes"](rows, title, ax)
        _restyle_boxes(ax, ylabel)
        _panel_letter(ax, letter)
        audits.append(audit)
    fig.subplots_adjust(left=0.16, right=0.98, top=0.94, bottom=0.08, hspace=0.42)
    paths = save_figure(fig, folder, stem)
    record = {
        "renderer": "grouped_boxes",
        "method": "Quartile boxes of one summary per sample. No significance marker.",
        "synthetic": False,
        "panels": [
            {"panel": "A", "summary": "sample mean", "n_rows": len(sample_mean)},
            {"panel": "B", "summary": "exact-zero fraction", "n_rows": len(zero_fraction)},
        ],
        "ylabel_override": "The skill renderer labels the y-axis 'Value'. Thesis axes name the summary.",
        "files": paths,
        "panel_audits": audits,
    }
    _write_audit(folder, stem, record)
    return record


def _restyle_boxes(ax, ylabel: str) -> None:
    """Keep the skill boxplot, then name the axes and separate medians from group color.

    grouped_boxes labels both axes in generic words and draws medians in Matplotlib
    orange, which is the same hue as the second Okabe–Ito group.
    """
    ax.set_xlabel("Dataset", fontsize=10, fontweight="bold")
    ax.set_ylabel(ylabel, fontsize=10, fontweight="bold")
    for line in ax.lines:
        marker = line.get_marker()
        if marker in ("o", "O", "."):
            line.set_markersize(3.2)
            line.set_markeredgewidth(0.45)
            line.set_alpha(0.75)
        elif str(line.get_color()).lower() in {"orange", "c1", "#ff7f0e"}:
            line.set_color("#000000")
            line.set_linewidth(1.15)


def plot_pca(name: str, fitted: dict, folder: Path) -> dict:
    scores = fitted["scores"]
    explained = fitted["explained"]
    diagnosis = fitted["diagnosis"]
    flagged = fitted["flagged"]
    colors = palette(DIAGNOSIS_ORDER)
    fig, ax = plt.subplots(figsize=(6.4, 5.4), dpi=120)
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")
    size = 36 if len(diagnosis) < 100 else 22 if len(diagnosis) < 500 else 14
    alpha = 1.0 if len(diagnosis) < 100 else 0.85 if len(diagnosis) < 500 else 0.55
    for label in DIAGNOSIS_ORDER:
        chosen = diagnosis == label
        ax.scatter(
            scores[chosen, 0],
            scores[chosen, 1],
            c=colors[label],
            label=label,
            s=size,
            alpha=alpha,
            linewidths=0,
        )
    if flagged.any():
        ax.scatter(
            scores[flagged, 0],
            scores[flagged, 1],
            facecolors="none",
            edgecolors="#000000",
            s=size + 28,
            linewidths=0.8,
            label="Flagged",
        )
    ax.set_aspect("equal", adjustable="datalim")
    style_axes(
        ax,
        f"{name} (n = {len(diagnosis)})",
        f"PC1 ({100 * explained[0]:.1f}% variance)",
        f"PC2 ({100 * explained[1]:.1f}% variance)",
    )
    legend(ax)
    stem = "pca_" + name.lower()
    paths = save_figure(fig, folder, stem)
    audit = {
        "renderer": "pca_scatter mathematics via column-centered SVD; full matrix, not long-form rows",
        "method": "PCA by singular value decomposition. Diagnosis colors points after the fit. No ellipse.",
        "synthetic": False,
        "variance_pc1": float(explained[0]),
        "variance_pc2": float(explained[1]),
        "n_samples": int(len(diagnosis)),
        "point_size": size,
        "point_alpha": alpha,
        "n_flagged": int(flagged.sum()),
        "files": paths,
    }
    _write_audit(folder, stem, audit)
    return audit


def _write_audit(folder: Path, stem: str, audit: dict) -> None:
    folder.mkdir(parents=True, exist_ok=True)
    (folder / f"{stem}_audit.json").write_text(json.dumps(audit, indent=2) + "\n", encoding="utf-8")
