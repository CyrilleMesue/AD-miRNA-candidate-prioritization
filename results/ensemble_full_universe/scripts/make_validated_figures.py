#!/usr/bin/env python
"""Figures for the validated full-universe ensemble analysis.

The atlas heatmap row-centers values and turns missing cells into zeros.
Rank percentiles and unmeasured datasets cannot be drawn that way, so these
figures are drawn directly from the validated tables.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import Normalize
from matplotlib.patches import Patch

ROOT = Path(__file__).resolve().parents[3]
OUT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "third_party"))

from bioinformatics_visual_evidence_atlas.visualization.recommend import recommend
from ensemble_common import AGGREGATORS, CANDIDATES, DATASETS

plt.rcParams.update(
    {
        "font.family": "DejaVu Sans",
        "font.size": 9,
        "axes.titlesize": 11,
        "axes.titleweight": "semibold",
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    }
)

STATUS_COLOR = {
    "resolved mature": "#0072B2",
    "unresolved": "#E69F00",
    "ambiguous": "#CC79A7",
    "non-miRNA/control": "#999999",
}
DATASET_COLOR = {"ROSMAP": "#0072B2", "GSE120584": "#E69F00", "GSE215789": "#009E73"}


def save(fig: plt.Figure, folder: Path, stem: str) -> None:
    folder.mkdir(parents=True, exist_ok=True)
    for suffix in (".png", ".svg", ".pdf"):
        fig.savefig(folder / f"{stem}{suffix}", dpi=300, bbox_inches="tight")
    plt.close(fig)


def figure1(folder: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(11.4, 7.2), constrained_layout=True)
    for ax, dataset, letter in zip(axes, DATASETS, "ABC"):
        table = pd.read_csv(OUT / "dataset_ensembles" / dataset / "dataset_ensemble_summary.csv")
        top = table.nsmallest(15, "dataset_summary_rank").iloc[::-1]
        colors = [STATUS_COLOR.get(status, "#999999") for status in top["identifier_resolution_status"]]
        values = 1.0 - top["median_rank_percentile"].to_numpy()
        ax.barh(top["original_mirna_id"], values, color=colors, height=0.72)
        ax.set_xlabel("1 − median rank percentile")
        ax.set_title(f"{dataset}\n{EXPECTED_LABEL[dataset]}")
        ax.set_xlim(0.92, 1.008)
        ax.text(-0.18, 1.06, letter, transform=ax.transAxes, fontsize=14, fontweight="bold", va="bottom")
    handles = [
        Patch(facecolor=STATUS_COLOR["resolved mature"], label="Resolved mature miRNA"),
        Patch(facecolor=STATUS_COLOR["unresolved"], label="Unresolved identifier"),
        Patch(facecolor=STATUS_COLOR["ambiguous"], label="Ambiguous identifier"),
    ]
    fig.legend(handles=handles, loc="outside lower center", ncol=3, frameon=False)
    fig.suptitle("Dataset-specific ensemble priority (median of 12 rank percentiles)", fontsize=13)
    save(fig, folder, "figure1_dataset_ensemble_leaders")


EXPECTED_LABEL = {
    "ROSMAP": "309 measured miRNAs",
    "GSE120584": "2,521 measured miRNAs",
    "GSE215789": "1,137 measured miRNAs",
}


def heatmap(ax, frame: pd.DataFrame, title: str, letter: str) -> None:
    columns = ["percentile_ROSMAP", "percentile_GSE120584", "percentile_GSE215789"]
    labels = ["ROSMAP", "GSE120584", "GSE215789"]
    matrix = frame[columns].to_numpy(dtype=float)
    masked = np.ma.masked_invalid(matrix)
    cmap = plt.cm.Blues_r.copy()
    cmap.set_bad("#D9D9D9")
    image = ax.imshow(masked, aspect="auto", cmap=cmap, norm=Normalize(0, 1))
    ax.set_xticks(range(3), labels, rotation=20, ha="right")
    ax.set_yticks(range(len(frame)), frame["resolved_mature_mirna_id"])
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            value = matrix[i, j]
            if value != value:
                text = "—"
            elif value < 0.02:
                text = f"{value:.3f}"
            else:
                text = f"{value:.2f}"
            color = "white" if value == value and value < 0.35 else "black"
            ax.text(j, i, text, ha="center", va="center", fontsize=7, color=color)
    ax.set_title(title)
    ax.text(-0.22, 1.04, letter, transform=ax.transAxes, fontsize=14, fontweight="bold", va="bottom")
    return image


def figure2(folder: Path) -> None:
    integrated = pd.read_csv(OUT / "cross_dataset" / "primary_integration.csv")
    primary = integrated.nsmallest(20, "primary_rank")
    coverage3 = integrated[integrated["n_datasets"] == 3].nsmallest(15, "stratum_rank")
    fig, axes = plt.subplots(1, 2, figsize=(11.2, 8.4), constrained_layout=True, gridspec_kw={"width_ratios": [1.15, 1]})
    image = heatmap(
        axes[0],
        primary,
        "Primary mean, top 20\nGray cells were not measured",
        "A",
    )
    heatmap(
        axes[1],
        coverage3,
        "Coverage = 3, top 15 of 58\nSeparate stratum, same percentile",
        "B",
    )
    colorbar = fig.colorbar(image, ax=axes, fraction=0.03, pad=0.02)
    colorbar.set_label("Dataset median rank percentile (0 = best)")
    fig.suptitle("Cross-dataset priority with unmeasured datasets left blank", fontsize=13)
    save(fig, folder, "figure2_cross_dataset_coverage")
    print("PRIMARY_COVERAGE", primary["n_datasets"].value_counts().to_dict())
    leader = integrated[integrated["resolved_mature_mirna_id"] == "hsa-miR-208a-5p"]
    if not leader.empty:
        row = leader.iloc[0]
        print("SERUM_LEADER", int(row["primary_rank"]), int(row["n_datasets"]), row["datasets"], float(row["mean_available_percentile"]))


def figure3(folder: Path) -> None:
    ranks = pd.read_csv(OUT / "candidates" / "candidate_dataset_ensemble_ranks.csv")
    fig, axes = plt.subplots(3, 1, figsize=(10.5, 8.6), sharex=True, sharey=True, constrained_layout=True)
    positions = np.arange(len(AGGREGATORS))
    for ax, candidate, letter in zip(axes, CANDIDATES, "ABC"):
        for dataset in DATASETS:
            subset = ranks[(ranks["candidate"] == candidate) & (ranks["dataset"] == dataset)]
            subset = subset.set_index("aggregator").loc[list(AGGREGATORS)]
            ax.plot(
                positions,
                subset["rank_percentile"],
                marker="o",
                color=DATASET_COLOR[dataset],
                label=dataset,
                linewidth=1.4,
            )
        ax.set_ylim(1.02, -0.02)
        ax.set_ylabel("Rank percentile")
        ax.set_title(candidate)
        ax.text(-0.08, 1.08, letter, transform=ax.transAxes, fontsize=14, fontweight="bold", va="bottom")
    axes[-1].set_xticks(positions, AGGREGATORS, rotation=40, ha="right")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside lower center", ncol=3, frameon=False)
    fig.suptitle("Pre-specified candidates across 12 ensembles", fontsize=13)
    axes[0].set_title("hsa-miR-324-3p\nPercentile within the measured dataset, not expression")
    save(fig, folder, "figure3_candidate_ensemble_stability")


def main() -> None:
    folder = OUT / "figures"
    advice = recommend(
        "Which miRNAs have the best ensemble rank percentiles, and which datasets did not measure them?",
        ["feature", "sample", "value"],
    )
    (folder).mkdir(parents=True, exist_ok=True)
    (folder / "atlas_recommendation.json").write_text(json.dumps(advice, indent=2))
    figure1(folder)
    figure2(folder)
    figure3(folder)
    print("FIGURES_DONE", advice.get("renderer"))


if __name__ == "__main__":
    main()
