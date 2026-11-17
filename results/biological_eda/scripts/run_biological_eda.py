#!/usr/bin/env python
"""Literature-guided biological EDA of the three original AD miRNA datasets.

Reads the ranking-cohort matrices and linked metadata. Does not modify
original files, rankings, or feature-selection rules.

python results/biological_eda/scripts/run_biological_eda.py
"""
from __future__ import annotations

import hashlib
import json
import os
import platform
import sys
from importlib.metadata import version
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
from scipy.stats import rankdata

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "third_party"))

from prioritization.matrices import load_ranking_datasets

ORIGINAL = Path(os.environ.get("ORIGINAL_PROJECT", "../Alzheimers-Multi-Omics-Analysis"))
PROCESSED = {
    "ROSMAP": ROOT / "data/processed/ROSMAP_brain_norm.csv",
    "GSE120584": ROOT / "data/processed/GSE120584_serum_norm.csv",
    "GSE215789": ROOT / "data/processed/GSE215789_plasma_count.csv",
}
ALIAS_MAP = ORIGINAL / "artifacts/mirBase_to_mirTARBase.json"
CLINICAL = ORIGINAL / "downloads/ROSMAP_clinical.csv"
BIOSPECIMEN = ORIGINAL / "downloads/ROSMAP_biospecimen_metadata.csv"
ASSAY_META = ORIGINAL / "downloads/microRNA/ROSMAP_assay_miRNAarray_nanostring_metadata.csv"
GCT = ORIGINAL / "downloads/microRNA/ROSMAP_arraymiRNA.gct"

OUT = ROOT / "results/biological_eda"
FIG = OUT / "figures"
TAB = OUT / "tables"
SUP = OUT / "supporting_data"
REP = OUT / "reports"

CANDIDATES = ("hsa-miR-324-3p", "hsa-miR-574-3p", "hsa-miR-769-3p")
META_DROP = {
    "ROSMAP": ["Gender", "Race", "PMI", "Braak", "Diagnosis"],
    "GSE120584": ["Diagnosis", "Age", "Gender", "APOE4"],
    "GSE215789": ["Gender", "Age", "Diagnosis"],
}
CONTROL_COLOR = "#0072B2"
AD_COLOR = "#E69F00"
DATASET_ORDER = ("ROSMAP", "GSE120584", "GSE215789")
DATASET_LABEL = {
    "ROSMAP": "ROSMAP DLPFC",
    "GSE120584": "GSE120584 serum",
    "GSE215789": "GSE215789 plasma",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def fmt(value, digits=3):
    if value is None or (isinstance(value, float) and not np.isfinite(value)):
        return "NA"
    if isinstance(value, (int, np.integer)):
        return str(int(value))
    return f"{float(value):.{digits}f}"


def quantiles(values) -> dict:
    array = np.asarray(list(values), dtype=float)
    array = array[np.isfinite(array)]
    if array.size == 0:
        return {"n": 0, "median": np.nan, "q1": np.nan, "q3": np.nan, "minimum": np.nan, "maximum": np.nan}
    return {
        "n": int(array.size),
        "median": float(np.quantile(array, 0.50, method="linear")),
        "q1": float(np.quantile(array, 0.25, method="linear")),
        "q3": float(np.quantile(array, 0.75, method="linear")),
        "minimum": float(array.min()),
        "maximum": float(array.max()),
    }


def load_alias_map(path: Path) -> dict[str, list[str]]:
    with path.open() as handle:
        raw = json.load(handle)
    inverse: dict[str, list[str]] = {}
    for accession, name in raw.items():
        inverse.setdefault(str(name), []).append(str(accession))
    if any(len(accessions) != 1 for accessions in inverse.values()):
        raise RuntimeError("Alias map is not one product name to one accession.")
    return inverse


def classify_feature(name: str, inverse: dict[str, list[str]]) -> tuple[str, str, str]:
    if not str(name).startswith("hsa-"):
        return (
            "non-miRNA/control",
            "",
            "Non-human original feature. Excluded from the human mature intersection.",
        )
    if "/" in name or "_" in name:
        return (
            "ambiguous",
            "",
            "Identifier joins more than one miRNA name. It was not split into arms.",
        )
    if ".SNP" in name or "SNP" in name:
        return (
            "unresolved",
            "",
            "Variant suffix. Not treated as an unmodified mature identifier.",
        )
    accessions = inverse.get(name, [])
    if len(accessions) == 1:
        return (
            "resolved mature",
            name,
            f"One-to-one product name for {accessions[0]} in mirBase_to_mirTARBase.json.",
        )
    return (
        "unresolved",
        "",
        "No one-to-one accession in the project alias map. Arm identity was not inferred.",
    )


def ranking_labels(diagnosis: pd.Series, collapse_ad: bool) -> pd.Series:
    labels = diagnosis.astype(str)
    if collapse_ad:
        labels = labels.map(lambda value: "AD" if "AD" in value else value)
    return labels


def read_processed(dataset: str) -> pd.DataFrame:
    frame = pd.read_csv(PROCESSED[dataset])
    if "Sample ID" not in frame.columns:
        raise RuntimeError(f"{dataset} has no Sample ID column")
    frame = frame.set_index("Sample ID")
    if not frame.index.is_unique:
        raise RuntimeError(f"{dataset} sample IDs are not unique")
    return frame


def assert_alignment(expression: pd.DataFrame, labels: pd.Series, ranked: pd.DataFrame, dataset: str) -> None:
    if not expression.index.equals(ranked.index):
        raise RuntimeError(f"{dataset} sample IDs do not match the ranking matrix")
    if not labels.index.equals(ranked.index):
        raise RuntimeError(f"{dataset} labels are not aligned to the ranking matrix")
    if not labels.astype(str).equals(ranked["Diagnosis"].astype(str)):
        raise RuntimeError(f"{dataset} diagnosis labels disagree with the ranking matrix")
    missing = [feature for feature in ranked.columns if feature != "Diagnosis" and feature not in expression.columns]
    if missing:
        raise RuntimeError(f"{dataset} is missing {len(missing)} ranking features in the pre-scale matrix")


def heterogeneity_scores(expression: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    numeric = expression.apply(pd.to_numeric, errors="coerce")
    finite = np.isfinite(numeric.to_numpy(dtype=float)).all(axis=0)
    varying = numeric.nunique(dropna=True).to_numpy() > 1
    eligible_mask = finite & varying
    excluded = {
        "non_finite": int((~finite).sum()),
        "zero_variability": int((finite & ~varying).sum()),
        "eligible": int(eligible_mask.sum()),
        "input_features": int(numeric.shape[1]),
    }
    if excluded["eligible"] < 30:
        return pd.DataFrame(), excluded
    used = numeric.loc[:, eligible_mask]
    values = used.to_numpy(dtype=float)
    ranks = rankdata(values, axis=1, method="average")
    rho = np.corrcoef(ranks)
    dissimilarity = 1.0 - rho
    np.fill_diagonal(dissimilarity, np.nan)
    scores = np.nanquantile(dissimilarity, 0.50, axis=1, method="linear")
    if not np.isfinite(scores).all():
        raise RuntimeError("A within-group Spearman median is undefined")
    zero_fraction = (values == 0).mean(axis=1)
    distinct = np.array([len(np.unique(row)) for row in values], dtype=int)
    out = pd.DataFrame(
        {
            "sample_id": used.index.astype(str),
            "heterogeneity": scores,
            "zero_value_fraction": zero_fraction,
            "n_distinct_values": distinct,
            "n_eligible_features": excluded["eligible"],
        }
    )
    return out, excluded


def link_rosmap_extra(sample_ids: pd.Index) -> tuple[pd.DataFrame, list[dict]]:
    notes = []
    individuals = pd.Series(sample_ids.astype(str).str.replace("ROSMAP_", "", regex=False), index=sample_ids)
    if individuals.duplicated().any():
        raise RuntimeError("ROSMAP individual IDs are duplicated after removing the prefix")
    clinical = pd.read_csv(CLINICAL, usecols=["individualID", "age_death", "apoe_genotype"])
    if clinical["individualID"].duplicated().any():
        raise RuntimeError("ROSMAP clinical individualID is not unique")
    clinical = clinical.set_index("individualID")
    missing_clinical = [item for item in individuals if item not in clinical.index]
    if missing_clinical:
        raise RuntimeError(f"{len(missing_clinical)} ranking individuals are absent from ROSMAP clinical")
    linked = pd.DataFrame(index=sample_ids)
    linked["age_at_death_years"] = clinical.loc[individuals.to_numpy(), "age_death"].to_numpy()
    linked["apoe_genotype"] = clinical.loc[individuals.to_numpy(), "apoe_genotype"].astype(str).to_numpy()
    notes.append(
        {
            "variable": "age_at_death_years",
            "dataset": "ROSMAP",
            "source": str(CLINICAL),
            "join_key": "individualID from Sample ID prefix ROSMAP_",
            "status": "linked",
            "note": "One clinical row per ranking individual. Age is age at death. The source code 90+ means 90 years or older and is not recoded to 90. Quantiles use numeric ages only.",
        }
    )
    notes.append(
        {
            "variable": "apoe_genotype",
            "dataset": "ROSMAP",
            "source": str(CLINICAL),
            "join_key": "individualID from Sample ID prefix ROSMAP_",
            "status": "linked",
            "note": "Stored genotype code. Not recoded.",
        }
    )

    biospecimen = pd.read_csv(BIOSPECIMEN, usecols=["individualID", "specimenID"]).dropna()
    if biospecimen.groupby("specimenID")["individualID"].nunique().max() != 1:
        raise RuntimeError("A ROSMAP specimen maps to more than one individual")
    specimen_to_individual = biospecimen.drop_duplicates("specimenID").set_index("specimenID")["individualID"]
    assay = pd.read_csv(ASSAY_META)
    assay["individualID"] = assay["specimenID"].map(specimen_to_individual)
    assay = assay[assay["individualID"].isin(set(individuals))]
    if assay["individualID"].duplicated().any() or assay["individualID"].nunique() != len(sample_ids):
        notes.append(
            {
                "variable": "RIN",
                "dataset": "ROSMAP",
                "source": str(ASSAY_META),
                "join_key": "specimenID to individualID",
                "status": "NOT_ASSESSABLE",
                "note": "Assay metadata is not one row per ranking individual.",
            }
        )
        return linked, notes
    order = assay.set_index("individualID").loc[individuals.to_numpy()]
    for column, label in (
        ("RIN", "RIN"),
        ("rnaBatch", "rna_batch"),
        ("libraryBatch", "library_batch"),
        ("nanostringBatch", "nanostring_batch"),
        ("bindingDensity", "binding_density"),
        ("plate", "plate"),
    ):
        values = order[column]
        if values.notna().sum() == 0:
            notes.append(
                {
                    "variable": label,
                    "dataset": "ROSMAP",
                    "source": str(ASSAY_META),
                    "join_key": "specimenID to individualID",
                    "status": "NOT_ASSESSABLE",
                    "note": "Column is present but missing for every ranking sample.",
                }
            )
            continue
        linked[label] = values.to_numpy()
        notes.append(
            {
                "variable": label,
                "dataset": "ROSMAP",
                "source": str(ASSAY_META),
                "join_key": "specimenID to individualID",
                "status": "linked",
                "note": "One assay row per ranking individual.",
            }
        )
    return linked, notes


def group_rows(dataset: str, labels: pd.Series, frame: pd.DataFrame, kinds: dict[str, str]) -> list[dict]:
    rows = []
    for diagnosis in ("AD", "Control"):
        subset = frame.loc[labels == diagnosis]
        for variable, kind in kinds.items():
            if variable not in subset.columns:
                continue
            series = subset[variable]
            if kind == "continuous":
                stats = quantiles(pd.to_numeric(series, errors="coerce"))
                n_missing = int(pd.to_numeric(series, errors="coerce").isna().sum())
                for statistic in ("median", "q1", "q3", "minimum", "maximum"):
                    rows.append(
                        {
                            "dataset": dataset,
                            "diagnosis": diagnosis,
                            "variable": variable,
                            "kind": kind,
                            "level": "",
                            "statistic": statistic,
                            "n_samples": int(len(subset)),
                            "n_nonmissing": stats["n"],
                            "n_missing": n_missing,
                            "value": stats[statistic],
                        }
                    )
            else:
                text = series.astype(str)
                n_missing = int(series.isna().sum())
                counts = text.value_counts(dropna=True)
                for level, count in counts.items():
                    rows.append(
                        {
                            "dataset": dataset,
                            "diagnosis": diagnosis,
                            "variable": variable,
                            "kind": kind,
                            "level": level,
                            "statistic": "count",
                            "n_samples": int(len(subset)),
                            "n_nonmissing": int(text.notna().sum()),
                            "n_missing": n_missing,
                            "value": int(count),
                        }
                    )
    return rows


def exclusive_counts(sets: dict[str, set[str]]) -> list[dict]:
    labels = list(DATASET_ORDER)
    rows = []
    for mask in range(1, 8):
        present = [labels[i] for i in range(3) if mask & (1 << i)]
        absent = [labels[i] for i in range(3) if not mask & (1 << i)]
        shared = set.intersection(*(sets[name] for name in present))
        for name in absent:
            shared -= sets[name]
        rows.append(
            {
                "ROSMAP": int("ROSMAP" in present),
                "GSE120584": int("GSE120584" in present),
                "GSE215789": int("GSE215789" in present),
                "n_resolved_mature": len(shared),
                "membership": "+".join(present),
            }
        )
    return rows


def jaccard(left: set[str], right: set[str]) -> float:
    union = left | right
    if not union:
        return float("nan")
    return len(left & right) / len(union)


def draw_figures(exclusive: pd.DataFrame, resolved_sets: dict[str, set[str]], prevalence: pd.DataFrame, heterogeneity: pd.DataFrame, candidate_prevalence: pd.DataFrame) -> None:
    FIG.mkdir(parents=True, exist_ok=True)
    figure1(exclusive, resolved_sets, prevalence, candidate_prevalence)
    figure2(heterogeneity)


def figure1(exclusive, resolved_sets, prevalence, candidate_prevalence) -> None:
    figure = plt.figure(figsize=(12.4, 7.8), dpi=120)
    figure.patch.set_facecolor("white")
    bar_ax = figure.add_axes([0.34, 0.58, 0.26, 0.32])
    dot_ax = figure.add_axes([0.34, 0.36, 0.26, 0.18])
    size_ax = figure.add_axes([0.08, 0.36, 0.16, 0.18], sharey=dot_ax)
    rosmap_ax = figure.add_axes([0.68, 0.73, 0.28, 0.16])
    serum_ax = figure.add_axes([0.68, 0.50, 0.28, 0.16])
    plasma_ax = figure.add_axes([0.68, 0.16, 0.28, 0.24])

    ordered = exclusive.sort_values(["n_resolved_mature", "membership"], ascending=[False, True])
    positions = np.arange(len(ordered))
    heights = ordered["n_resolved_mature"].to_numpy(dtype=float)
    bar_ax.bar(positions, heights, color="#009E73", width=0.72)
    bar_ax.set_ylabel("Resolved mature miRNAs")
    bar_ax.tick_params(axis="x", labelbottom=False, length=0)
    bar_ax.set_title("A   Exclusive mature-miRNA intersections", loc="left", fontweight="bold", fontsize=11)
    bar_ax.set_ylim(0, max(heights.max() * 1.16, 1))
    bar_ax.set_xlim(-0.6, len(ordered) - 0.4)
    for xpos, height in zip(positions, heights):
        bar_ax.text(xpos, height + heights.max() * 0.02, str(int(height)), ha="center", va="bottom", fontsize=7)
    bar_ax.spines["top"].set_visible(False)
    bar_ax.spines["right"].set_visible(False)

    set_names = ["ROSMAP", "GSE120584", "GSE215789"]
    set_y = np.array([2, 1, 0])
    set_sizes = [len(resolved_sets[name]) for name in DATASET_ORDER]
    size_ax.barh(set_y, set_sizes, color="#0072B2", height=0.55)
    size_ax.set_xlabel("Set size")
    size_ax.tick_params(axis="y", labelleft=False, length=0)
    size_ax.tick_params(axis="x", labelsize=7)
    size_ax.spines["top"].set_visible(False)
    size_ax.spines["right"].set_visible(False)
    for ypos, size in zip(set_y, set_sizes):
        size_ax.text(size + max(set_sizes) * 0.03, ypos, str(size), va="center", fontsize=7)
    size_ax.set_xlim(0, max(set_sizes) * 1.28)

    for column, (_, row) in enumerate(ordered.iterrows()):
        active = []
        for row_index, dataset in enumerate(DATASET_ORDER):
            present = int(row[dataset]) == 1
            y_pos = set_y[row_index]
            if present:
                active.append(y_pos)
            dot_ax.scatter(
                column,
                y_pos,
                s=36,
                color="#009E73" if present else "white",
                edgecolor="#333333",
                linewidth=0.6,
                zorder=3,
            )
        if len(active) > 1:
            dot_ax.plot([column, column], [min(active), max(active)], color="#333333", linewidth=1, zorder=2)
    dot_ax.set_ylim(-0.6, 2.6)
    dot_ax.set_yticks(set_y, set_names)
    dot_ax.set_xlim(-0.6, len(ordered) - 0.4)
    dot_ax.tick_params(axis="y", pad=2)
    dot_ax.tick_params(axis="x", labelbottom=False, length=0)
    for spine in ("top", "right", "left", "bottom"):
        dot_ax.spines[spine].set_visible(False)

    notes = {
        rosmap_ax: (
            "B   ROSMAP DLPFC",
            "Unavailable\n\nDetection calls are not in the NanoString GCT\nor assay metadata. Normalized values\nwere not used as detection.",
        ),
        serum_ax: (
            "GSE120584 serum",
            "Unavailable\n\nThe GEO series matrix has no detection-flag\ncolumn. Normalized intensities were not\nused as detection.",
        ),
    }
    for axis, (title, text) in notes.items():
        axis.axis("off")
        axis.set_title(title, loc="left", fontweight="bold", fontsize=10)
        axis.text(0.0, 0.15, text, transform=axis.transAxes, va="bottom", fontsize=8)

    for diagnosis, color in (("Control", CONTROL_COLOR), ("AD", AD_COLOR)):
        values = prevalence.loc[prevalence["diagnosis"] == diagnosis, "read_observation_prevalence"].to_numpy(dtype=float)
        ordered_values = np.sort(values)
        fraction = np.arange(1, ordered_values.size + 1) / ordered_values.size
        plasma_ax.step(ordered_values, fraction, where="post", color=color, label=diagnosis, linewidth=1.6)
    plasma_ax.set_xlim(0, 1)
    plasma_ax.set_ylim(0, 1)
    plasma_ax.set_xlabel("Fraction of samples with count > 0")
    plasma_ax.set_ylabel("Fraction of features")
    plasma_ax.set_title("GSE215789 read observation", loc="left", fontweight="bold", fontsize=10)
    plasma_ax.legend(frameon=False, fontsize=8, loc="lower right")
    plasma_ax.spines["top"].set_visible(False)
    plasma_ax.spines["right"].set_visible(False)
    lines = []
    for candidate in CANDIDATES:
        match = candidate_prevalence[candidate_prevalence["feature_id"] == candidate]
        if match.empty:
            lines.append(f"{candidate}: absent from retained matrix")
            continue
        bits = []
        for diagnosis in ("AD", "Control"):
            row = match.loc[match["diagnosis"] == diagnosis].iloc[0]
            bits.append(f"{diagnosis} {int(row['n_observed'])}/{int(row['n_samples'])}")
        lines.append(f"{candidate}: " + ", ".join(bits))
    plasma_ax.text(0.0, -0.48, "Read-observed candidates (count > 0)\n" + "\n".join(lines), transform=plasma_ax.transAxes, fontsize=7, va="top", clip_on=False)

    figure.savefig(FIG / "figure1_coverage_and_observation.png", dpi=300, bbox_inches="tight", facecolor="white")
    figure.savefig(FIG / "figure1_coverage_and_observation.svg", bbox_inches="tight", facecolor="white")
    figure.savefig(FIG / "figure1_coverage_and_observation.pdf", bbox_inches="tight", facecolor="white")
    plt.close(figure)
    caption = (
        "Figure 1. Mature-miRNA coverage uses only identifiers with one accession in the project alias map. "
        "Panel B is read-observation prevalence (count > 0) for GSE215789. "
        "NanoString and microarray detection calls were not available, so those panels are unmarked."
    )
    (FIG / "figure1_coverage_and_observation_caption.txt").write_text(caption + "\n")


def figure2(heterogeneity: pd.DataFrame) -> None:
    figure, axes = plt.subplots(1, 3, figsize=(11.2, 4.6), dpi=120)
    figure.patch.set_facecolor("white")
    for axis, dataset in zip(axes[:2], ("ROSMAP", "GSE120584")):
        subset = heterogeneity[heterogeneity["dataset"] == dataset]
        groups = []
        colors = []
        for diagnosis, color in (("Control", CONTROL_COLOR), ("AD", AD_COLOR)):
            values = subset.loc[subset["diagnosis"] == diagnosis, "heterogeneity"].to_numpy(dtype=float)
            groups.append(values)
            colors.append(color)
        drawn = axis.boxplot(groups, tick_labels=["Control", "AD"], patch_artist=True, widths=0.55, showfliers=False)
        for patch, color in zip(drawn["boxes"], colors, strict=True):
            patch.set_facecolor("white")
            patch.set_edgecolor(color)
            patch.set_linewidth(1.4)
        for element in ("whiskers", "caps", "medians"):
            for line in drawn[element]:
                line.set_color("#222222")
        stats = []
        for values in groups:
            summary = quantiles(values)
            stats.append(f"median {summary['median']:.3f} (IQR {summary['q1']:.3f}–{summary['q3']:.3f})")
        letter = "A   " if dataset == "ROSMAP" else ""
        axis.set_title(f"{letter}{DATASET_LABEL[dataset]}", loc="left", fontweight="bold", fontsize=11)
        axis.text(
            0.03,
            0.97,
            f"Control {stats[0]}\nAD {stats[1]}",
            transform=axis.transAxes,
            va="top",
            ha="left",
            fontsize=8,
        )
        axis.set_ylabel("Dissimilarity, median (1 − Spearman ρ)")
        upper = float(np.nanmax(np.concatenate(groups))) * 1.45
        axis.set_ylim(0, upper)
        axis.spines["top"].set_visible(False)
        axis.spines["right"].set_visible(False)
    axes[2].axis("off")
    axes[2].set_title("B   Not assessable", loc="left", fontweight="bold")
    axes[2].text(
        0.0,
        0.72,
        "Saved PCA scores do not cover the original cohorts.\n"
        "Only PC1 and PC2 of flagged outliers were written.\n"
        "Associations were not recalculated.\n\n"
        "GSE215789 profile heterogeneity was not calculated.\n"
        "The pre-scaling matrix is a read-count matrix,\n"
        "not documented normalized expression.",
        transform=axes[2].transAxes,
        va="top",
        fontsize=8,
    )
    figure.tight_layout()
    figure.savefig(FIG / "figure2_heterogeneity_and_pca_context.png", dpi=300, bbox_inches="tight", facecolor="white")
    figure.savefig(FIG / "figure2_heterogeneity_and_pca_context.svg", bbox_inches="tight", facecolor="white")
    figure.savefig(FIG / "figure2_heterogeneity_and_pca_context.pdf", bbox_inches="tight", facecolor="white")
    plt.close(figure)
    caption = (
        "Figure 2. Within-group Spearman dissimilarity on normalized expression before feature-wise scaling. "
        "Boxes show the median and interquartile range. Points are omitted because both cohorts are large. "
        "Panel B records analyses that failed their input rules."
    )
    (FIG / "figure2_heterogeneity_and_pca_context_caption.txt").write_text(caption + "\n")


def write_reports(context: dict) -> None:
    REP.mkdir(parents=True, exist_ok=True)
    (REP / "EDA_FEASIBILITY_REPORT.md").write_text(feasibility_report(context))
    (REP / "BIOLOGICAL_EDA_RESULTS.md").write_text(results_report(context))
    (REP / "EDA_REPRODUCIBILITY.md").write_text(reproducibility_report(context))
    (REP / "EDA_provenance_and_limitations.md").write_text(provenance_report(context))


def feasibility_report(context: dict) -> str:
    counts = context["counts"]
    lines = [
        "# EDA feasibility report",
        "",
        "Feasibility follows the handoff conditions. An analysis marked completed used the original files named here. An analysis marked NOT ASSESSABLE was not replaced.",
        "",
        "## Input roles",
        "",
        "| Dataset | F, retained feature universe | D, assay-supported observation | E, normalized expression before feature scaling |",
        "| --- | --- | --- | --- |",
        "| ROSMAP | Ranking matrix after MinMax on the full processed cohort, AD/AD+ collapse, AD/Control filter, and all-zero column drop | Not available. `ROSMAP_arraymiRNA.gct` stores continuous normalized values and has no detection-call field | `ROSMAP_brain_norm.csv` values for the retained samples and F columns, before MinMax |",
        "| GSE120584 | Same ranking sequence, without the unused 300-AD downsample | Not available. The series matrix table is one intensity column per sample and has no detection flag | `GSE120584_serum_norm.csv` values for the retained samples and F columns, before MinMax |",
        "| GSE215789 | Same ranking sequence | `GSE215789_plasma_count.csv` integer counts for the retained samples and F columns. Count > 0 is read observation | Not available. The pre-scaling file is a count matrix |",
        "",
        "## Cohort check",
        "",
        f"- ROSMAP: {counts['ROSMAP']['n_ad']} AD, {counts['ROSMAP']['n_control']} Control, {counts['ROSMAP']['n_features']} features.",
        f"- GSE120584: {counts['GSE120584']['n_ad']} AD, {counts['GSE120584']['n_control']} Control, {counts['GSE120584']['n_features']} features.",
        f"- GSE215789: {counts['GSE215789']['n_ad']} AD, {counts['GSE215789']['n_control']} Control, {counts['GSE215789']['n_features']} features.",
        "",
        "These match the expected original ranking inputs. The unused GSE120584 downsample to 300 AD samples was not applied.",
        "",
        "## Decisions",
        "",
        "| Analysis | Status | Reason |",
        "| --- | --- | --- |",
        "| A1 mature identity and coverage | Completed | Feature identifiers and the one-to-one project alias map are available. Names absent from that map stay unresolved. |",
        "| A2 NanoString detection | NOT ASSESSABLE | No detection call or documented background rule is stored with the original matrix. |",
        "| A2 microarray detection | NOT ASSESSABLE | No detection flag or documented intensity threshold is stored. |",
        "| A2 sequencing read observation | Completed | Retained plasma features have finite, non-negative integer counts. Prevalence is count > 0. |",
        "| A3 metadata summaries | Completed for linked variables | Sample ID joins were one-to-one. Empty batch fields were left unused. Assay plate is populated and is crosstabulated. |",
        "| A3 PCA associations | NOT ASSESSABLE | Full PC scores were not saved. PCA was not rerun. |",
        "| A4 ROSMAP and GSE120584 | Completed | Pre-MinMax normalized values exist for every retained sample and F feature. |",
        "| A4 GSE215789 | NOT ASSESSABLE | The pre-scaling matrix contains counts, not documented normalized expression. |",
        "",
    ]
    return "\n".join(lines)


def _meta_value(frame: pd.DataFrame, diagnosis: str, variable: str, statistic: str, level: str = "") -> pd.Series:
    match = frame[
        (frame["dataset"] == "ROSMAP")
        & (frame["diagnosis"] == diagnosis)
        & (frame["variable"] == variable)
        & (frame["statistic"] == statistic)
        & (frame["level"].astype(str) == level)
    ]
    if len(match) != 1:
        raise RuntimeError(f"Expected one ROSMAP metadata row for {diagnosis} {variable} {statistic} {level}")
    return match.iloc[0]


def metadata_sentence(frame: pd.DataFrame) -> str:
    ad_age = _meta_value(frame, "AD", "age_at_death_years", "median")
    control_age = _meta_value(frame, "Control", "age_at_death_years", "median")
    ad_rin = _meta_value(frame, "AD", "RIN", "median")
    control_rin = _meta_value(frame, "Control", "RIN", "median")
    ad_plus = _meta_value(frame, "AD", "source_diagnosis", "count", "AD+")
    ad_only = _meta_value(frame, "AD", "source_diagnosis", "count", "AD")
    ad_censored = _meta_value(frame, "AD", "age_censor", "count", "90+")
    control_censored = _meta_value(frame, "Control", "age_censor", "count", "90+")
    return (
        "ROSMAP ranking labels collapse every processed diagnosis containing “AD”. "
        f"In this cohort that combines {int(ad_only['value'])} AD and {int(ad_plus['value'])} AD+ labels. "
        "Age is age at death. The source stores ages at or above 90 years as `90+` "
        f"({int(ad_censored['value'])} AD, {int(control_censored['value'])} Control). "
        "Those values stay censored and are excluded from the age quantiles. "
        f"Among numeric ages, the AD median is {fmt(ad_age['value'], 1)} years "
        f"({int(ad_age['n_nonmissing'])} samples) and the Control median is {fmt(control_age['value'], 1)} years "
        f"({int(control_age['n_nonmissing'])} samples). "
        f"RIN is present for all ranking individuals (AD median {fmt(ad_rin['value'], 1)}, "
        f"Control median {fmt(control_rin['value'], 1)}). "
        "Assay plate is a populated technical identifier with eight levels; the diagnosis counts are in "
        "`tables/batch_diagnosis_counts.tsv`. RNA batch, library batch, NanoString batch, and binding density "
        "are missing for every ranking sample. PMI is missing for one AD sample and one Control sample."
    )


def results_report(context: dict) -> str:
    identity = context["identity_summary"].set_index("dataset")
    coverage = context["coverage"]
    hetero = context["hetero_summary"].set_index(["dataset", "diagnosis"])
    cand = context["candidates"]
    breadth = context["breadth_summary"]
    lines = [
        "# Biological EDA results",
        "",
        "These descriptions apply to the original AD/Control ranking cohorts. They do not change feature selection or candidate ranking.",
        "",
        "## A1. Mature-miRNA identity and coverage",
        "",
        "A feature is a resolved human mature miRNA only when its stored name is a unique product name in `mirBase_to_mirTARBase.json`. The map has one accession for every product name. Multi-name identifiers were not split. Names without an accession were not assigned an arm by removing or adding a 3p/5p suffix.",
        "",
    ]
    for dataset in DATASET_ORDER:
        row = identity.loc[dataset]
        lines.append(
            f"- {dataset}: {int(row['resolved_mature'])} resolved mature identifiers, "
            f"{int(row['ambiguous'])} ambiguous, {int(row['unresolved'])} unresolved, "
            f"{int(row['non_mirna_control'])} non-human features, from {int(row['n_features'])} retained features."
        )
    plasma = identity.loc["GSE215789"]
    lines += [
        "",
        f"ROSMAP and GSE120584 have the same resolved-name counts in the assay matrix and in the retained feature set. "
        f"GSE215789 has {int(plasma['resolved_in_original_assay_matrix'])} resolved names in the count matrix and "
        f"{int(plasma['resolved_mature'])} in the retained feature set. The difference is the original all-zero column filter, not a second identifier map.",
        "",
        "",
        f"Pairwise overlap of resolved names: ROSMAP ∩ GSE120584 = {coverage['rosmap_gse120584_overlap']} (Jaccard {fmt(coverage['rosmap_gse120584_jaccard'])}); "
        f"ROSMAP ∩ GSE215789 = {coverage['rosmap_gse215789_overlap']} (Jaccard {fmt(coverage['rosmap_gse215789_jaccard'])}); "
        f"GSE120584 ∩ GSE215789 = {coverage['gse120584_gse215789_overlap']} (Jaccard {fmt(coverage['gse120584_gse215789_jaccard'])}).",
        f"Three-way overlap: {coverage['three_way']} resolved names.",
        "Exclusive intersections, counted once: "
        + "; ".join(f"{row.membership} = {int(row.n_resolved_mature)}" for row in context["exclusive"].itertuples()),
        "",
        "",
        "The low overlap with ROSMAP is an identifier-resolution result. Most ROSMAP probe names have no arm and no accession in the project map, so they stay out of the intersection. That is not evidence that the corresponding miRNAs were biologically absent from DLPFC.",
        "",
        "Candidate representation in the retained matrices:",
        "",
    ]
    for candidate in CANDIDATES:
        bits = []
        for dataset in DATASET_ORDER:
            row = cand[(cand["candidate"] == candidate) & (cand["dataset"] == dataset)].iloc[0]
            if int(row["in_retained_matrix"]) == 1:
                state = "retained"
            elif int(row["in_original_assay_matrix"]) == 1:
                state = "in the assay matrix, absent after the ranking filter"
            else:
                state = "absent from the assay matrix"
            bits.append(f"{dataset} {state}")
        lines.append(f"- {candidate}: " + "; ".join(bits) + ".")
    lines += [
        "",
        "## A2. Detection and read observation",
        "",
        "ROSMAP and GSE120584 detection prevalence are NOT ASSESSABLE. The original files do not contain NanoString detection calls, negative-control background rules, or microarray detection flags. Normalized intensities and scaled zeros were not converted into detection calls.",
        "",
        "GSE215789 uses original mature-feature counts on the retained samples. A feature is read-observed in a sample when its count is greater than 0. This is not an analytical limit of detection.",
        "",
    ]
    for diagnosis in ("AD", "Control"):
        row = breadth[breadth["diagnosis"] == diagnosis].iloc[0]
        lines.append(
            f"- GSE215789 {diagnosis} sample-level observed-feature fraction: median {fmt(row['median'])} "
            f"(IQR {fmt(row['q1'])}–{fmt(row['q3'])}), n = {int(row['n'])}."
        )
    lines += [
        "",
        "Candidate read-observation counts in the retained GSE215789 matrix:",
        "",
    ]
    observed = context["candidate_prevalence"]
    for candidate in CANDIDATES:
        bits = []
        for diagnosis in ("AD", "Control"):
            row = observed[(observed["feature_id"] == candidate) & (observed["diagnosis"] == diagnosis)].iloc[0]
            bits.append(f"{diagnosis} {int(row['n_observed'])}/{int(row['n_samples'])}")
        lines.append(f"- {candidate}: " + ", ".join(bits) + ".")
    lines += [
        "",
        "Feature-level distributions are the GSE215789 panel of Figure 1.",
        "",
        "## A3. Metadata",
        "",
        "Linked variables are summarized by the ranking diagnosis in `tables/metadata_group_summary.tsv`.",
        "",
        metadata_sentence(context["metadata_summary"]),
        "",
        "GSE120584 contributes age, sex, and APOE4 allele count from the processed serum table. GSE215789 contributes age and sex from the processed plasma table. No hemolysis, platelet, cell-composition, or circulating-processing field is present for the serum or plasma cohorts.",
        "",
        "Associations with PCA scores are NOT ASSESSABLE. `results/eda/pca_summary.csv` stores variance percentages, and `outlier_summary.csv` stores PC1 and PC2 only for flagged samples. The full score matrices were not saved, and PCA was not rerun.",
        "",
        "## A4. Within-group profile heterogeneity",
        "",
        "For each sample, expression was ranked across eligible features with average ranks for ties. Dissimilarity to another sample in the same diagnosis is 1 minus Spearman ρ. The sample score is the median of those dissimilarities, using linear quantile interpolation. No significance test was applied.",
        "",
    ]
    for dataset in ("ROSMAP", "GSE120584"):
        excluded = context["excluded"][dataset]
        lines.append(
            f"- {dataset}: {excluded['eligible']} eligible features; {excluded['non_finite']} excluded for a non-finite value; "
            f"{excluded['zero_variability']} excluded for no variation across retained samples."
        )
        for diagnosis in ("AD", "Control"):
            row = hetero.loc[(dataset, diagnosis)]
            lines.append(
                f"  - {diagnosis}: median dissimilarity {fmt(row['median'])} "
                f"(IQR {fmt(row['q1'])}–{fmt(row['q3'])}), n = {int(row['n'])}."
            )
    lines += [
        "",
        "GSE215789 heterogeneity is NOT ASSESSABLE because the available pre-scaling matrix is a count matrix.",
        "",
        "These scores describe similarity of normalized profiles inside this study. They do not establish intrinsic biological disease subtypes, and they are not comparable as expression magnitudes across NanoString, microarray, and sequencing.",
        "",
        "## Figures",
        "",
        "- `figures/figure1_coverage_and_observation.png` shows exclusive resolved-name intersections and the sequencing read-observation distribution.",
        "- `figures/figure2_heterogeneity_and_pca_context.png` shows within-group dissimilarity for the two normalized cohorts and states the unavailable PCA and sequencing-profile analyses.",
        "",
    ]
    return "\n".join(lines) + "\n"


def reproducibility_report(context: dict) -> str:
    versions = context["versions"]
    lines = [
        "# EDA reproducibility",
        "",
        "From the repository root:",
        "",
        "```bash",
        "python results/biological_eda/scripts/run_biological_eda.py",
        "```",
        "",
        "The script reads original files and writes only under `results/biological_eda/`.",
        "",
        "## Software",
        "",
        f"- Python {versions['python']}",
        f"- pandas {versions['pandas']}",
        f"- numpy {versions['numpy']}",
        f"- scipy {versions['scipy']}",
        f"- matplotlib {versions['matplotlib']}",
        f"- Platform: {versions['platform']}",
        "",
        "These are the versions used for this EDA. They are not claimed to be the original Notebook 11 environment.",
        "",
        "## Parameters",
        "",
        "- Ranking cohorts are produced by `load_ranking_datasets` in `src/prioritization/matrices.py`.",
        "- Quantiles use NumPy `method='linear'`.",
        "- Spearman ranks use SciPy `rankdata(..., method='average')`.",
        "- A sequencing feature is read-observed when count > 0.",
        "- A mature identifier is resolved when the stored name has exactly one accession in `mirBase_to_mirTARBase.json`.",
        "- Heterogeneity requires at least 30 eligible features and at least 3 samples in the diagnosis group.",
        "- Eligible features have a finite value in every retained sample and more than one distinct value.",
        "",
        "## Inputs",
        "",
    ]
    for row in context["manifest"]:
        lines.append(f"- `{row['path']}` — {row['role']} — sha256 `{row['sha256']}`")
    lines += [
        "",
        "## Outputs",
        "",
        "- `reports/EDA_FEASIBILITY_REPORT.md`",
        "- `reports/BIOLOGICAL_EDA_RESULTS.md`",
        "- `reports/EDA_REPRODUCIBILITY.md`",
        "- `reports/EDA_provenance_and_limitations.md`",
        "- `tables/biological_eda_summary.tsv`",
        "- `tables/` and `supporting_data/` TSV files listed in the provenance report",
        "- `figures/figure1_coverage_and_observation.png` and PDF/SVG",
        "- `figures/figure2_heterogeneity_and_pca_context.png` and PDF/SVG",
        "",
        "The visualization atlas has bar, box, scatter, heatmap, PCA, and ROC renderers. It has no UpSet or ECDF renderer. Figure 1 is therefore drawn with Matplotlib, using the atlas Okabe–Ito colours and 300 dpi PNG, SVG, and PDF export. Figure 2 boxes follow the atlas box convention: white boxes, coloured edges, and no significance markers.",
        "",
    ]
    return "\n".join(lines)


def provenance_report(context: dict) -> str:
    return "\n".join(
        [
            "# Provenance and limitations",
            "",
            "## How F, D, and E were separated",
            "",
            "F is the column set of the matrix returned by `prepare_*_for_ranking`. MinMaxScaler is fit on the full processed cohort, including diagnoses other than AD and Control. Samples are then restricted to AD and Control. ROSMAP maps every diagnosis string containing AD, including AD+, to AD before that restriction. Columns that are entirely zero after scaling are dropped. The GSE120584 function that downsamples AD to 300 rows is not called.",
            "",
            "The values inside F are feature-wise scaled. They are not used for detection prevalence or profile heterogeneity.",
            "",
            "E, where used, is the same sample index and F columns taken from the processed CSV before that scaling. Sample identity is the `Sample ID` index. Diagnosis labels were compared with the ranking labels after the ROSMAP collapse and were required to agree.",
            "",
            "D for GSE215789 is that same pre-scaling count matrix. The preparation script reads a miRNA count table, maps sample titles to GEO accessions, and keeps plasma samples. This EDA then keeps the ranking-cohort rows and F columns. Counts were checked to be finite, non-negative, and integer-valued.",
            "",
            "## Files inspected but not used as detection inputs",
            "",
            f"- `{GCT}` is a GCT of continuous normalized NanoString values, 309 features by the source header. It has no detection-call column.",
            f"- `{ORIGINAL / 'downloads/microRNA/GSE120584_series_matrix.txt.gz'}` has a sample-by-intensity table and no detection-flag field.",
            "- `results/eda/pca_summary.csv` and `results/eda/outlier_summary.csv` do not contain PC scores for the full cohorts.",
            "",
            "## Limitations",
            "",
            "- The alias map does not record a miRBase release. Resolved identity means one stored product name per accession in that file.",
            "- ROSMAP probe names that omit the arm remain unresolved. Their absence from an intersection is an annotation limit.",
            "- Read observation is not analytical sensitivity, specificity, or a call against background.",
            "- Metadata summaries are descriptive. They are not disease-association models and they are not batch correction.",
            "- Profile dissimilarity depends on the original normalization and on the features retained before ranking. It is not a test of biological heterogeneity.",
            "- No original sample was removed beyond the diagnosis definition already used to build the ranking matrices. No ranking file was edited.",
            "",
        ]
    )


def main() -> None:
    for folder in (FIG, TAB, SUP, REP):
        folder.mkdir(parents=True, exist_ok=True)
    required = [ALIAS_MAP, CLINICAL, BIOSPECIMEN, ASSAY_META, GCT, *PROCESSED.values()]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise SystemExit("Missing required inputs:\n" + "\n".join(missing))

    inverse = load_alias_map(ALIAS_MAP)
    ranked = load_ranking_datasets(
        {
            "rosmap": str(PROCESSED["ROSMAP"]),
            "gse120584": str(PROCESSED["GSE120584"]),
            "gse215789": str(PROCESSED["GSE215789"]),
        }
    )
    expected = {"ROSMAP": (209, 169, 309), "GSE120584": (1021, 288, 2521), "GSE215789": (23, 21, 1137)}
    counts = {}
    processed = {}
    expression = {}
    labels = {}
    identity_rows = []
    resolved_sets = {}
    assay_sets = {}
    for dataset in DATASET_ORDER:
        frame = read_processed(dataset)
        processed[dataset] = frame
        matrix = ranked[dataset]
        features = [column for column in matrix.columns if column != "Diagnosis"]
        n_ad = int((matrix["Diagnosis"] == "AD").sum())
        n_control = int((matrix["Diagnosis"] == "Control").sum())
        if (n_ad, n_control, len(features)) != expected[dataset]:
            raise RuntimeError(f"{dataset} dimensions {(n_ad, n_control, len(features))} differ from the original cohort")
        counts[dataset] = {"n_ad": n_ad, "n_control": n_control, "n_features": len(features), "n_samples": int(len(matrix))}
        raw_labels = ranking_labels(frame["Diagnosis"], collapse_ad=(dataset == "ROSMAP"))
        keep = raw_labels.isin(["AD", "Control"])
        expr = frame.loc[keep, features].apply(pd.to_numeric, errors="coerce")
        expr = expr.loc[matrix.index]
        lab = raw_labels.loc[matrix.index]
        assert_alignment(expr, lab, matrix, dataset)
        expression[dataset] = expr
        labels[dataset] = lab
        assay_names = [column for column in frame.columns if column not in META_DROP[dataset]]
        assay_resolved = set()
        for name in assay_names:
            feature_class, resolved_name, _reason = classify_feature(name, inverse)
            if feature_class == "resolved mature":
                assay_resolved.add(resolved_name)
        assay_sets[dataset] = assay_resolved
        resolved = set()
        for name in features:
            feature_class, resolved_name, reason = classify_feature(name, inverse)
            accession = inverse.get(name, [""])[0] if feature_class == "resolved mature" else ""
            identity_rows.append(
                {
                    "dataset": dataset,
                    "feature_id": name,
                    "class": feature_class,
                    "resolved_mature_id": resolved_name,
                    "accession": accession,
                    "reason": reason,
                    "in_retained_matrix": 1,
                    "in_original_assay_matrix": 1,
                }
            )
            if feature_class == "resolved mature":
                resolved.add(resolved_name)
        resolved_sets[dataset] = resolved

    identity = pd.DataFrame(identity_rows)
    identity.to_csv(SUP / "feature_identity_audit.tsv", sep="\t", index=False)
    class_counts = (
        identity.groupby(["dataset", "class"]).size().unstack(fill_value=0).reindex(DATASET_ORDER).reset_index()
    )
    for column in ("resolved mature", "ambiguous", "unresolved", "non-miRNA/control"):
        if column not in class_counts.columns:
            class_counts[column] = 0
    identity_summary = pd.DataFrame(
        {
            "dataset": class_counts["dataset"],
            "n_features": [counts[dataset]["n_features"] for dataset in class_counts["dataset"]],
            "resolved_mature": class_counts["resolved mature"].astype(int),
            "ambiguous": class_counts["ambiguous"].astype(int),
            "unresolved": class_counts["unresolved"].astype(int),
            "non_mirna_control": class_counts["non-miRNA/control"].astype(int),
            "resolved_in_original_assay_matrix": [len(assay_sets[dataset]) for dataset in class_counts["dataset"]],
        }
    )
    identity_summary.to_csv(TAB / "mature_identity_summary.tsv", sep="\t", index=False)

    pairs = (
        ("ROSMAP", "GSE120584"),
        ("ROSMAP", "GSE215789"),
        ("GSE120584", "GSE215789"),
    )
    coverage_rows = exclusive_counts(resolved_sets)
    coverage = {
        "three_way": len(resolved_sets["ROSMAP"] & resolved_sets["GSE120584"] & resolved_sets["GSE215789"]),
    }
    for left, right in pairs:
        key = f"{left.lower()}_{right.lower()}"
        overlap = len(resolved_sets[left] & resolved_sets[right])
        coverage[f"{key}_overlap"] = overlap
        coverage[f"{key}_jaccard"] = jaccard(resolved_sets[left], resolved_sets[right])
        coverage_rows.append(
            {
                "ROSMAP": "",
                "GSE120584": "",
                "GSE215789": "",
                "n_resolved_mature": overlap,
                "membership": f"pairwise {left} ∩ {right}",
                "jaccard": coverage[f"{key}_jaccard"],
            }
        )
    coverage_rows.append(
        {
            "ROSMAP": 1,
            "GSE120584": 1,
            "GSE215789": 1,
            "n_resolved_mature": coverage["three_way"],
            "membership": "three-way intersection, not exclusive",
            "jaccard": "",
        }
    )
    coverage_table = pd.DataFrame(coverage_rows)
    coverage_table.to_csv(TAB / "coverage_intersections.tsv", sep="\t", index=False)

    candidate_rows = []
    for dataset in DATASET_ORDER:
        assay_columns = set(processed[dataset].columns)
        retained = set(expression[dataset].columns)
        for candidate in CANDIDATES:
            feature_class, _resolved, reason = classify_feature(candidate, inverse)
            candidate_rows.append(
                {
                    "dataset": dataset,
                    "candidate": candidate,
                    "class_if_present": feature_class,
                    "accession": inverse.get(candidate, [""])[0],
                    "in_original_assay_matrix": int(candidate in assay_columns),
                    "in_retained_matrix": int(candidate in retained),
                    "reason": reason,
                }
            )
    candidates = pd.DataFrame(candidate_rows)
    candidates.to_csv(TAB / "candidate_representation.tsv", sep="\t", index=False)

    # A2 sequencing counts
    counts_matrix = expression["GSE215789"]
    values = counts_matrix.to_numpy(dtype=float)
    integer_counts = np.isfinite(values).all() and np.all(values >= 0) and np.allclose(values, np.rint(values))
    if not integer_counts:
        raise RuntimeError("GSE215789 retained values are not non-negative integer counts")
    observed = values > 0
    prevalence_rows = []
    for feature_index, feature in enumerate(counts_matrix.columns):
        for diagnosis in ("AD", "Control"):
            mask = (labels["GSE215789"] == diagnosis).to_numpy()
            prevalence_rows.append(
                {
                    "dataset": "GSE215789",
                    "feature_id": feature,
                    "diagnosis": diagnosis,
                    "n_samples": int(mask.sum()),
                    "n_observed": int(observed[mask, feature_index].sum()),
                    "read_observation_prevalence": float(observed[mask, feature_index].mean()),
                    "definition": "count > 0; not an analytical detection call",
                }
            )
    prevalence = pd.DataFrame(prevalence_rows)
    prevalence.to_csv(SUP / "feature_observation_prevalence.tsv", sep="\t", index=False)
    unavailable = pd.DataFrame(
        [
            {
                "dataset": "ROSMAP",
                "status": "NOT_ASSESSABLE",
                "reason": "No NanoString detection call or documented background rule in the original GCT or assay metadata.",
            },
            {
                "dataset": "GSE120584",
                "status": "NOT_ASSESSABLE",
                "reason": "No microarray detection flag or documented threshold in the series matrix used originally.",
            },
        ]
    )
    unavailable.to_csv(SUP / "detection_not_assessable.tsv", sep="\t", index=False)

    breadth_rows = []
    for sample_index, sample_id in enumerate(counts_matrix.index):
        breadth_rows.append(
            {
                "dataset": "GSE215789",
                "sample_id": sample_id,
                "diagnosis": labels["GSE215789"].iloc[sample_index],
                "n_features": int(observed.shape[1]),
                "n_observed_features": int(observed[sample_index].sum()),
                "observed_feature_fraction": float(observed[sample_index].mean()),
            }
        )
    breadth = pd.DataFrame(breadth_rows)
    breadth.to_csv(SUP / "sample_observation_breadth.tsv", sep="\t", index=False)
    breadth_summary_rows = []
    for diagnosis in ("AD", "Control"):
        stats = quantiles(breadth.loc[breadth["diagnosis"] == diagnosis, "observed_feature_fraction"])
        stats["diagnosis"] = diagnosis
        breadth_summary_rows.append(stats)
    breadth_summary = pd.DataFrame(breadth_summary_rows)
    breadth_summary.to_csv(TAB / "sample_observation_breadth_summary.tsv", sep="\t", index=False)
    candidate_prevalence = prevalence[prevalence["feature_id"].isin(CANDIDATES)].copy()
    candidate_prevalence.to_csv(TAB / "candidate_read_observation.tsv", sep="\t", index=False)

    # A3 metadata
    dictionary = [
        {
            "variable": "diagnosis",
            "dataset": "ROSMAP",
            "source": str(PROCESSED["ROSMAP"]),
            "join_key": "Sample ID",
            "status": "linked",
            "note": "Ranking label collapses processed labels containing AD, including AD+, to AD.",
        },
        {
            "variable": "sex",
            "dataset": "ROSMAP",
            "source": str(PROCESSED["ROSMAP"]),
            "join_key": "Sample ID",
            "status": "linked",
            "note": "Processed Gender column.",
        },
        {
            "variable": "race_code",
            "dataset": "ROSMAP",
            "source": str(PROCESSED["ROSMAP"]),
            "join_key": "Sample ID",
            "status": "linked",
            "note": "Integer code as stored. Not recoded to a population label.",
        },
        {
            "variable": "pmi_hours",
            "dataset": "ROSMAP",
            "source": str(PROCESSED["ROSMAP"]),
            "join_key": "Sample ID",
            "status": "linked",
            "note": "Post-mortem interval from the processed clinical columns.",
        },
        {
            "variable": "braak_stage",
            "dataset": "ROSMAP",
            "source": str(PROCESSED["ROSMAP"]),
            "join_key": "Sample ID",
            "status": "linked",
            "note": "Braak score as stored.",
        },
    ]
    rosmap_meta = processed["ROSMAP"].loc[expression["ROSMAP"].index, ["Gender", "Race", "PMI", "Braak", "Diagnosis"]].copy()
    rosmap_meta = rosmap_meta.rename(columns={"Gender": "sex", "Race": "race_code", "PMI": "pmi_hours", "Braak": "braak_stage", "Diagnosis": "source_diagnosis"})
    extra, extra_notes = link_rosmap_extra(expression["ROSMAP"].index)
    rosmap_meta = rosmap_meta.join(extra)
    dictionary.extend(extra_notes)
    serum = processed["GSE120584"].loc[expression["GSE120584"].index, ["Age", "Gender", "APOE4", "Diagnosis"]].copy()
    serum = serum.rename(columns={"Age": "age_years", "Gender": "sex", "APOE4": "apoe4_allele_count", "Diagnosis": "source_diagnosis"})
    plasma = processed["GSE215789"].loc[expression["GSE215789"].index, ["Age", "Gender", "Diagnosis"]].copy()
    plasma = plasma.rename(columns={"Age": "age_years", "Gender": "sex", "Diagnosis": "source_diagnosis"})
    for dataset, note in (
        ("GSE120584", "Age, sex, and APOE4 allele count from the processed serum table."),
        ("GSE215789", "Age and sex from the processed plasma table. Tissue was already restricted to plasma."),
    ):
        dictionary.append(
            {
                "variable": "age_years, sex" + (", apoe4_allele_count" if dataset == "GSE120584" else ""),
                "dataset": dataset,
                "source": str(PROCESSED[dataset]),
                "join_key": "Sample ID",
                "status": "linked",
                "note": note,
            }
        )
    dictionary.append(
        {
            "variable": "hemolysis_platelet_batch",
            "dataset": "GSE120584 and GSE215789",
            "source": str(PROCESSED["GSE120584"]),
            "join_key": "Sample ID",
            "status": "NOT_ASSESSABLE",
            "note": "No hemolysis, platelet, cell-composition, or processing-batch field is stored on the original sample tables.",
        }
    )
    dictionary.append(
        {
            "variable": "pca_scores",
            "dataset": "all",
            "source": str(ROOT / "results/eda/pca_summary.csv"),
            "join_key": "not applicable",
            "status": "NOT_ASSESSABLE",
            "note": "Full PC scores were not saved. Associations were not recalculated.",
        }
    )
    pd.DataFrame(dictionary).to_csv(TAB / "metadata_dictionary.tsv", sep="\t", index=False)

    age_code = rosmap_meta["age_at_death_years"].astype(str)
    rosmap_meta["age_censor"] = np.where(age_code.eq("90+"), "90+", "numeric age")
    rosmap_meta["age_at_death_years"] = pd.to_numeric(rosmap_meta["age_at_death_years"], errors="coerce")
    rosmap_kinds = {
        "sex": "categorical",
        "race_code": "categorical",
        "pmi_hours": "continuous",
        "braak_stage": "continuous",
        "source_diagnosis": "categorical",
        "age_at_death_years": "continuous",
        "age_censor": "categorical",
        "apoe_genotype": "categorical",
    }
    if "RIN" in rosmap_meta.columns:
        rosmap_kinds["RIN"] = "continuous"
    summary_rows = group_rows("ROSMAP", labels["ROSMAP"], rosmap_meta, rosmap_kinds)
    summary_rows += group_rows("GSE120584", labels["GSE120584"], serum, {"age_years": "continuous", "sex": "categorical", "apoe4_allele_count": "categorical"})
    summary_rows += group_rows("GSE215789", labels["GSE215789"], plasma, {"age_years": "continuous", "sex": "categorical"})
    pd.DataFrame(summary_rows).to_csv(TAB / "metadata_group_summary.tsv", sep="\t", index=False)
    linked_samples = rosmap_meta.copy()
    linked_samples.insert(0, "diagnosis", labels["ROSMAP"].to_numpy())
    linked_samples.insert(0, "sample_id", rosmap_meta.index)
    linked_samples.to_csv(SUP / "rosmap_linked_metadata.tsv", sep="\t", index=False)
    batch_rows = [
        {
            "dataset": "ROSMAP",
            "batch_variable": "rna_batch, library_batch, nanostring_batch, binding_density",
            "diagnosis": "",
            "level": "",
            "n": "",
            "status": "NOT_ASSESSABLE",
            "note": "These assay fields are missing for every ranking individual.",
        }
    ]
    if "plate" in rosmap_meta.columns:
        plate_counts = (
            pd.DataFrame({"diagnosis": labels["ROSMAP"], "plate": rosmap_meta["plate"].astype(str)})
            .groupby(["plate", "diagnosis"], observed=True)
            .size()
            .reset_index(name="n")
        )
        for record in plate_counts.to_dict(orient="records"):
            batch_rows.append(
                {
                    "dataset": "ROSMAP",
                    "batch_variable": "plate",
                    "diagnosis": record["diagnosis"],
                    "level": record["plate"],
                    "n": int(record["n"]),
                    "status": "linked",
                    "note": "One assay plate per ranking individual.",
                }
            )
    batch_rows.extend(
        [
            {
                "dataset": "GSE120584",
                "batch_variable": "none stored",
                "diagnosis": "",
                "level": "",
                "n": "",
                "status": "NOT_ASSESSABLE",
                "note": "No batch identifier is present on the original serum table.",
            },
            {
                "dataset": "GSE215789",
                "batch_variable": "none stored",
                "diagnosis": "",
                "level": "",
                "n": "",
                "status": "NOT_ASSESSABLE",
                "note": "No batch identifier is present on the original plasma table.",
            },
        ]
    )
    pd.DataFrame(batch_rows).to_csv(TAB / "batch_diagnosis_counts.tsv", sep="\t", index=False)
    pd.DataFrame(
        [
            {
                "dataset": dataset,
                "status": "NOT_ASSESSABLE",
                "reason": "Saved PCA artifacts do not contain scores for every original sample. PCA was not rerun.",
            }
            for dataset in DATASET_ORDER
        ]
    ).to_csv(TAB / "pc_metadata_descriptive_associations.tsv", sep="\t", index=False)

    # A4
    hetero_frames = []
    excluded = {}
    hetero_summary_rows = []
    for dataset in ("ROSMAP", "GSE120584"):
        scores, exclusion = heterogeneity_scores(expression[dataset])
        excluded[dataset] = exclusion
        if scores.empty or exclusion["eligible"] < 30:
            raise RuntimeError(f"{dataset} heterogeneity became not assessable after the feature rule")
        scores.insert(0, "dataset", dataset)
        scores.insert(2, "diagnosis", labels[dataset].loc[scores["sample_id"]].to_numpy())
        for diagnosis, group_size in scores.groupby("diagnosis").size().items():
            if int(group_size) < 3:
                raise RuntimeError(f"{dataset} {diagnosis} has fewer than 3 samples")
        hetero_frames.append(scores)
        for diagnosis in ("AD", "Control"):
            stats = quantiles(scores.loc[scores["diagnosis"] == diagnosis, "heterogeneity"])
            hetero_summary_rows.append(
                {
                    "dataset": dataset,
                    "diagnosis": diagnosis,
                    "n": stats["n"],
                    "median": stats["median"],
                    "q1": stats["q1"],
                    "q3": stats["q3"],
                    "n_eligible_features": exclusion["eligible"],
                    "n_excluded_non_finite": exclusion["non_finite"],
                    "n_excluded_zero_variability": exclusion["zero_variability"],
                }
            )
    excluded["GSE215789"] = {
        "eligible": 0,
        "non_finite": 0,
        "zero_variability": 0,
        "input_features": counts["GSE215789"]["n_features"],
        "status": "NOT_ASSESSABLE",
        "reason": "Pre-scaling matrix is a count matrix, not documented normalized expression.",
    }
    hetero = pd.concat(hetero_frames, ignore_index=True)
    hetero.to_csv(SUP / "sample_profile_heterogeneity.tsv", sep="\t", index=False)
    hetero_summary = pd.DataFrame(hetero_summary_rows)
    hetero_summary.to_csv(TAB / "profile_heterogeneity_group_summary.tsv", sep="\t", index=False)
    pd.DataFrame([{"dataset": "GSE215789", **excluded["GSE215789"]}]).to_csv(
        SUP / "gse215789_heterogeneity_not_assessable.tsv", sep="\t", index=False
    )

    summary = []
    for dataset in DATASET_ORDER:
        row = {
            "dataset": dataset,
            "n_ad": counts[dataset]["n_ad"],
            "n_control": counts[dataset]["n_control"],
            "n_features": counts[dataset]["n_features"],
            "n_resolved_mature": int(identity_summary.loc[identity_summary["dataset"] == dataset, "resolved_mature"].iloc[0]),
            "candidates_retained": ",".join(
                candidate
                for candidate in CANDIDATES
                if int(candidates[(candidates["dataset"] == dataset) & (candidates["candidate"] == candidate)]["in_retained_matrix"].iloc[0]) == 1
            ),
            "detection_or_observation": "read observation, count > 0" if dataset == "GSE215789" else "NOT_ASSESSABLE",
            "profile_heterogeneity": "completed" if dataset in ("ROSMAP", "GSE120584") else "NOT_ASSESSABLE",
        }
        if dataset in ("ROSMAP", "GSE120584"):
            for diagnosis in ("AD", "Control"):
                match = hetero_summary[(hetero_summary["dataset"] == dataset) & (hetero_summary["diagnosis"] == diagnosis)].iloc[0]
                row[f"heterogeneity_median_{diagnosis}"] = match["median"]
                row[f"heterogeneity_q1_{diagnosis}"] = match["q1"]
                row[f"heterogeneity_q3_{diagnosis}"] = match["q3"]
        summary.append(row)
    summary_frame = pd.DataFrame(summary)
    summary_frame.to_csv(TAB / "biological_eda_summary.tsv", sep="\t", index=False)

    exclusive = pd.DataFrame(exclusive_counts(resolved_sets))
    draw_figures(exclusive, resolved_sets, prevalence, hetero, candidate_prevalence)

    manifest = []
    for path, role in (
        (PROCESSED["ROSMAP"], "E and metadata for ROSMAP; F is derived"),
        (PROCESSED["GSE120584"], "E and metadata for GSE120584; F is derived"),
        (PROCESSED["GSE215789"], "D counts and metadata for GSE215789; F is derived"),
        (ALIAS_MAP, "one-to-one mature-product alias map"),
        (CLINICAL, "ROSMAP age at death and APOE genotype"),
        (BIOSPECIMEN, "ROSMAP specimen-to-individual key"),
        (ASSAY_META, "ROSMAP RIN and empty batch fields"),
    ):
        manifest.append({"path": str(path), "role": role, "sha256": sha256(path), "bytes": path.stat().st_size})
    pd.DataFrame(manifest).to_csv(SUP / "input_manifest.tsv", sep="\t", index=False)
    versions = {
        "python": platform.python_version(),
        "pandas": version("pandas"),
        "numpy": version("numpy"),
        "scipy": version("scipy"),
        "matplotlib": version("matplotlib"),
        "platform": platform.platform(),
    }
    (SUP / "environment.json").write_text(json.dumps(versions, indent=2) + "\n")
    (FIG / "figure_audit.json").write_text(
        json.dumps(
            {
                "atlas_renderers_used": [],
                "reason": "The atlas has no UpSet or ECDF renderer. Boxes in Figure 2 were drawn to the atlas box convention without a significance test.",
                "colours": {"Control": CONTROL_COLOR, "AD": AD_COLOR, "intersection": "#009E73"},
                "exports": ["png-300dpi", "svg", "pdf"],
                "pdf_fonttype": 42,
            },
            indent=2,
        )
        + "\n"
    )
    context = {
        "counts": counts,
        "identity_summary": identity_summary,
        "coverage": coverage,
        "exclusive": exclusive,
        "metadata_summary": pd.DataFrame(summary_rows),
        "candidates": candidates,
        "candidate_prevalence": candidate_prevalence,
        "breadth_summary": breadth_summary,
        "hetero_summary": hetero_summary,
        "excluded": excluded,
        "versions": versions,
        "manifest": manifest,
    }
    write_reports(context)
    print(json.dumps({"counts": counts, "coverage": coverage, "resolved": identity_summary.to_dict(orient="records")}, indent=2, default=str))


if __name__ == "__main__":
    main()
