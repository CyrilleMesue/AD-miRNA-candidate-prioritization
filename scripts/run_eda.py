#!/usr/bin/env python
"""Exploratory analysis of the three final AD-versus-Control ranking matrices.

python scripts/run_eda.py --config config/analysis_config.yaml
"""
from __future__ import annotations

import argparse
import json
import platform
import sys
from importlib.metadata import version
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
matplotlib.rcParams["pdf.fonttype"] = 42
matplotlib.rcParams["svg.fonttype"] = "none"

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "third_party"))

import pandas as pd
import yaml

from bioinformatics_visual_evidence_atlas.visualization.figures import render
from bioinformatics_visual_evidence_atlas.visualization.recommend import recommend
from eda.exploratory_analysis import (
    BIOLOGICAL_SOURCE,
    assess_outliers,
    file_record,
    fit_dataset_pca,
    load_matrices,
    sample_summaries,
    summarize_datasets,
)
from eda.plotting import plot_class_counts, plot_expression, plot_pca


def resolve(root: Path, value: str) -> str:
    path = Path(value)
    if not path.is_absolute():
        path = root / path
    return str(path)


def load_config(path: str) -> dict:
    with open(path) as handle:
        return yaml.safe_load(handle)


def recommendations(rankings: list[dict]) -> dict:
    questions = [
        (
            "class_counts",
            "How many AD and Control samples are in each dataset?",
            ["value", "group"],
            "Group or condition comparison",
        ),
        (
            "expression",
            "How are sample-level mean miRNA values distributed across the three datasets?",
            ["value", "group"],
            "Distribution and variability",
        ),
        (
            "zeros",
            "How does the per-sample fraction of exact zeros vary across datasets?",
            ["value", "group"],
            "Data quality and measurement reliability",
        ),
        (
            "pca",
            "What broad multivariate structure is visible after unsupervised PCA?",
            ["feature", "sample", "value"],
            "Dimensionality reduction and sample separation",
        ),
    ]
    return {
        key: recommend(question, columns, category=category, rankings=rankings)
        for key, question, columns, category in questions
    }


def write_thesis_report(path: Path, summary: pd.DataFrame, pca: pd.DataFrame, outliers: pd.DataFrame, inputs: list[dict]) -> None:
    by_name = summary.set_index("dataset")
    pca_by = pca.set_index("dataset")
    dataset_rows = outliers[outliers["row_type"] == "dataset"]
    sample_rows = outliers[outliers["row_type"] == "sample"]

    def line(name: str) -> str:
        row = by_name.loc[name]
        return (
            f"{name} ({row['biological_source']}): {int(row['n_samples'])} samples "
            f"({int(row['n_ad'])} AD, {int(row['n_control'])} Control), "
            f"{int(row['n_mirnas'])} miRNAs, {int(row['n_missing'])} missing values, "
            f"range {row['minimum']:.17g} to {row['maximum']:.17g}, "
            f"exact-zero proportion {row['zero_proportion']:.6f}."
        )

    def pca_line(name: str) -> str:
        row = pca_by.loc[name]
        return (
            f"{name}: PC1 {100 * row['variance_pc1']:.1f}% "
            f"({row['variance_pc1']:.6f}), PC2 {100 * row['variance_pc2']:.1f}% "
            f"({row['variance_pc2']:.6f})."
        )

    flag_lines = []
    for _, row in dataset_rows.iterrows():
        if int(row["n_flagged"]) == 0:
            flag_lines.append(f"{row['dataset']}: no sample exceeded the threshold {row['threshold']:.4f}.")
        else:
            ids = row["flagged_sample_ids"].split(";")
            classes = row["flagged_classes"].split(";")
            listed = ", ".join(f"{sample} ({label})" for sample, label in zip(ids, classes))
            flag_lines.append(
                f"{row['dataset']}: {int(row['n_flagged'])} of {int(row['n_samples'])} samples exceeded "
                f"the threshold {row['threshold']:.4f}: {listed}."
            )
    input_lines = [
        f"- `{record['path']}` ({record['bytes']} bytes, SHA-256 `{record['sha256']}`)"
        for record in inputs
    ]
    text = f"""# Exploratory analysis of AD miRNA datasets

This note records the descriptive analysis of the three matrices supplied to biomarker ranking. It is written so that it can support `\\subsection{{Exploratory analysis of AD miRNA datasets}}`. It does not interpret point clouds as differential expression, diagnostic performance, or biomarker validity.

## Methods

The matrices were read with `prioritization.matrices.load_ranking_datasets`, the same loader used by the ranking pipeline. That loader drops the clinical columns recorded in `src/prioritization/matrices.py`, applies `MinMaxScaler` to the full processed cohort, keeps rows labeled AD or Control, and drops columns that are entirely zero. The unused GSE120584 downsample of 300 AD samples was not applied. No feature was selected with diagnosis. Processed files were not rewritten.

Biological source labels in the summary table are DLPFC for ROSMAP, serum for GSE120584, and plasma for GSE215789. These labels are not columns of the ranking matrices. Serum and plasma are the specimen names already used for the processed files. DLPFC is the source label assigned for this descriptive table.

Class counts were drawn as one grouped bar chart. Counts were printed on the bars. The visualization skill routes a group comparison to `grouped_bars`, which always computes a mean, an SEM, and a Welch test. Those statistics were not computed, because each bar is a census rather than a set of replicates.

Expression was summarized with one number per sample. A density of every matrix entry was not drawn: every matrix is min-max scaled to the unit interval, and the matrices contain 116,802, 3,299,989, and 50,028 entries. Sample medians were calculated and then not used as the plotted summary, because the GSE215789 sample medians are mostly zero. The plotted summaries are the sample mean and the fraction of exact zeros. Both are quartile box plots from the skill renderer `grouped_boxes`, which does not add a significance marker. Axis titles were set to the summary name and to "Dataset" after that renderer drew them. Median lines are black so they are not the same hue as the second dataset color, and flier markers are smaller than the renderer default. No points were subsampled, so no random seed was required.

PCA was fit separately to each matrix by a column-centered singular value decomposition without scaling. This is the same linear algebra as the skill renderer `pca_scatter`. Diagnosis was used only to color points after the scores were obtained. The three matrices were not concatenated, because their miRNA sets differ. Confidence ellipses were not drawn. The skill companion bar of variance explained was not drawn as a separate figure; PC1 and PC2 percentages are the axis labels and are stored in `pca_summary.csv`. GSE120584 was not passed to `pca_scatter` as a long table, because that table would contain one row per sample-feature pair.

Sample-level outliers were assessed in the PC1–PC2 score plane. Each component was centered at its median and divided by 1.4826 times its median absolute deviation. A component with a MAD of zero would have been omitted; none was. The squared robust distance was compared with the chi-square 0.999 quantile on 2 degrees of freedom (threshold {float(dataset_rows['threshold'].iloc[0]):.4f}). Flagged points, when present, are outlined on the PCA plot. No sample was removed. Exclusion is not recommended: the rule identifies tails of the same unsupervised scores, and no technical failure is recorded for any sample.

## Results

### Dataset composition

{line("ROSMAP")}

{line("GSE120584")}

{line("GSE215789")}

The observed maxima are 1 within floating-point rounding of the min-max scaler. Missing values in the numeric matrices are zero.

### PCA variance

{pca_line("ROSMAP")}

{pca_line("GSE120584")}

{pca_line("GSE215789")}

### Outlier flags

{chr(10).join(flag_lines)}

Flagged samples: {0 if sample_rows.empty else len(sample_rows)}. Exclusion recommended: no. Samples removed: none.

## Suggested figure captions

**Class counts.** Number of AD and Control samples in the ROSMAP (DLPFC), GSE120584 (serum), and GSE215789 (plasma) matrices used for miRNA ranking. Counts are printed above the bars. No hypothesis test is shown.

**Expression.** Distribution of sample-level summaries of the min-max scaled ranking matrices. (A) Mean value across miRNAs in each sample. (B) Fraction of miRNAs with an exact zero in each sample. Boxes show quartiles. No significance test is shown. ROSMAP n = {int(by_name.loc['ROSMAP', 'n_samples'])}; GSE120584 n = {int(by_name.loc['GSE120584', 'n_samples'])}; GSE215789 n = {int(by_name.loc['GSE215789', 'n_samples'])}.

**ROSMAP PCA.** Unsupervised column-centered PCA of the ROSMAP ranking matrix ({int(pca_by.loc['ROSMAP', 'n_samples'])} samples, {int(pca_by.loc['ROSMAP', 'n_mirnas'])} miRNAs). Points are colored by diagnosis after the decomposition. PC1 explains {100 * pca_by.loc['ROSMAP', 'variance_pc1']:.1f}% of the variance and PC2 explains {100 * pca_by.loc['ROSMAP', 'variance_pc2']:.1f}%. Open circles, if present, mark samples whose robust distance in this plane exceeds the pre-specified chi-square 0.999 threshold. The coloring is descriptive.

**GSE120584 PCA.** Unsupervised column-centered PCA of the GSE120584 ranking matrix ({int(pca_by.loc['GSE120584', 'n_samples'])} samples, {int(pca_by.loc['GSE120584', 'n_mirnas'])} miRNAs). Points are colored by diagnosis after the decomposition. PC1 explains {100 * pca_by.loc['GSE120584', 'variance_pc1']:.1f}% of the variance and PC2 explains {100 * pca_by.loc['GSE120584', 'variance_pc2']:.1f}%. Open circles, if present, mark samples above the same robust-distance threshold. The coloring is descriptive.

**GSE215789 PCA.** Unsupervised column-centered PCA of the GSE215789 ranking matrix ({int(pca_by.loc['GSE215789', 'n_samples'])} samples, {int(pca_by.loc['GSE215789', 'n_mirnas'])} miRNAs). Points are colored by diagnosis after the decomposition. PC1 explains {100 * pca_by.loc['GSE215789', 'variance_pc1']:.1f}% of the variance and PC2 explains {100 * pca_by.loc['GSE215789', 'variance_pc2']:.1f}%. Open circles, if present, mark samples above the same robust-distance threshold. The coloring is descriptive.

## Input files

{chr(10).join(input_lines)}

## Limits of this note

The min-max scaler was fit on each processed cohort before rows other than AD and Control were removed, so the unit interval reflects that larger cohort. PCA axes have an arbitrary sign. The outlier rule is a descriptive screen of two components, not a formal hypothesis test, and it was not used to edit the matrices. Visual arrangement of colors in the PCA plots is not evidence of differential expression or of classification accuracy.
"""
    path.write_text(text, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Exploratory analysis of the ranking matrices")
    parser.add_argument("--config", default=str(ROOT / "config" / "analysis_config.yaml"))
    args = parser.parse_args()
    root = ROOT
    config = load_config(args.config)
    paths = {key: resolve(root, value) for key, value in config["processed"].items()}
    datasets = load_matrices(paths)
    summary = summarize_datasets(datasets)
    summaries = sample_summaries(datasets)
    pca, fitted = fit_dataset_pca(datasets)
    outliers = assess_outliers(fitted)

    out = root / "results" / "eda"
    figures = out / "figures"
    audits = out / "skill_audits"
    out.mkdir(parents=True, exist_ok=True)
    summary.to_csv(out / "dataset_summary.csv", index=False)
    pca.to_csv(out / "pca_summary.csv", index=False)
    outliers.to_csv(out / "outlier_summary.csv", index=False)

    rankings = json.loads((root / "third_party" / "template_rankings.json").read_text())
    chosen = recommendations(rankings)
    (out / "visualization_recommendations.json").write_text(json.dumps(chosen, indent=2) + "\n")

    render(
        chosen["expression"]["renderer"],
        summaries["sample_mean"],
        audits / "sample_mean",
        "sample_mean",
        "Sample mean of min-max scaled values",
        False,
    )
    render(
        chosen["zeros"]["renderer"],
        summaries["zero_fraction"],
        audits / "zero_fraction",
        "zero_fraction",
        "Per-sample fraction of exact zeros",
        False,
    )

    plot_class_counts(summary, figures)
    plot_expression(summaries["sample_mean"], summaries["zero_fraction"], figures)
    for name in fitted:
        plot_pca(name, fitted[name], figures)

    inputs = [file_record(paths[key]) for key in ("rosmap", "gse120584", "gse215789")]
    pd.DataFrame(inputs).to_csv(out / "input_files.csv", index=False)
    packages = {}
    for package in ("numpy", "pandas", "matplotlib", "scipy", "PyYAML"):
        packages[package] = version(package)
    environment = {
        "python": sys.version,
        "platform": platform.platform(),
        "packages": packages,
        "random_seed": None,
        "stochastic_sampling": False,
        "biological_source_labels": BIOLOGICAL_SOURCE,
    }
    (out / "environment.json").write_text(json.dumps(environment, indent=2) + "\n")
    write_thesis_report(out / "THESIS_EDA_REPORT.md", summary, pca, outliers, inputs)
    print(summary.to_string(index=False))
    print(pca.to_string(index=False))
    print(outliers[outliers["row_type"] == "dataset"][["dataset", "n_flagged", "threshold"]].to_string(index=False))


if __name__ == "__main__":
    main()
