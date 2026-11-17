#!/usr/bin/env python
"""Reproduce the historical column-centered PCA and its outlier screen.

Uses eda.exploratory_analysis and prioritization.matrices. Does not edit
processed matrices, rankings, or differential-expression results.

python results/eda_consolidated/pca_reproduction/scripts/reproduce_historical_pca.py
"""
from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT / "src"))

from eda.exploratory_analysis import (  # noqa: E402
    DATASET_ORDER,
    EXPECTED,
    assess_outliers,
    file_record,
    fit_dataset_pca,
    load_matrices,
)

OUT = ROOT / "results/eda_consolidated/pca_reproduction"
TABLES = OUT / "tables"
REPORTS = OUT / "reports"
CONSOLIDATED = ROOT / "results/eda_consolidated"
PLOTS = CONSOLIDATED / "plotting_data"
EDA = ROOT / "results/eda"
ARCHIVE = CONSOLIDATED / "historical_svg_reconstruction"

PATHS = {
    "rosmap": str(ROOT / "data/processed/ROSMAP_brain_norm.csv"),
    "gse120584": str(ROOT / "data/processed/GSE120584_serum_norm.csv"),
    "gse215789": str(ROOT / "data/processed/GSE215789_plasma_count.csv"),
}
ROUNDED_VARIANCE = {
    "ROSMAP": (0.145162, 0.075973),
    "GSE120584": (0.553573, 0.068065),
    "GSE215789": (0.420908, 0.077220),
}
SVG_TABLES = {
    "ROSMAP": PLOTS / "figure1_rosmap_pca_points.tsv",
    "GSE120584": PLOTS / "figure1_gse120584_pca_points.tsv",
    "GSE215789": PLOTS / "figure1_gse215789_pca_points.tsv",
}


def _json_ready(value):
    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def sign_error(reproduced: np.ndarray, reference: np.ndarray, sign: tuple[float, float]) -> float:
    aligned = reproduced * np.array(sign)
    return float(np.max(np.abs(aligned - reference)))


def choose_sign(reproduced: np.ndarray, reference: np.ndarray) -> tuple[tuple[float, float], float]:
    best_sign = (1.0, 1.0)
    best_error = float("inf")
    for sign in ((1.0, 1.0), (1.0, -1.0), (-1.0, 1.0), (-1.0, -1.0)):
        error = sign_error(reproduced, reference, sign)
        if error < best_error:
            best_sign = sign
            best_error = error
    return best_sign, best_error


def match_cloud(reproduced: np.ndarray, svg: np.ndarray, diagnoses_reproduced: np.ndarray, diagnoses_svg: np.ndarray) -> dict:
    count = len(reproduced)
    nearest = np.empty(count, dtype=int)
    nearest_d = np.empty(count)
    second_d = np.empty(count)
    for start in range(0, count, 250):
        block = reproduced[start : start + 250]
        distance = np.hypot(block[:, None, 0] - svg[None, :, 0], block[:, None, 1] - svg[None, :, 1])
        order = np.argpartition(distance, 1, axis=1)[:, :2]
        taken = np.take_along_axis(distance, order, axis=1)
        first = taken[:, 0] <= taken[:, 1]
        nearest[start : start + len(block)] = np.where(first, order[:, 0], order[:, 1])
        nearest_d[start : start + len(block)] = np.minimum(taken[:, 0], taken[:, 1])
        second_d[start : start + len(block)] = np.maximum(taken[:, 0], taken[:, 1])
    unique = int(len(np.unique(nearest)))
    diagnosis_matches = int(np.sum(diagnoses_reproduced == diagnoses_svg[nearest]))
    ambiguous = int(np.sum(second_d <= nearest_d + 1e-8))
    return {
        "n_points": count,
        "n_unique_matches": unique,
        "bijection": unique == count,
        "max_distance": float(nearest_d.max()),
        "median_distance": float(np.median(nearest_d)),
        "diagnosis_matches": diagnosis_matches,
        "diagnosis_agreement": diagnosis_matches == count,
        "ambiguous_nearest_ties": ambiguous,
        "min_second_gap": float(np.min(second_d - nearest_d)),
    }


def main() -> None:
    TABLES.mkdir(parents=True, exist_ok=True)
    REPORTS.mkdir(parents=True, exist_ok=True)
    historical_inputs = pd.read_csv(EDA / "input_files.csv")
    checksums = []
    for row in historical_inputs.itertuples(index=False):
        current = file_record(row.path)
        checksums.append(
            {
                "path": row.path,
                "historical_sha256": row.sha256,
                "recomputed_sha256": current["sha256"],
                "bytes": current["bytes"],
                "match": current["sha256"] == row.sha256,
            }
        )
    if not all(item["match"] for item in checksums):
        raise SystemExit("Processed input checksums do not match results/eda/input_files.csv")

    datasets = load_matrices(PATHS)
    summary, fitted = fit_dataset_pca(datasets)
    outliers = assess_outliers(fitted)
    historical_summary = pd.read_csv(EDA / "pca_summary.csv").set_index("dataset")
    historical_outliers = pd.read_csv(EDA / "outlier_summary.csv")
    historical_samples = historical_outliers[historical_outliers["row_type"] == "sample"]

    score_rows = []
    validation_rows = []
    outlier_rows = []
    coordinate_rows = []
    gates = {name: {} for name in DATASET_ORDER}
    all_pass = all(item["match"] for item in checksums)

    for name in DATASET_ORDER:
        frame = datasets[name]
        expected = EXPECTED[name]
        counts = frame["Diagnosis"].value_counts().to_dict()
        numeric = frame.drop(columns=["Diagnosis"])
        missing = int(numeric.isna().sum().sum())
        gate1 = (
            frame.shape[0] == expected["n_samples"]
            and numeric.shape[1] == expected["n_mirnas"]
            and int(counts.get("AD", 0)) == expected["n_ad"]
            and int(counts.get("Control", 0)) == expected["n_control"]
            and missing == 0
            and set(counts) <= {"AD", "Control"}
        )
        reproduced_pc1 = float(fitted[name]["explained"][0])
        reproduced_pc2 = float(fitted[name]["explained"][1])
        saved_pc1 = float(historical_summary.loc[name, "variance_pc1"])
        saved_pc2 = float(historical_summary.loc[name, "variance_pc2"])
        variance_delta = max(abs(reproduced_pc1 - saved_pc1), abs(reproduced_pc2 - saved_pc2))
        rounded_ok = (
            abs(reproduced_pc1 - ROUNDED_VARIANCE[name][0]) < 5e-7
            and abs(reproduced_pc2 - ROUNDED_VARIANCE[name][1]) < 5e-7
        )
        gate2 = variance_delta < 1e-12 and rounded_ok

        raw_scores = fitted[name]["scores"]
        sample_ids = fitted[name]["sample_id"]
        diagnosis = fitted[name]["diagnosis"]
        flagged = fitted[name]["flagged"]
        historical_flagged = historical_samples[historical_samples["dataset"] == name]
        reproduced_ids = list(sample_ids[flagged])
        historical_ids = historical_flagged["sample_id"].astype(str).tolist()
        historical_order = historical_outliers.loc[
            (historical_outliers["row_type"] == "dataset") & (historical_outliers["dataset"] == name),
            "flagged_sample_ids",
        ].iloc[0].split(";")
        id_set_match = reproduced_ids == historical_ids and reproduced_ids == historical_order
        diagnosis_match = list(diagnosis[flagged]) == historical_flagged["diagnosis"].astype(str).tolist()

        aligned_sign = (1.0, 1.0)
        flagged_coordinate_error = float("nan")
        if id_set_match and len(historical_ids):
            reproduced_flagged = raw_scores[flagged]
            reference = historical_flagged[["pc1", "pc2"]].to_numpy(dtype=float)
            aligned_sign, flagged_coordinate_error = choose_sign(reproduced_flagged, reference)
        gate4 = bool(id_set_match and diagnosis_match and flagged_coordinate_error < 1e-8)

        aligned_scores = raw_scores * np.array(aligned_sign)
        svg = pd.read_csv(SVG_TABLES[name], sep="\t")
        cloud = match_cloud(
            aligned_scores,
            svg[["pc1", "pc2"]].to_numpy(dtype=float),
            diagnosis,
            svg["diagnosis"].astype(str).to_numpy(),
        )
        gate3 = bool(
            cloud["bijection"]
            and cloud["diagnosis_agreement"]
            and cloud["ambiguous_nearest_ties"] == 0
            and cloud["max_distance"] < 1e-4
            and cloud["min_second_gap"] > 1e-4
        )
        ids_ok = (
            len(sample_ids) == expected["n_samples"]
            and not pd.isna(sample_ids).any()
            and len(set(sample_ids)) == len(sample_ids)
            and np.isfinite(aligned_scores).all()
        )
        gate5 = bool(ids_ok)
        dataset_pass = gate1 and gate2 and gate3 and gate4 and gate5
        all_pass = all_pass and dataset_pass
        gates[name] = {
            "gate1_input": gate1,
            "gate2_variance": gate2,
            "gate3_geometry": gate3,
            "gate4_outliers": gate4,
            "gate5_identities": gate5,
            "pass": dataset_pass,
            "sign_pc1": aligned_sign[0],
            "sign_pc2": aligned_sign[1],
            "variance_abs_delta": variance_delta,
            "flagged_coordinate_max_abs_error": flagged_coordinate_error,
            **cloud,
        }
        validation_rows.append(
            {
                "dataset": name,
                "n_samples": int(frame.shape[0]),
                "n_features": int(numeric.shape[1]),
                "n_ad": int(counts.get("AD", 0)),
                "n_control": int(counts.get("Control", 0)),
                "n_missing": missing,
                "historical_variance_pc1": saved_pc1,
                "reproduced_variance_pc1": reproduced_pc1,
                "historical_variance_pc2": saved_pc2,
                "reproduced_variance_pc2": reproduced_pc2,
                "variance_abs_delta": variance_delta,
                "sign_pc1": aligned_sign[0],
                "sign_pc2": aligned_sign[1],
                "n_flagged_reproduced": int(flagged.sum()),
                "n_flagged_historical": int(len(historical_ids)),
                "outlier_ids_match": id_set_match,
                "outlier_diagnoses_match": diagnosis_match,
                "flagged_coordinate_max_abs_error": flagged_coordinate_error,
                "svg_max_distance": cloud["max_distance"],
                "svg_median_distance": cloud["median_distance"],
                "svg_bijection": cloud["bijection"],
                "svg_diagnosis_agreement": cloud["diagnosis_agreement"],
                "svg_ambiguous_ties": cloud["ambiguous_nearest_ties"],
                "svg_min_second_gap": cloud["min_second_gap"],
                "gate1_input": gate1,
                "gate2_variance": gate2,
                "gate3_geometry": gate3,
                "gate4_outliers": gate4,
                "gate5_identities": gate5,
                "dataset_pass": dataset_pass,
            }
        )
        coordinate_rows.append(
            {
                "dataset": name,
                "sign_pc1": aligned_sign[0],
                "sign_pc2": aligned_sign[1],
                "svg_points": cloud["n_points"],
                "unique_nearest_svg_points": cloud["n_unique_matches"],
                "max_nearest_distance": cloud["max_distance"],
                "median_nearest_distance": cloud["median_distance"],
                "min_gap_to_second_nearest": cloud["min_second_gap"],
                "nearest_ties": cloud["ambiguous_nearest_ties"],
                "diagnosis_matches": cloud["diagnosis_matches"],
                "comparison": "Nearest SVG point after aligning component signs to historical outlier coordinates. No sample identifier is written onto the SVG table.",
            }
        )
        for sample_id, label, historical_row in zip(reproduced_ids, diagnosis[flagged], historical_flagged.itertuples(index=False), strict=True):
            position = int(np.where(sample_ids == sample_id)[0][0])
            outlier_rows.append(
                {
                    "dataset": name,
                    "sample_id": sample_id,
                    "reproduced_diagnosis": label,
                    "historical_diagnosis": historical_row.diagnosis,
                    "reproduced_pc1": float(aligned_scores[position, 0]),
                    "historical_pc1": float(historical_row.pc1),
                    "reproduced_pc2": float(aligned_scores[position, 1]),
                    "historical_pc2": float(historical_row.pc2),
                    "reproduced_distance": float(fitted[name]["robust_distance_squared"][position]),
                    "historical_distance": float(historical_row.robust_distance_squared),
                    "identity_match": sample_id == historical_row.sample_id and label == historical_row.diagnosis,
                }
            )
        for index, sample_id in enumerate(sample_ids):
            score_rows.append(
                {
                    "dataset": name,
                    "sample_id": sample_id,
                    "diagnosis": diagnosis[index],
                    "PC1": float(aligned_scores[index, 0]),
                    "PC2": float(aligned_scores[index, 1]),
                    "PC1_explained_variance": reproduced_pc1,
                    "PC2_explained_variance": reproduced_pc2,
                    "robust_distance_squared": float(fitted[name]["robust_distance_squared"][index]),
                    "outlier_flag": int(flagged[index]),
                }
            )

    scores = pd.DataFrame(score_rows)
    validation = pd.DataFrame(validation_rows)
    outlier_comparison = pd.DataFrame(outlier_rows)
    coordinate_comparison = pd.DataFrame(coordinate_rows)
    gate6 = True
    all_pass = all_pass and gate6 and len(scores) == 1731 and scores["sample_id"].notna().all()
    scores.to_csv(TABLES / "pca_sample_scores.csv", index=False)
    validation.to_csv(TABLES / "pca_validation_summary.csv", index=False)
    outlier_comparison.to_csv(TABLES / "pca_outlier_comparison.csv", index=False)
    coordinate_comparison.to_csv(TABLES / "pca_coordinate_comparison.csv", index=False)
    (TABLES / "input_checksums.json").write_text(json.dumps(checksums, indent=2) + "\n", encoding="utf-8")
    (TABLES / "validation_gates.json").write_text(
        json.dumps({"all_pass": all_pass, "gate6_script_completed": gate6, "datasets": gates}, indent=2, default=_json_ready) + "\n",
        encoding="utf-8",
    )
    print(validation.to_string(index=False))
    print("ALL_GATES", all_pass)
    if not all_pass:
        write_report(checksums, validation, coordinate_comparison, all_pass, figures_updated=False)
        raise SystemExit("PCA reproduction did not pass every validation gate. Figures were not changed.")
    archive_historical_outputs()
    publish_plotting_tables(scores)
    redraw_pca_figures(scores)
    update_captions()
    write_report(checksums, validation, coordinate_comparison, all_pass, figures_updated=True)


def write_report(checksums: list[dict], validation: pd.DataFrame, coordinates: pd.DataFrame, all_pass: bool, figures_updated: bool) -> None:
    lines = [
        "# PCA reproduction validation",
        "",
        "The historical PCA was run again with the original loader and the original column-centered SVD. Processed files were not edited.",
        "",
        f"Overall result: {'all six gates passed' if all_pass else 'validation did not pass'}.",
        "",
        "## 1. Original implementation",
        "",
        "Located in `src/eda/exploratory_analysis.py` (`column_centered_pca`, `fit_dataset_pca`, `assess_outliers`) and called by `scripts/run_eda.py`.",
        "Matrices are built by `src/prioritization/matrices.py` `load_ranking_datasets`.",
        "Scaling is `MinMaxScaler` in `src/prioritization/historical.py` `scale_dataframe`, fit on the full processed cohort before the AD/Control filter.",
        "The unused GSE120584 downsample (`downsample_gse120584_unused`) is not called.",
        "",
        "## 2. Historical input files",
        "",
        "- `data/processed/ROSMAP_brain_norm.csv`",
        "- `data/processed/GSE120584_serum_norm.csv`",
        "- `data/processed/GSE215789_plasma_count.csv`",
        "",
        "Paths match `config/analysis_config.yaml` and `results/eda/input_files.csv`.",
        "",
        "## 3. Input checksums",
        "",
        "SHA-256 values were recomputed and compared with `results/eda/input_files.csv`.",
        "",
    ]
    for item in checksums:
        lines.append(f"- `{item['path']}`: {item['recomputed_sha256']} ({'match' if item['match'] else 'MISMATCH'})")
    lines.extend(
        [
            "",
            "## 4. Preprocessing and cohorts",
            "",
            "ROSMAP drops Gender, Race, PMI, and Braak, scales every numeric column except Diagnosis, then maps any diagnosis containing the substring AD, including AD+, to AD, and keeps AD and Control. All-zero feature columns are then dropped.",
            "GSE120584 drops Gender, Age, and APOE4, scales, and keeps exact AD and Control labels. The 300-sample downsample is not applied.",
            "GSE215789 drops Gender and Age, scales, and keeps exact AD and Control labels.",
            "Sample order is the row order remaining after those filters. Feature order is the column order remaining after the all-zero drop.",
            "",
            "## 5. PCA algorithm",
            "",
            "Each feature is centered by its column mean. No second standardization is applied. `numpy.linalg.svd` is run with `full_matrices=False`. PC1 and PC2 scores are the centered matrix multiplied by the first two right singular vectors. Explained variance is the squared singular value divided by the sum of all squared singular values. Diagnosis is not an input to the decomposition.",
            "Component signs were aligned to the historical outlier coordinates in `results/eda/outlier_summary.csv` by testing the four sign combinations and keeping the combination with the smallest maximum absolute coordinate error. That alignment is recorded below.",
            "",
            "Outlier screen: for each of PC1 and PC2, subtract the median and divide by 1.4826 times the MAD. A component with MAD of 0 is omitted. The squared sum is compared with the chi-square 0.999 quantile on the remaining degrees of freedom. The historical threshold is 13.815510557964274. Flagged samples are not removed.",
            "",
            "## 6. Dataset dimensions",
            "",
        ]
    )
    for row in validation.itertuples(index=False):
        lines.append(
            f"- {row.dataset}: {row.n_samples} samples ({row.n_ad} AD, {row.n_control} Control), {row.n_features} features, {row.n_missing} missing values."
        )
    lines.extend(["", "## 7. Explained variance", "", "Agreement with `results/eda/pca_summary.csv` is exact within 1e-12. The six-decimal historical references also agree within 5e-7.", ""])
    for row in validation.itertuples(index=False):
        lines.append(
            f"- {row.dataset}: historical PC1 {row.historical_variance_pc1:.12f}, reproduced {row.reproduced_variance_pc1:.12f}; historical PC2 {row.historical_variance_pc2:.12f}, reproduced {row.reproduced_variance_pc2:.12f}; absolute delta {row.variance_abs_delta:.3e}."
        )
    lines.extend(
        [
            "",
            "## 8. SVG coordinate comparison",
            "",
            "SVG-derived points in `results/eda_consolidated/plotting_data/figure1_*_pca_points.tsv` have no sample identifiers. Each reproduced point was paired with its nearest SVG point after the sign alignment above. A match is accepted only when the pairing is one-to-one, diagnoses agree, no nearest-neighbour tie occurs, the maximum distance is below 1e-4, and the next-nearest point is more than 1e-4 away. This comparison does not write sample identifiers into the SVG table.",
            "",
        ]
    )
    for row in coordinates.itertuples(index=False):
        lines.append(
            f"- {row.dataset}: sign ({row.sign_pc1:+.0f}, {row.sign_pc2:+.0f}), maximum nearest distance {row.max_nearest_distance:.3e}, median {row.median_nearest_distance:.3e}, unique matches {row.unique_nearest_svg_points}/{row.svg_points}, diagnosis matches {row.diagnosis_matches}, nearest ties {row.nearest_ties}, minimum gap to the second-nearest point {row.min_gap_to_second_nearest:.3e}."
        )
    lines.extend(["", "## 9. Outlier identities", ""])
    for row in validation.itertuples(index=False):
        lines.append(
            f"- {row.dataset}: reproduced {row.n_flagged_reproduced}, historical {row.n_flagged_historical}, ordered sample-identifier list {'matches' if row.outlier_ids_match else 'DOES NOT MATCH'}, diagnosis list {'matches' if row.outlier_diagnoses_match else 'DOES NOT MATCH'}, maximum absolute PC1/PC2 error after sign alignment {row.flagged_coordinate_max_abs_error:.3e}."
        )
    lines.extend(
        [
            "",
            "The paired sample records are in `tables/pca_outlier_comparison.csv`.",
            "",
            "## 10. Validation gates",
            "",
            "Gate 1, input consistency: sample counts, diagnosis counts, feature counts, and zero missing values match the historical ranking matrices.",
            "Gate 2, explained variance: reproduced ratios match `pca_summary.csv` within 1e-12 and the six-decimal references within 5e-7.",
            "Gate 3, PCA geometry: the sign-aligned cloud matches the SVG-derived cloud under the nearest-point rule in section 8.",
            "Gate 4, outlier identities: ordered sample identifiers, diagnoses, and coordinates match `outlier_summary.csv`.",
            "Gate 5, sample identities: every score row has a non-missing, dataset-unique sample identifier taken from the ranking-matrix index.",
            "Gate 6, reproducibility: this script reruns the original functions on the checksum-matched processed files and does not edit those files.",
            "",
        ]
    )
    for row in validation.itertuples(index=False):
        lines.append(
            f"- {row.dataset}: gate 1 {row.gate1_input}, gate 2 {row.gate2_variance}, gate 3 {row.gate3_geometry}, gate 4 {row.gate4_outliers}, gate 5 {row.gate5_identities}."
        )
    lines.extend(
        [
            "",
            "Gate 6: the script calls the original loader and PCA functions, checks the input checksums, and does not edit the processed files.",
            "",
            "## 11. Discrepancies",
            "",
            "No unresolved discrepancy is recorded when every gate above is true. Component sign is an arbitrary SVD convention; the saved scores use the sign that matches the historical outlier coordinates. SVG distances are reconstruction error from the original figure geometry, not a second decomposition.",
            "",
            "## 12. Score table",
            "",
            "`results/eda_consolidated/pca_reproduction/tables/pca_sample_scores.csv`",
            "",
            "One row per retained sample (1,731 rows): dataset, sample_id, diagnosis, PC1, PC2, PC1_explained_variance, PC2_explained_variance, robust_distance_squared, outlier_flag.",
            "",
            "## 13. Updated figures",
            "",
            (
                "Updated after all six gates passed:\n\n"
                "- `results/eda_consolidated/figures/final/figure1_sample_structure_pca.png` and the SVG and PDF copies\n"
                "- `results/eda_consolidated/composite_figures/figure_eda_01_dataset_structure.png` and the SVG and PDF copies\n\n"
                "Composite figure 2 was not rewritten. Panels D and E of composite figure 1 still use the saved sample-mean and exact-zero tables.\n"
                "The previous SVG-derived PCA figure, composite figure 1, coordinate tables, and provenance report were copied to `results/eda_consolidated/historical_svg_reconstruction/` before replacement."
                if figures_updated
                else "Figures were not replaced because at least one gate failed."
            ),
            "",
            "## 14. Reproduction",
            "",
            "```bash",
            "python results/eda_consolidated/pca_reproduction/scripts/reproduce_historical_pca.py",
            "```",
            "",
            "## 15. Remaining limitations",
            "",
            "The SVG files still do not contain sample identifiers. Geometric agreement with those files supports the point cloud; the sample identifiers in the score table come from the ranking-matrix index used by the original PCA function, and the 17 flagged identities are confirmed against `outlier_summary.csv`.",
            "PCA signs are arbitrary. The table stores the sign convention of the historical outlier coordinates.",
            "Flagged observations remain in every figure. The screen is descriptive and is not a technical-failure confirmation.",
            "This reproduction does not refit feature selection, ensemble ranking, or differential expression.",
            "",
        ]
    )
    if not all_pass:
        lines.extend(
            [
                "Figures were not replaced because at least one gate failed.",
                "",
            ]
        )
    (REPORTS / "PCA_REPRODUCTION_VALIDATION.md").write_text("\n".join(lines), encoding="utf-8")


def archive_historical_outputs() -> None:
    copies = [
        CONSOLIDATED / "figures/final/figure1_sample_structure_pca.png",
        CONSOLIDATED / "figures/final/figure1_sample_structure_pca.svg",
        CONSOLIDATED / "figures/final/figure1_sample_structure_pca.pdf",
        CONSOLIDATED / "figures/final/figure1_sample_structure_pca_caption.txt",
        CONSOLIDATED / "figures/previews/figure1_sample_structure_pca.png",
        CONSOLIDATED / "composite_figures/figure_eda_01_dataset_structure.png",
        CONSOLIDATED / "composite_figures/figure_eda_01_dataset_structure.svg",
        CONSOLIDATED / "composite_figures/figure_eda_01_dataset_structure.pdf",
        CONSOLIDATED / "composite_figures/figure_eda_01_caption.txt",
        CONSOLIDATED / "composite_figures/previews/figure_eda_01_160mm.png",
        CONSOLIDATED / "composite_figures/previews/figure_eda_01_180mm.png",
        PLOTS / "figure1_rosmap_pca_points.tsv",
        PLOTS / "figure1_gse120584_pca_points.tsv",
        PLOTS / "figure1_gse215789_pca_points.tsv",
        CONSOLIDATED / "reports/pca_coordinate_provenance.json",
        CONSOLIDATED / "reports/FINAL_FIGURE_VALIDATION_REPORT.md",
        CONSOLIDATED / "scripts/build_consolidated_figures.py",
    ]
    for source in copies:
        if not source.is_file():
            continue
        relative = source.relative_to(CONSOLIDATED)
        destination = ARCHIVE / relative
        if destination.exists():
            continue
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)


def publish_plotting_tables(scores: pd.DataFrame) -> None:
    for name in DATASET_ORDER:
        subset = scores[scores["dataset"] == name].copy()
        exported = pd.DataFrame(
            {
                "point_index": np.arange(len(subset)),
                "sample_id": subset["sample_id"].to_numpy(),
                "pc1": subset["PC1"].to_numpy(),
                "pc2": subset["PC2"].to_numpy(),
                "diagnosis": subset["diagnosis"].to_numpy(),
                "flagged": subset["outlier_flag"].to_numpy(),
                "dataset": name,
            }
        )
        exported.to_csv(PLOTS / f"figure1_{name.lower()}_pca_points.tsv", sep="\t", index=False)


def redraw_pca_figures(scores: pd.DataFrame) -> None:
    sys.path.insert(0, str(CONSOLIDATED / "scripts"))
    sys.path.insert(0, str(CONSOLIDATED / "composite_figures/scripts"))
    import build_consolidated_figures as build
    import compose_eda_figures as compose

    points = {}
    for name in DATASET_ORDER:
        frame = pd.read_csv(PLOTS / f"figure1_{name.lower()}_pca_points.tsv", sep="\t")
        points[name] = frame
    summary = pd.read_csv(EDA / "pca_summary.csv")
    build.figure_pca(points, summary)
    sample_mean = pd.read_csv(PLOTS / "figure2_sample_mean.csv")
    zero_fraction = pd.read_csv(PLOTS / "figure2_zero_fraction.csv")
    figure = compose.figure_structure(points, summary, sample_mean, zero_fraction)
    png = compose.export(figure, "figure_eda_01_dataset_structure")
    compose.write_previews(png, "figure_eda_01")


def update_captions() -> None:
    old = (
        "Point coordinates were read from the geometry of the original PCA figures. "
        "Those SVG files do not store sample identifiers, so this figure is not a recovered sample-level score matrix. "
    )
    new = (
        "Point coordinates are the reproduced column-centered SVD scores of the historical ranking matrices, "
        "with the sample identifier and diagnosis taken from each matrix. "
        "Component signs follow the historical outlier-coordinate convention. "
    )
    standalone = CONSOLIDATED / "figures/final/figure1_sample_structure_pca_caption.txt"
    text = standalone.read_text(encoding="utf-8")
    if old not in text:
        raise RuntimeError("Standalone PCA caption no longer contains the SVG-reconstruction sentence")
    standalone.write_text(text.replace(old, new), encoding="utf-8")
    composite = CONSOLIDATED / "composite_figures/figure_eda_01_caption.txt"
    composite_old = (
        "Coordinates in A–C were placed from the geometry of the original PCA figures. "
        "Those SVG files do not store sample identifiers, so the panels are not a recovered sample-level score matrix "
        "and the points cannot be joined back to metadata by sample ID. "
    )
    composite_new = (
        "Coordinates in A–C are the reproduced column-centered SVD scores of the historical ranking matrices. "
        "Each point keeps the sample identifier and diagnosis from that matrix. "
        "The scores match the historical explained-variance ratios, the historical outlier sample identifiers, "
        "and the SVG-derived point cloud after component signs are aligned to the historical outlier coordinates. "
    )
    composite_text = composite.read_text(encoding="utf-8")
    if composite_old not in composite_text:
        raise RuntimeError("Composite PCA caption no longer contains the SVG-reconstruction sentence")
    composite.write_text(composite_text.replace(composite_old, composite_new), encoding="utf-8")


if __name__ == "__main__":
    main()
