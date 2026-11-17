"""Summaries, unsupervised PCA, and sample-level outlier flags.

Matrices come from prioritization.matrices.load_ranking_datasets. Diagnosis is
not used to select features or to fit PCA.
"""
from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import chi2

DATASET_ORDER = ["ROSMAP", "GSE120584", "GSE215789"]

BIOLOGICAL_SOURCE = {
    "ROSMAP": "DLPFC",
    "GSE120584": "serum",
    "GSE215789": "plasma",
}

PATH_KEYS = {
    "ROSMAP": "rosmap",
    "GSE120584": "gse120584",
    "GSE215789": "gse215789",
}

EXPECTED = {
    "ROSMAP": {"n_samples": 378, "n_mirnas": 309, "n_ad": 209, "n_control": 169},
    "GSE120584": {"n_samples": 1309, "n_mirnas": 2521, "n_ad": 1021, "n_control": 288},
    "GSE215789": {"n_samples": 44, "n_mirnas": 1137, "n_ad": 23, "n_control": 21},
}

OUTLIER_QUANTILE = 0.999
MAD_SCALE = 1.4826
CRITERION = (
    "Squared robust distance on unsupervised PC1 and PC2. "
    "Each component is centered at its median and scaled by 1.4826 times its MAD. "
    "A component with MAD of 0 is omitted. "
    f"A sample is flagged when the squared distance exceeds the chi-square {OUTLIER_QUANTILE} quantile. "
    "Diagnosis is not used. No sample is removed."
)


def require_processed(paths: dict[str, str]) -> None:
    missing = [paths[key] for key in ("rosmap", "gse120584", "gse215789") if not Path(paths[key]).is_file()]
    if missing:
        joined = "\n".join(missing)
        raise SystemExit(
            "Missing processed ranking input(s):\n"
            f"{joined}\n"
            "The EDA does not substitute example or synthetic matrices."
        )


def load_matrices(paths: dict[str, str]) -> dict[str, pd.DataFrame]:
    from prioritization.matrices import load_ranking_datasets

    require_processed(paths)
    datasets = load_ranking_datasets(paths)
    for name in DATASET_ORDER:
        observed = _counts(datasets[name])
        expected = EXPECTED[name]
        if observed != expected:
            raise SystemExit(
                f"{name} ranking matrix does not match the expected final matrix. "
                f"Observed {observed}. Expected {expected}."
            )
    return datasets


def _counts(frame: pd.DataFrame) -> dict:
    values = frame["Diagnosis"].value_counts().to_dict()
    numeric = frame.drop(columns=["Diagnosis"])
    return {
        "n_samples": int(frame.shape[0]),
        "n_mirnas": int(numeric.shape[1]),
        "n_ad": int(values.get("AD", 0)),
        "n_control": int(values.get("Control", 0)),
    }


def expression_matrix(frame: pd.DataFrame) -> np.ndarray:
    return frame.drop(columns=["Diagnosis"]).to_numpy(dtype=float)


def summarize_datasets(datasets: dict[str, pd.DataFrame]) -> pd.DataFrame:
    rows = []
    for name in DATASET_ORDER:
        frame = datasets[name]
        numeric = frame.drop(columns=["Diagnosis"])
        values = numeric.to_numpy(dtype=float)
        counts = frame["Diagnosis"].value_counts().to_dict()
        rows.append(
            {
                "dataset": name,
                "biological_source": BIOLOGICAL_SOURCE[name],
                "n_ad": int(counts.get("AD", 0)),
                "n_control": int(counts.get("Control", 0)),
                "n_samples": int(frame.shape[0]),
                "n_mirnas": int(numeric.shape[1]),
                "n_missing": int(numeric.isna().sum().sum()),
                "minimum": float(np.nanmin(values)),
                "maximum": float(np.nanmax(values)),
                "zero_proportion": float(np.mean(values == 0)),
            }
        )
    return pd.DataFrame(rows)


def sample_summaries(datasets: dict[str, pd.DataFrame]) -> dict[str, list[dict]]:
    """One row per sample. Medians are retained for the table; the figure uses means.

    GSE215789 sample medians are mostly zero after min-max scaling, so a median
    box would not show the location of that matrix. Means stay on the same
    [0, 1] scale and use one number per sample.
    """
    means = []
    zeros = []
    for name in DATASET_ORDER:
        frame = datasets[name]
        values = expression_matrix(frame)
        for index in range(values.shape[0]):
            row = values[index]
            means.append({"group": name, "value": float(np.mean(row))})
            zeros.append({"group": name, "value": float(np.mean(row == 0))})
    return {"sample_mean": means, "zero_fraction": zeros}


def column_centered_pca(matrix: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Column-centered SVD. Matches the visualization skill PCA and does not scale."""
    if matrix.ndim != 2 or matrix.shape[0] < 2 or matrix.shape[1] < 2:
        raise ValueError("PCA requires at least two samples and two features")
    centered = matrix - matrix.mean(axis=0)
    _u, singular, vt = np.linalg.svd(centered, full_matrices=False)
    scores = centered @ vt[:2].T
    variance = singular**2
    total = float(variance.sum())
    explained = variance / total if total else np.zeros_like(variance)
    return scores, explained


def fit_dataset_pca(datasets: dict[str, pd.DataFrame]) -> tuple[pd.DataFrame, dict[str, dict]]:
    summary_rows = []
    fitted = {}
    for name in DATASET_ORDER:
        frame = datasets[name]
        matrix = expression_matrix(frame)
        scores, explained = column_centered_pca(matrix)
        summary_rows.append(
            {
                "dataset": name,
                "n_samples": int(matrix.shape[0]),
                "n_mirnas": int(matrix.shape[1]),
                "variance_pc1": float(explained[0]),
                "variance_pc2": float(explained[1]),
                "centering": "column mean",
                "scaling": "none",
                "diagnosis_in_fit": False,
            }
        )
        fitted[name] = {
            "scores": scores,
            "explained": explained,
            "sample_id": frame.index.astype(str).to_numpy(),
            "diagnosis": frame["Diagnosis"].astype(str).to_numpy(),
        }
    return pd.DataFrame(summary_rows), fitted


def assess_outliers(fitted: dict[str, dict]) -> pd.DataFrame:
    rows = []
    for name in DATASET_ORDER:
        scores = fitted[name]["scores"]
        distances, threshold, degrees = _robust_distance(scores)
        flagged = np.zeros(scores.shape[0], dtype=bool) if degrees == 0 else distances > threshold
        sample_ids = fitted[name]["sample_id"]
        diagnosis = fitted[name]["diagnosis"]
        flagged_ids = sample_ids[flagged]
        flagged_classes = diagnosis[flagged]
        rows.append(
            {
                "row_type": "dataset",
                "dataset": name,
                "sample_id": "",
                "diagnosis": "",
                "n_samples": int(scores.shape[0]),
                "n_flagged": int(flagged.sum()),
                "pc1": np.nan,
                "pc2": np.nan,
                "robust_distance_squared": np.nan,
                "threshold": threshold,
                "degrees_of_freedom": degrees,
                "criterion": CRITERION,
                "exclusion_recommended": "no",
                "flagged_sample_ids": ";".join(flagged_ids),
                "flagged_classes": ";".join(flagged_classes),
            }
        )
        for index in np.flatnonzero(flagged):
            rows.append(
                {
                    "row_type": "sample",
                    "dataset": name,
                    "sample_id": sample_ids[index],
                    "diagnosis": diagnosis[index],
                    "n_samples": int(scores.shape[0]),
                    "n_flagged": int(flagged.sum()),
                    "pc1": float(scores[index, 0]),
                    "pc2": float(scores[index, 1]),
                    "robust_distance_squared": float(distances[index]),
                    "threshold": threshold,
                    "degrees_of_freedom": degrees,
                    "criterion": CRITERION,
                    "exclusion_recommended": "no",
                    "flagged_sample_ids": "",
                    "flagged_classes": "",
                }
            )
        fitted[name]["flagged"] = flagged
        fitted[name]["robust_distance_squared"] = distances
        fitted[name]["outlier_threshold"] = threshold
    return pd.DataFrame(rows)


def _robust_distance(scores: np.ndarray) -> tuple[np.ndarray, float, int]:
    columns = []
    for component in range(scores.shape[1]):
        values = scores[:, component]
        center = float(np.median(values))
        mad = float(np.median(np.abs(values - center)))
        scale = MAD_SCALE * mad
        if scale == 0 or not np.isfinite(scale):
            continue
        columns.append((values - center) / scale)
    if not columns:
        return np.zeros(scores.shape[0]), float("nan"), 0
    standardized = np.column_stack(columns)
    distance = np.sum(standardized**2, axis=1)
    degrees = int(standardized.shape[1])
    threshold = float(chi2.ppf(OUTLIER_QUANTILE, degrees))
    return distance, threshold, degrees


def file_record(path: str) -> dict:
    target = Path(path)
    digest = hashlib.sha256()
    with target.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return {
        "path": str(target),
        "bytes": target.stat().st_size,
        "sha256": digest.hexdigest(),
    }
