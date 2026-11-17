"""Shared paths and rank definitions for the full-universe ensemble analysis."""

from __future__ import annotations

import hashlib
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
OUT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from modules.featureSelection.ensemble_feature_selection import EnsembleFeatureRanker
from prioritization.historical import (
    AGGREGATORS,
    DECREASING_IMPORTANCE_SUFFIXES,
    FEATURE_RANKERS,
)

INDIVIDUAL = ROOT / "results/feature_selection_recomputed/individual"
IDENTITY = ROOT / "results/biological_eda/supporting_data/feature_identity_audit.tsv"
ENSEMBLE_58 = ROOT / "results/feature_selection_recomputed/ensembles"
HISTORICAL_TOP20 = ROOT / "results/feature_selection_recomputed/reference/cell23_historical_top20.json"
DATASETS = ("ROSMAP", "GSE120584", "GSE215789")
EXPECTED = {"ROSMAP": 309, "GSE120584": 2521, "GSE215789": 1137}
CANDIDATES = ("hsa-miR-324-3p", "hsa-miR-574-3p", "hsa-miR-769-3p")
PRIMARY_RULE = "mean_of_available_dataset_median_percentiles"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_rankings() -> dict[str, dict[str, pd.DataFrame]]:
    loaded: dict[str, dict[str, pd.DataFrame]] = {}
    for dataset in DATASETS:
        loaded[dataset] = {}
        for method in FEATURE_RANKERS:
            path = INDIVIDUAL / f"{dataset}__{method}.csv"
            if not path.is_file():
                raise FileNotFoundError(path)
            frame = pd.read_csv(path)
            if list(frame.columns[:5]) != [
                "feature_id",
                "score",
                "rank_stable",
                "tie_group_size",
                "score_is_finite",
            ]:
                raise ValueError(f"Unexpected columns in {path}")
            if frame["feature_id"].duplicated().any():
                raise ValueError(f"Duplicate features in {path}")
            if len(frame) != EXPECTED[dataset]:
                raise ValueError(f"{path} has {len(frame)} rows, expected {EXPECTED[dataset]}")
            scores = frame["score"].to_numpy(dtype=float)
            if not bool(frame["score_is_finite"].all()) or not np.isfinite(scores).all():
                raise ValueError(f"Non-finite score in {path}")
            loaded[dataset][method] = frame
    return loaded


def score_maps(rankings: dict[str, dict[str, pd.DataFrame]], dataset: str) -> dict[str, dict[str, float]]:
    return {
        f"{dataset}:{method}": dict(zip(frame["feature_id"], frame["score"].astype(float)))
        for method, frame in rankings[dataset].items()
    }


def run_aggregator(weights: dict[str, dict[str, float]], aggregator: str) -> dict[str, float]:
    """Historical aggregator on one complete dataset. Intersection removes nothing."""
    decreasing = [key for key in weights if key.split(":")[-1] in DECREASING_IMPORTANCE_SUFFIXES]
    ranker = EnsembleFeatureRanker(
        feature_weights=weights,
        feature_type="microRNA",
        outdir="",
        decreasing_importance_cases=decreasing,
        save_results=False,
        show_plots=False,
    )
    method, kind = aggregator.split("_")
    if kind == "rank":
        return ranker.Rank_by_Ranks(plot=False, method=method, universe="intersection")
    return ranker.Rank_by_Weights(plot=False, method=method, universe="intersection")


def rank_scores(scores: dict[str, float], smaller_is_better: bool) -> pd.DataFrame:
    frame = pd.Series(scores, name="ensemble_score", dtype=float)
    average_rank = frame.rank(method="average", ascending=smaller_is_better)
    competition_rank = frame.rank(method="min", ascending=smaller_is_better)
    tie_size = frame.groupby(frame).transform("size")
    n_features = int(frame.shape[0])
    percentile = (average_rank - 1.0) / (n_features - 1.0)
    out = pd.DataFrame(
        {
            "original_mirna_id": frame.index.astype(str),
            "ensemble_score": frame.to_numpy(),
            "ensemble_rank": competition_rank.to_numpy().astype(int),
            "average_rank": average_rank.to_numpy(),
            "rank_percentile": percentile.to_numpy(),
            "tie_group_size": tie_size.to_numpy().astype(int),
            "n_features": n_features,
            "score_direction": "smaller_is_better" if smaller_is_better else "larger_is_better",
        }
    )
    return out.sort_values(["ensemble_rank", "original_mirna_id"]).reset_index(drop=True)


def attach_identity(frame: pd.DataFrame, dataset: str) -> pd.DataFrame:
    audit = pd.read_csv(IDENTITY, sep="\t")
    audit = audit[audit["dataset"] == dataset][
        ["feature_id", "class", "resolved_mature_id"]
    ].drop_duplicates("feature_id")
    merged = frame.merge(audit, left_on="original_mirna_id", right_on="feature_id", how="left")
    status = merged["class"].fillna("not_in_identity_audit")
    resolved = merged["resolved_mature_id"].where(status == "resolved mature", "")
    resolved = resolved.fillna("")
    merged.insert(0, "dataset", dataset)
    merged["identifier_resolution_status"] = status
    merged["resolved_mature_mirna_id"] = resolved
    return merged.drop(columns=["feature_id", "class", "resolved_mature_id"])
