#!/usr/bin/env python
"""Twelve ensemble rankings inside each dataset, using every measured miRNA."""

from __future__ import annotations

import json
import platform
from importlib.metadata import version

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from ensemble_common import (
    AGGREGATORS,
    DATASETS,
    EXPECTED,
    FEATURE_RANKERS,
    INDIVIDUAL,
    OUT,
    attach_identity,
    load_rankings,
    rank_scores,
    run_aggregator,
    score_maps,
    sha256_file,
)


def wide_summary(long: pd.DataFrame) -> pd.DataFrame:
    identity = long[
        ["dataset", "original_mirna_id", "resolved_mature_mirna_id", "identifier_resolution_status"]
    ].drop_duplicates()
    pieces = [identity.set_index("original_mirna_id")]
    for aggregator, group in long.groupby("aggregator", sort=False):
        block = group.set_index("original_mirna_id")[
            ["ensemble_score", "ensemble_rank", "rank_percentile"]
        ].rename(
            columns={
                "ensemble_score": f"{aggregator}__ensemble_score",
                "ensemble_rank": f"{aggregator}__ensemble_rank",
                "rank_percentile": f"{aggregator}__rank_percentile",
            }
        )
        pieces.append(block)
    wide = pd.concat(pieces, axis=1).reset_index()
    rank_columns = [f"{name}__ensemble_rank" for name in AGGREGATORS]
    percentile_columns = [f"{name}__rank_percentile" for name in AGGREGATORS]
    ranks = wide[rank_columns]
    percentiles = wide[percentile_columns]
    wide["median_ensemble_rank"] = ranks.median(axis=1)
    wide["best_ensemble_rank"] = ranks.min(axis=1).astype(int)
    wide["worst_ensemble_rank"] = ranks.max(axis=1).astype(int)
    wide["median_rank_percentile"] = percentiles.median(axis=1)
    wide["top10_method_count"] = (ranks <= 10).sum(axis=1).astype(int)
    wide["top20_method_count"] = (ranks <= 20).sum(axis=1).astype(int)
    wide["top50_method_count"] = (ranks <= 50).sum(axis=1).astype(int)
    order = wide["median_rank_percentile"].rank(method="min", ascending=True).astype(int)
    wide.insert(4, "dataset_summary_rank", order)
    return wide.sort_values(["dataset_summary_rank", "original_mirna_id"]).reset_index(drop=True)


def agreement_tables(long: pd.DataFrame, dataset: str) -> tuple[pd.DataFrame, dict]:
    matrix = long.pivot(index="original_mirna_id", columns="aggregator", values="rank_percentile")
    matrix = matrix[list(AGGREGATORS)]
    rows = []
    correlations = []
    for left_index, left in enumerate(AGGREGATORS):
        for right in AGGREGATORS[left_index + 1 :]:
            rho = float(spearmanr(matrix[left], matrix[right]).statistic)
            correlations.append(rho)
            overlaps = {}
            for k in (10, 20, 50):
                left_set = set(matrix.index[matrix[left].rank(method="min", ascending=True) <= k])
                right_set = set(matrix.index[matrix[right].rank(method="min", ascending=True) <= k])
                overlaps[f"top{k}_overlap"] = len(left_set & right_set)
            rows.append({"dataset": dataset, "method_a": left, "method_b": right, "spearman": rho, **overlaps})
    pairs = pd.DataFrame(rows)
    summary = {
        "dataset": dataset,
        "n_features": int(matrix.shape[0]),
        "n_methods": int(matrix.shape[1]),
        "median_pairwise_spearman": float(np.median(correlations)),
        "minimum_pairwise_spearman": float(np.min(correlations)),
        "maximum_pairwise_spearman": float(np.max(correlations)),
        "median_top10_overlap": float(pairs["top10_overlap"].median()),
        "median_top20_overlap": float(pairs["top20_overlap"].median()),
        "median_top50_overlap": float(pairs["top50_overlap"].median()),
    }
    return pairs, summary


def main() -> None:
    rankings = load_rankings()
    checksums = []
    pair_frames = []
    summaries = []
    for dataset in DATASETS:
        weights = score_maps(rankings, dataset)
        folder = OUT / "dataset_ensembles" / dataset
        folder.mkdir(parents=True, exist_ok=True)
        long_parts = []
        for aggregator in AGGREGATORS:
            scores = run_aggregator(weights, aggregator)
            if set(scores) != set(weights[f"{dataset}:{FEATURE_RANKERS[0]}"]):
                raise SystemExit(f"{dataset} {aggregator} dropped or added a feature")
            ranked = rank_scores(scores, smaller_is_better="rank" in aggregator)
            ranked.insert(0, "aggregator", aggregator)
            ranked = attach_identity(ranked, dataset)
            ranked.to_csv(folder / f"{aggregator}.csv", index=False)
            long_parts.append(ranked)
        for method in FEATURE_RANKERS:
            path = INDIVIDUAL / f"{dataset}__{method}.csv"
            checksums.append({"path": str(path.relative_to(OUT.parents[1])), "sha256": sha256_file(path), "n_rows": EXPECTED[dataset]})
        long = pd.concat(long_parts, ignore_index=True)
        long.to_csv(folder / "ensemble_long.csv", index=False)
        wide = wide_summary(long)
        if wide["original_mirna_id"].nunique() != EXPECTED[dataset]:
            raise SystemExit(f"{dataset} summary has {wide['original_mirna_id'].nunique()} features")
        wide.to_csv(folder / "dataset_ensemble_summary.csv", index=False)
        wide.head(20).to_csv(folder / "top20_by_median_percentile.csv", index=False)
        pairs, summary = agreement_tables(long, dataset)
        pair_frames.append(pairs)
        summaries.append(summary)
        repeated = run_aggregator(weights, "mean_rank")
        if repeated != long_parts[AGGREGATORS.index("mean_rank")].set_index("original_mirna_id")["ensemble_score"].to_dict():
            # Dict equality is sensitive to key order; compare values.
            saved = long_parts[AGGREGATORS.index("mean_rank")].set_index("original_mirna_id")["ensemble_score"]
            if not np.allclose([repeated[feature] for feature in saved.index], saved.to_numpy()):
                raise SystemExit(f"{dataset} mean_rank repeat differed")
    validation = OUT / "validation"
    validation.mkdir(parents=True, exist_ok=True)
    pd.concat(pair_frames, ignore_index=True).to_csv(validation / "ensemble_agreement_pairs.csv", index=False)
    pd.DataFrame(summaries).to_csv(validation / "ensemble_agreement_summary.csv", index=False)
    (validation / "input_checksums.json").write_text(json.dumps(checksums, indent=2))
    versions = {
        "python": platform.python_version(),
        "numpy": version("numpy"),
        "pandas": version("pandas"),
        "scipy": version("scipy"),
    }
    (validation / "software_versions.json").write_text(json.dumps(versions, indent=2))
    print("DATASET_ENSEMBLES_DONE", json.dumps(summaries))


if __name__ == "__main__":
    main()
