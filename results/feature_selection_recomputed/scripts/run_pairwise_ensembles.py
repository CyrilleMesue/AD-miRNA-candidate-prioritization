#!/usr/bin/env python
"""Run the 12 ensemble aggregators on each PAIR of datasets (26 rankings per pair).

Derived from run_per_dataset_ensembles.py. Universe = miRNAs present in all 26 rankings of the pair.
Original docstring of the parent script follows.

Run the 12 ensemble aggregators separately within each dataset.

For each of ROSMAP, GSE120584 and GSE215789, the 13 individual rankings
of that dataset are aggregated with the same EnsembleFeatureRanker calls
used for the 39-list ensemble (intersection universe, fixed rank
normalisation). Output: ensembles_pairwise/ensemble_rankings_pairwise.csv
and candidate_ensemble_ranks_pairwise.csv.

Usage: python scripts/run_per_dataset_ensembles.py
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("MPLBACKEND", "Agg")
ROOT = Path(__file__).resolve().parents[3]
OUT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

import matplotlib

matplotlib.use("Agg", force=True)
import pandas as pd

from modules.featureSelection.ensemble_feature_selection import EnsembleFeatureRanker
from prioritization.historical import AGGREGATORS, DECREASING_IMPORTANCE_SUFFIXES, sort_dict_features

DATASETS = ("ROSMAP", "GSE120584", "GSE215789")
INDIVIDUAL = OUT / "individual"
DEST = OUT / "ensembles_pairwise"
DEST.mkdir(parents=True, exist_ok=True)
CANDIDATES = ["hsa-miR-324-3p", "hsa-miR-574-3p", "hsa-miR-769-3p"]


def load_dataset(dataset: str) -> dict:
    weights = {}
    for meta in sorted(INDIVIDUAL.glob(f"{dataset}__*.meta.json")):
        method = meta.name.split("__", 1)[1].removesuffix(".meta.json")
        csv = INDIVIDUAL / f"{dataset}__{method}.csv"
        frame = pd.read_csv(csv)
        weights[f"{dataset}:{method}"] = dict(zip(frame["feature_id"], frame["score"]))
    return weights


def main() -> None:
    rows, meta = [], {}
    import itertools
    for d1, d2 in itertools.combinations(DATASETS, 2):
        dataset = f"{d1}+{d2}"
        weights = {**load_dataset(d1), **load_dataset(d2)}
        if len(weights) != 26:
            raise SystemExit(f"{dataset}: expected 26 rankings, found {len(weights)}")
        decreasing = [k for k in weights if k.split(":")[-1] in DECREASING_IMPORTANCE_SUFFIXES]
        ranker = EnsembleFeatureRanker(
            feature_weights=weights,
            feature_type="microRNA",
            outdir="",
            decreasing_importance_cases=decreasing,
            save_results=False,
            show_plots=False,
        )
        shared = set.intersection(*(set(d) for d in weights.values()))
        meta[dataset] = {
            "n_lists": len(weights),
            "list_lengths": {k: len(v) for k, v in weights.items()},
            "n_shared_features": len(shared),
            "decreasing_importance_cases": decreasing,
        }
        for aggregator in AGGREGATORS:
            method, kind = aggregator.split("_")
            t0 = time.perf_counter()
            if kind == "rank":
                scores = ranker.Rank_by_Ranks(plot=False, method=method)
            else:
                scores = ranker.Rank_by_Weights(plot=False, method=method)
            ascending = kind == "rank"
            ordered = sort_dict_features(scores, ascending=ascending, return_with_values=True)
            n = len(ordered)
            for position, (feature, score) in enumerate(ordered, start=1):
                rows.append(
                    dict(dataset=dataset, aggregator=aggregator, feature_id=feature,
                         ensemble_score=float(score), ensemble_rank=position, n_ranked=n)
                )
            print(f"{dataset} {aggregator} n={n} {time.perf_counter()-t0:.1f}s", flush=True)

    table = pd.DataFrame(rows)
    # Ties: competition rank (min) alongside the positional rank from the stable sort.
    table["tie_rank_min"] = table.groupby(["dataset", "aggregator"])["ensemble_score"].rank(method="min")
    table.loc[table.aggregator.str.endswith("_weight"), "tie_rank_min"] = (
        table[table.aggregator.str.endswith("_weight")]
        .groupby(["dataset", "aggregator"])["ensemble_score"].rank(method="min", ascending=False)
    )
    table["tie_rank_min"] = table["tie_rank_min"].astype(int)
    table["n_tied"] = table.groupby(["dataset", "aggregator", "ensemble_score"])["feature_id"].transform("size")
    table["percentile"] = (100 * table.ensemble_rank / table.n_ranked).round(1)
    table.to_csv(DEST / "ensemble_rankings_pairwise.csv", index=False)
    table[table.feature_id.isin(CANDIDATES)].sort_values(["feature_id", "dataset", "aggregator"]).to_csv(
        DEST / "candidate_ensemble_ranks_pairwise.csv", index=False
    )
    (DEST / "run_meta.json").write_text(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
