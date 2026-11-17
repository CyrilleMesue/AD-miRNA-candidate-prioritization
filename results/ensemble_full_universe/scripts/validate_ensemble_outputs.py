#!/usr/bin/env python
"""Validate dataset ensembles and the coverage-aware integration."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd

from ensemble_common import (
    AGGREGATORS,
    CANDIDATES,
    DATASETS,
    EXPECTED,
    FEATURE_RANKERS,
    INDIVIDUAL,
    OUT,
    PRIMARY_RULE,
    load_rankings,
    rank_scores,
    run_aggregator,
)


def add(rows: list[dict], name: str, passed: bool, detail: str) -> None:
    rows.append({"check": name, "passed": bool(passed), "detail": detail})


def synthetic_case() -> dict[str, bool]:
    weights = {
        "A:t_test": {"a": 0.01, "b": 0.40},
        "A:lasso": {"a": 3.0, "b": 0.2},
        "B:t_test": {"b": 0.20, "c": 0.01},
        "B:lasso": {"b": 0.1, "c": 4.0},
    }
    combined = run_aggregator(weights, "mean_rank")
    within_a = run_aggregator({"A:t_test": weights["A:t_test"], "A:lasso": weights["A:lasso"]}, "mean_rank")
    within_b = run_aggregator({"B:t_test": weights["B:t_test"], "B:lasso": weights["B:lasso"]}, "mean_rank")
    repeated = run_aggregator({"A:t_test": weights["A:t_test"], "A:lasso": weights["A:lasso"]}, "mean_rank")
    ranked = rank_scores(within_a, smaller_is_better=True)
    best = ranked.loc[ranked["ensemble_rank"].idxmin()]
    worst = ranked.loc[ranked["average_rank"].idxmax()]
    return {
        "historical_combined_call_keeps_only_shared_name": set(combined) == {"b"},
        "within_dataset_keeps_dataset_only_feature": set(within_a) == {"a", "b"} and "c" not in within_a,
        "unmeasured_feature_has_no_score": "a" not in within_b and "c" not in within_a,
        "repeat_identical": within_a == repeated,
        "best_percentile_is_zero_when_unique": bool(np.isclose(best["rank_percentile"], 0.0)),
        "worst_percentile_is_one_when_unique": bool(np.isclose(worst["rank_percentile"], 1.0)),
    }


def main() -> None:
    rows: list[dict] = []
    rankings = load_rankings()
    add(rows, "all_39_inputs_present", len(rankings) == 3 and all(len(rankings[d]) == 13 for d in DATASETS), "13 methods in each of 3 datasets")
    checksum_path = OUT / "validation" / "input_checksums.json"
    checksums = json.loads(checksum_path.read_text()) if checksum_path.is_file() else []
    add(rows, "input_checksums_recorded", len(checksums) == 39, f"{len(checksums)} checksums")

    synthetic = synthetic_case()
    for name, passed in synthetic.items():
        add(rows, name, passed, "synthetic two-dataset example")

    integrated = pd.read_csv(OUT / "cross_dataset" / "primary_integration.csv")
    add(rows, "primary_rule_unchanged", set(integrated["primary_rule"]) == {PRIMARY_RULE}, PRIMARY_RULE)
    add(
        rows,
        "no_fabricated_cross_dataset_score",
        bool(integrated[["percentile_ROSMAP", "percentile_GSE120584", "percentile_GSE215789"]].notna().sum(axis=1).eq(integrated["n_datasets"]).all()),
        "blank dataset percentiles match coverage",
    )
    add(
        rows,
        "primary_mean_uses_only_observed_percentiles",
        bool(
            np.allclose(
                integrated["mean_available_percentile"],
                integrated[["percentile_ROSMAP", "percentile_GSE120584", "percentile_GSE215789"]].mean(axis=1, skipna=True),
            )
        ),
        "mean skipna",
    )

    for dataset in DATASETS:
        wide = pd.read_csv(OUT / "dataset_ensembles" / dataset / "dataset_ensemble_summary.csv")
        long = pd.read_csv(OUT / "dataset_ensembles" / dataset / "ensemble_long.csv")
        add(rows, f"{dataset}_feature_count", wide["original_mirna_id"].nunique() == EXPECTED[dataset], str(wide["original_mirna_id"].nunique()))
        add(rows, f"{dataset}_twelve_methods", set(long["aggregator"]) == set(AGGREGATORS), "12 aggregators")
        source_ids = set(rankings[dataset][FEATURE_RANKERS[0]]["feature_id"])
        add(rows, f"{dataset}_all_features_retained", set(wide["original_mirna_id"]) == source_ids, "matches first individual ranking")
        finite = np.isfinite(long["ensemble_score"]).all() and np.isfinite(long["rank_percentile"]).all()
        add(rows, f"{dataset}_finite_scores", bool(finite), "scores and percentiles")
        in_range = long["rank_percentile"].between(0, 1).all()
        add(rows, f"{dataset}_percentile_range", bool(in_range), "[0, 1]")
        direction_ok = True
        formula_ok = True
        for aggregator, group in long.groupby("aggregator"):
            best = group.loc[group["ensemble_rank"].idxmin(), "ensemble_score"]
            if "rank" in aggregator and not np.isclose(best, group["ensemble_score"].min()):
                direction_ok = False
            if "weight" in aggregator and not np.isclose(best, group["ensemble_score"].max()):
                direction_ok = False
            expected = (group["average_rank"] - 1.0) / (EXPECTED[dataset] - 1.0)
            if not np.allclose(group["rank_percentile"], expected):
                formula_ok = False
        add(rows, f"{dataset}_direction", direction_ok, "rank methods smaller-better; weight methods larger-better")
        add(rows, f"{dataset}_percentile_formula", formula_ok, "(average_rank - 1) / (n - 1)")
        ties = long[long["tie_group_size"] > 1]
        if ties.empty:
            add(rows, f"{dataset}_ties", True, "no tied ensemble scores")
        else:
            shared = ties.groupby(["aggregator", "ensemble_score"])["rank_percentile"].nunique().eq(1).all()
            add(rows, f"{dataset}_ties_share_percentile", bool(shared), f"{len(ties)} tied rows")

    coverage = pd.read_csv(OUT / "validation" / "identifier_coverage_summary.csv")
    union = coverage[coverage["dataset"] == "union_resolved"].iloc[0]
    add(
        rows,
        "coverage_counts",
        int(union["resolved_in_one_dataset"]) + int(union["resolved_in_two_datasets"]) + int(union["resolved_in_three_datasets"]) == int(union["resolved_mature"]),
        f"1:{int(union['resolved_in_one_dataset'])} 2:{int(union['resolved_in_two_datasets'])} 3:{int(union['resolved_in_three_datasets'])}",
    )
    add(rows, "complete_case_is_not_the_only_universe", int(union["resolved_mature"]) > 58, str(int(union["resolved_mature"])))

    candidate_cross = pd.read_csv(OUT / "candidates" / "candidate_cross_dataset_summary.csv")
    add(rows, "three_candidates_tabulated", set(candidate_cross["candidate"]) == set(CANDIDATES), "pre-specified names")
    sensitivity = pd.read_csv(OUT / "cross_dataset" / "sensitivity_rankings.csv")
    add(rows, "sensitivity_table_complete", len(sensitivity) == len(integrated), str(len(sensitivity)))
    add(
        rows,
        "best_is_not_labeled_primary",
        "sensitivity_best_rank" in sensitivity.columns and "primary_rank" in sensitivity.columns,
        "best percentile remains a sensitivity column",
    )

    recomputed_sentinel = (OUT.parents[1] / "results" / "feature_selection_recomputed" / "ensembles" / "ensemble_rankings.csv").stat().st_mtime
    add(rows, "58_ensemble_file_still_present", recomputed_sentinel > 0, "read-only sentinel")
    individual_ok = all((INDIVIDUAL / f"{dataset}__{method}.csv").is_file() for dataset in DATASETS for method in FEATURE_RANKERS)
    add(rows, "individual_rankings_still_present", individual_ok, "39 files")

    frame = pd.DataFrame(rows)
    frame.to_csv(OUT / "validation" / "validation_checks.csv", index=False)
    failed = frame.loc[~frame["passed"], "check"].tolist()
    if failed:
        raise SystemExit("Validation failed: " + ", ".join(failed))
    print("VALIDATION_PASSED", len(frame))


if __name__ == "__main__":
    main()
