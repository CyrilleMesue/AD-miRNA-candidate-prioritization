#!/usr/bin/env python
"""Coverage-aware ensembles from the recomputed individual rankings.

Stage A applies the original 12 aggregators inside each dataset.
Stage B reports those ranks for resolved mature identifiers and does not
fill a score for a dataset that did not measure the miRNA.

The 13 feature-selection methods are not rerun.
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[3]
OUT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from modules.featureSelection.ensemble_feature_selection import EnsembleFeatureRanker
from prioritization.original import (
    AGGREGATORS,
    DECREASING_IMPORTANCE_SUFFIXES,
    FEATURE_RANKERS,
    sort_dict_features,
)

CONFIG = yaml.safe_load((OUT / "config/full_universe_config.yaml").read_text())
DATASETS = CONFIG["datasets"]
EXPECTED = CONFIG["expected_features"]
INDIVIDUAL = ROOT / CONFIG["inputs"]["individual_rankings"]
IDENTITY = ROOT / CONFIG["inputs"]["identity_audit"]
ENSEMBLE_58 = ROOT / CONFIG["inputs"]["recomputed_58_ensembles"]
ORIGINAL_TOP20 = ROOT / CONFIG["inputs"]["original_top20"]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_rankings() -> dict[str, dict[str, dict]]:
    loaded = {dataset: {} for dataset in DATASETS}
    for dataset in DATASETS:
        for method in FEATURE_RANKERS:
            path = INDIVIDUAL / f"{dataset}__{method}.csv"
            frame = pd.read_csv(path)
            if frame["feature_id"].duplicated().any():
                raise ValueError(f"Duplicate features in {path}")
            loaded[dataset][method] = dict(zip(frame["feature_id"], frame["score"].astype(float)))
    return loaded


def weights_for_dataset(rankings: dict, dataset: str) -> dict:
    return {f"{dataset}:{method}": rankings[dataset][method] for method in FEATURE_RANKERS}


def run_aggregator(weights: dict, aggregator: str, universe: str = "intersection") -> dict:
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
        return ranker.Rank_by_Ranks(plot=False, method=method, universe=universe)
    return ranker.Rank_by_Weights(plot=False, method=method, universe=universe)


def feature_coverage(weights: dict) -> pd.DataFrame:
    present: dict[str, list[str]] = {}
    for key, scores in weights.items():
        dataset = key.split(":")[0]
        for feature in scores:
            present.setdefault(feature, []).append(dataset)
    rows = []
    for feature, datasets in present.items():
        unique = sorted(set(datasets))
        rows.append(
            {
                "feature_id": feature,
                "n_rankings": len(datasets),
                "n_datasets": len(unique),
                "datasets": "|".join(unique),
            }
        )
    return pd.DataFrame(rows)


def scores_to_frame(scores: dict, aggregator: str, dataset: str, coverage: pd.DataFrame | None = None) -> pd.DataFrame:
    ascending = "rank" in aggregator
    ordered = sort_dict_features(scores, ascending=ascending, return_with_values=True)
    values = [value for _, value in ordered]
    tie_counts = {value: values.count(value) for value in set(values)}
    rows = []
    for position, (feature, score) in enumerate(ordered, start=1):
        rows.append(
            {
                "dataset": dataset,
                "aggregator": aggregator,
                "feature_id": feature,
                "ensemble_score": float(score),
                "ensemble_rank": position,
                "tie_group_size": int(tie_counts[score]),
                "n_features": len(ordered),
                "n_methods": 13,
                "score_direction": "smaller_is_better" if ascending else "larger_is_better",
                "missing_feature_policy": "not_applicable_within_dataset",
            }
        )
    frame = pd.DataFrame(rows)
    if coverage is not None:
        frame = frame.merge(coverage, on="feature_id", how="left")
        frame["missing_feature_policy"] = "observed_values_only"
        frame["rank_within_n_datasets"] = (
            frame.groupby("n_datasets")["ensemble_score"].rank(ascending=ascending, method="first").astype(int)
        )
    return frame


def synthetic_complete_case_check() -> dict:
    """Show that the original dropna keeps only names present in every dictionary."""
    weights = {
        "A:t_test": {"a": 0.01, "b": 0.40},
        "A:lasso": {"a": 3.0, "b": 0.2},
        "B:t_test": {"b": 0.20, "c": 0.01},
        "B:lasso": {"b": 0.1, "c": 4.0},
    }
    within_a_weights = {"A:t_test": weights["A:t_test"], "A:lasso": weights["A:lasso"]}
    within_b_weights = {"B:t_test": weights["B:t_test"], "B:lasso": weights["B:lasso"]}
    combined = run_aggregator(weights, "mean_rank")
    union = run_aggregator(weights, "mean_rank", universe="union")
    within_a = run_aggregator(within_a_weights, "mean_rank", universe="union")
    within_b = run_aggregator(within_b_weights, "mean_rank", universe="union")
    repeated = run_aggregator(within_a_weights, "mean_rank", universe="union")
    worst_rank_fill = {
        "A:t_test": weights["A:t_test"],
        "A:lasso": weights["A:lasso"],
        "B:t_test": {**weights["B:t_test"], "a": 999.0},
        "B:lasso": {**weights["B:lasso"], "a": 0.0},
    }
    filled = run_aggregator(worst_rank_fill, "mean_rank", universe="union")
    return {
        "combined_features": sorted(combined),
        "union_features": sorted(union),
        "within_a_features": sorted(within_a),
        "within_b_features": sorted(within_b),
        "combined_is_intersection": sorted(combined) == ["b"],
        "union_keeps_every_measured_feature": sorted(union) == ["a", "b", "c"],
        "within_a_keeps_dataset_only_feature": "a" in within_a and "c" not in within_a,
        "within_b_keeps_dataset_only_feature": "c" in within_b and "a" not in within_b,
        "unmeasured_feature_has_no_score": "c" not in within_a and "a" not in within_b,
        "dataset_only_score_matches_its_own_lists": bool(np.isclose(union["a"], within_a["a"]) and np.isclose(union["c"], within_b["c"])),
        "worst_rank_fill_changes_the_score": not bool(np.isclose(union["a"], filled["a"])),
        "repeat_identical": within_a == repeated,
    }


def build_inventory(rankings: dict) -> tuple[pd.DataFrame, dict]:
    identity = pd.read_csv(IDENTITY, sep="\t")
    rows = []
    for dataset in DATASETS:
        features = list(rankings[dataset][FEATURE_RANKERS[0]])
        audit = identity[identity["dataset"] == dataset].set_index("feature_id")
        for feature in features:
            if feature not in audit.index:
                klass = "not_in_identity_audit"
                mature = ""
                reason = "Feature is in the ranking but not in the identity audit."
            else:
                record = audit.loc[feature]
                klass = record["class"]
                mature = "" if pd.isna(record["resolved_mature_id"]) else str(record["resolved_mature_id"])
                reason = "" if pd.isna(record["reason"]) else str(record["reason"])
            rows.append(
                {
                    "dataset": dataset,
                    "feature_id": feature,
                    "identity_class": klass,
                    "resolved_mature_id": mature,
                    "reason": reason,
                }
            )
    inventory = pd.DataFrame(rows)
    resolved = inventory[inventory["identity_class"] == "resolved mature"].copy()
    duplicate_ids = set()
    for dataset, group in resolved.groupby("dataset"):
        counts = group["resolved_mature_id"].value_counts()
        duplicate_ids.update(counts[counts > 1].index)
    resolved = resolved[~resolved["resolved_mature_id"].isin(duplicate_ids)]
    presence = (
        resolved.groupby("resolved_mature_id")["dataset"]
        .agg(lambda values: tuple(sorted(set(values))))
        .rename("datasets")
        .reset_index()
    )
    presence["n_datasets"] = presence["datasets"].map(len)
    summary = {
        "distinct_original_identifiers": int(inventory["feature_id"].nunique()),
        "distinct_original_by_dataset": inventory.groupby("dataset")["feature_id"].nunique().to_dict(),
        "raw_name_in_one_dataset": int((inventory.groupby("feature_id")["dataset"].nunique() == 1).sum()),
        "raw_name_in_two_datasets": int((inventory.groupby("feature_id")["dataset"].nunique() == 2).sum()),
        "raw_name_in_three_datasets": int((inventory.groupby("feature_id")["dataset"].nunique() == 3).sum()),
        "distinct_resolved_mature_ids": int(presence["resolved_mature_id"].nunique()),
        "resolved_in_exactly_one_dataset": int((presence["n_datasets"] == 1).sum()),
        "resolved_in_exactly_two_datasets": int((presence["n_datasets"] == 2).sum()),
        "resolved_in_all_three_datasets": int((presence["n_datasets"] == 3).sum()),
        "original_features_unresolved": int((inventory["identity_class"] == "unresolved").sum()),
        "original_features_ambiguous": int((inventory["identity_class"] == "ambiguous").sum()),
        "original_features_non_mirna": int((inventory["identity_class"] == "non-miRNA/control").sum()),
        "duplicate_resolved_ids_excluded": sorted(duplicate_ids),
    }
    return inventory, {"summary": summary, "resolved_presence": presence, "resolved_rows": resolved}


def cross_dataset_tables(within: dict[str, pd.DataFrame], resolved_rows: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    lookup = resolved_rows.set_index(["dataset", "feature_id"])["resolved_mature_id"]
    long_rows = []
    for dataset, frame in within.items():
        for record in frame.itertuples(index=False):
            key = (record.dataset, record.feature_id)
            if key not in lookup.index:
                continue
            long_rows.append(
                {
                    "resolved_mature_id": lookup.loc[key],
                    "aggregator": record.aggregator,
                    "dataset": record.dataset,
                    "feature_id": record.feature_id,
                    "ensemble_rank": int(record.ensemble_rank),
                    "ensemble_score": float(record.ensemble_score),
                    "n_features_in_dataset": int(record.n_features),
                    "percentile": float(record.ensemble_rank) / float(record.n_features),
                    "measured": True,
                }
            )
    long = pd.DataFrame(long_rows)
    grouped = []
    for (mature, aggregator), group in long.groupby(["resolved_mature_id", "aggregator"], sort=False):
        grouped.append(
            {
                "resolved_mature_id": mature,
                "aggregator": aggregator,
                "n_datasets": int(group["dataset"].nunique()),
                "datasets": "|".join(sorted(group["dataset"])),
                "mean_observed_percentile": float(group["percentile"].mean()),
                "best_observed_percentile": float(group["percentile"].min()),
                "best_dataset": group.loc[group["percentile"].idxmin(), "dataset"],
                "unmeasured_datasets_left_blank": True,
                "summary_is_not_one_of_the_12_algorithms": True,
            }
        )
    return long, pd.DataFrame(grouped)


def compare_with_58(summary: pd.DataFrame, within: dict[str, pd.DataFrame]) -> pd.DataFrame:
    from scipy.stats import spearmanr

    original = json.loads(ORIGINAL_TOP20.read_text())
    rows = []
    for aggregator in AGGREGATORS:
        path = ENSEMBLE_58 / f"{aggregator}_top20.json"
        recomputed = json.loads(path.read_text()) if path.is_file() else []
        original_names = original.get(aggregator.upper(), [])
        shared = [name for name in recomputed if name in original_names]
        rows.append(
            {
                "comparison": "recomputed_58_vs_original_printed_top20",
                "aggregator": aggregator,
                "dataset": "combined_39_dictionaries",
                "left_n": 58,
                "right_n": 20,
                "top20_overlap": len(set(recomputed) & set(original_names)),
                "spearman_on_shared_top20": (
                    spearmanr(
                        [original_names.index(name) for name in shared],
                        [recomputed.index(name) for name in shared],
                    ).statistic
                    if len(shared) > 1
                    else np.nan
                ),
            }
        )
        global_scores = pd.read_csv(ENSEMBLE_58 / "ensemble_rankings.csv")
        global_scores = global_scores[global_scores["aggregator"] == aggregator]
        stratum = summary[(summary["aggregator"] == aggregator) & (summary["n_datasets"] == 3)]
        joined = stratum.merge(
            global_scores,
            left_on="resolved_mature_id",
            right_on="feature_id",
            how="inner",
        )
        if len(joined) > 2:
            rho = spearmanr(joined["mean_observed_percentile"], joined["ensemble_rank"]).statistic
        else:
            rho = np.nan
        three_top = [
            name.lower()
            for name in stratum.nsmallest(20, "mean_observed_percentile")["resolved_mature_id"].tolist()
        ]
        rows.append(
            {
                "comparison": "three_dataset_mean_percentile_vs_58_ensemble",
                "aggregator": aggregator,
                "dataset": "all_three",
                "left_n": int(len(stratum)),
                "right_n": int(global_scores["feature_id"].nunique()),
                "top20_overlap": len(set(three_top) & set(recomputed)),
                "spearman_on_shared_top20": rho,
            }
        )
        for dataset, frame in within.items():
            top = frame[frame["aggregator"] == aggregator].nsmallest(20, "ensemble_rank")["feature_id"].tolist()
            rows.append(
                {
                    "comparison": "within_dataset_top20_vs_58_ensemble_top20",
                    "aggregator": aggregator,
                    "dataset": dataset,
                    "left_n": int(EXPECTED[dataset]),
                    "right_n": 58,
                    "top20_overlap": len(set(name.lower() for name in top) & set(recomputed)),
                    "spearman_on_shared_top20": np.nan,
                }
            )
    return pd.DataFrame(rows)


def compare_all_sets(all_sets: pd.DataFrame) -> pd.DataFrame:
    from scipy.stats import spearmanr

    original = json.loads(ORIGINAL_TOP20.read_text())
    rows = []
    for aggregator in AGGREGATORS:
        recomputed = json.loads((ENSEMBLE_58 / f"{aggregator}_top20.json").read_text())
        original_names = original.get(aggregator.upper(), [])
        full_58 = pd.read_csv(ENSEMBLE_58 / "ensemble_rankings.csv")
        full_58 = full_58[full_58["aggregator"] == aggregator][["feature_id", "ensemble_rank"]].rename(
            columns={"feature_id": "shared_id", "ensemble_rank": "rank_in_58"}
        )
        group = all_sets[all_sets["aggregator"] == aggregator].copy()
        top = [name.lower() for name in group.nsmallest(20, "ensemble_rank")["feature_id"].tolist()]
        group["shared_id"] = group["feature_id"].str.lower()
        full_58["shared_id"] = full_58["shared_id"].str.lower()
        shared = group.merge(full_58, on="shared_id", how="inner")
        if len(shared) > 2:
            rho = spearmanr(shared["ensemble_rank"], shared["rank_in_58"]).statistic
        else:
            rho = np.nan
        coverage = group.nsmallest(20, "ensemble_rank")["n_datasets"].value_counts().to_dict()
        rows.append(
            {
                "comparison": "all_sets_union_vs_58_ensemble_top20",
                "aggregator": aggregator,
                "dataset": "union_of_39_rankings",
                "left_n": int(group["feature_id"].nunique()),
                "right_n": 58,
                "top20_overlap": len(set(top) & set(recomputed)),
                "spearman_on_shared_features": rho,
                "top20_overlap_with_original_print": len(set(top) & set(original_names)),
                "top20_datasets_1": int(coverage.get(1, 0)),
                "top20_datasets_2": int(coverage.get(2, 0)),
                "top20_datasets_3": int(coverage.get(3, 0)),
            }
        )
    return pd.DataFrame(rows)


def complete_matrix_matches(weights: dict) -> bool:
    """On a dataset whose 13 lists contain the same features, union and intersection agree."""
    for aggregator in AGGREGATORS:
        intersection = run_aggregator(weights, aggregator, universe="intersection")
        union = run_aggregator(weights, aggregator, universe="union")
        if set(intersection) != set(union):
            return False
        for feature, score in intersection.items():
            if not np.isfinite(score) or not np.isfinite(union[feature]):
                return False
            if not np.isclose(score, union[feature], rtol=1e-8, atol=1e-10):
                return False
    return True


def validate(within: dict[str, pd.DataFrame], inventory: pd.DataFrame, synthetic: dict, all_sets: pd.DataFrame) -> dict:
    checks = dict(synthetic)
    checks["stage_a_feature_counts"] = {}
    checks["stage_a_counts_match_expected"] = True
    checks["all_scores_finite"] = True
    checks["direction_consistent"] = True
    for dataset, frame in within.items():
        counts = frame.groupby("aggregator")["feature_id"].nunique().to_dict()
        checks["stage_a_feature_counts"][dataset] = counts
        if any(count != EXPECTED[dataset] for count in counts.values()):
            checks["stage_a_counts_match_expected"] = False
        if not np.isfinite(frame["ensemble_score"]).all():
            checks["all_scores_finite"] = False
        for aggregator, group in frame.groupby("aggregator"):
            best = group.loc[group["ensemble_rank"].idxmin(), "ensemble_score"]
            if "rank" in aggregator and best != group["ensemble_score"].min():
                checks["direction_consistent"] = False
            if "weight" in aggregator and best != group["ensemble_score"].max():
                checks["direction_consistent"] = False
    measured = set(zip(inventory["dataset"], inventory["feature_id"]))
    retained = set()
    for frame in within.values():
        retained.update(zip(frame["dataset"], frame["feature_id"]))
    checks["every_measured_feature_retained"] = measured == retained
    checks["no_feature_from_another_dataset_inserted"] = retained <= measured
    original_ids = set(inventory["feature_id"])
    checks["all_sets_feature_counts"] = all_sets.groupby("aggregator")["feature_id"].nunique().to_dict()
    checks["all_sets_ranks_every_original_identifier"] = set(checks["all_sets_feature_counts"].values()) == {len(original_ids)} and set(all_sets["feature_id"]) == original_ids
    checks["all_sets_scores_finite"] = bool(np.isfinite(all_sets["ensemble_score"]).all())
    expected_rankings = inventory.groupby("feature_id")["dataset"].nunique() * len(FEATURE_RANKERS)
    observed = all_sets.drop_duplicates("feature_id").set_index("feature_id")["n_rankings"]
    checks["coverage_counts_match_inventory"] = bool(observed.sort_index().equals(expected_rankings.sort_index().astype(int)))
    checks["all_sets_direction_consistent"] = True
    for aggregator, group in all_sets.groupby("aggregator"):
        best = group.loc[group["ensemble_rank"].idxmin(), "ensemble_score"]
        if "rank" in aggregator and best != group["ensemble_score"].min():
            checks["all_sets_direction_consistent"] = False
        if "weight" in aggregator and best != group["ensemble_score"].max():
            checks["all_sets_direction_consistent"] = False
    return checks


def main() -> None:
    for name in ("tables", "within_dataset", "cross_dataset", "all_sets", "validation"):
        (OUT / name).mkdir(parents=True, exist_ok=True)
    synthetic = synthetic_complete_case_check()
    rankings = load_rankings()
    inventory, identity_pack = build_inventory(rankings)
    inventory.to_csv(OUT / "tables/feature_universe_inventory.csv", index=False)
    identity_pack["resolved_presence"].assign(
        datasets=lambda frame: frame["datasets"].map(lambda value: "|".join(value))
    ).to_csv(OUT / "tables/resolved_mature_coverage.csv", index=False)
    (OUT / "tables/universe_summary.json").write_text(json.dumps(identity_pack["summary"], indent=2))

    checksums = []
    within = {}
    pieces = []
    for dataset in DATASETS:
        weights = weights_for_dataset(rankings, dataset)
        frames = []
        for aggregator in AGGREGATORS:
            scores = run_aggregator(weights, aggregator)
            frame = scores_to_frame(scores, aggregator, dataset)
            frame.to_csv(OUT / "within_dataset" / f"{dataset}__{aggregator}.csv", index=False)
            frames.append(frame)
        within[dataset] = pd.concat(frames, ignore_index=True)
        pieces.append(within[dataset])
        for method in FEATURE_RANKERS:
            path = INDIVIDUAL / f"{dataset}__{method}.csv"
            checksums.append({"path": str(path.relative_to(ROOT)), "sha256": sha256_file(path)})
    pd.concat(pieces, ignore_index=True).to_csv(OUT / "within_dataset/all_within_dataset_ensembles.csv", index=False)
    long, summary = cross_dataset_tables(within, identity_pack["resolved_rows"])
    long.to_csv(OUT / "cross_dataset/dataset_level_resolved_ranks.csv", index=False)
    summary.to_csv(OUT / "cross_dataset/coverage_stratified_summary.csv", index=False)
    all_weights = {}
    for dataset in DATASETS:
        all_weights.update(weights_for_dataset(rankings, dataset))
    coverage = feature_coverage(all_weights)
    coverage.to_csv(OUT / "all_sets/feature_coverage.csv", index=False)
    checks_match = complete_matrix_matches(weights_for_dataset(rankings, "ROSMAP"))
    all_frames = []
    for aggregator in AGGREGATORS:
        scores = run_aggregator(all_weights, aggregator, universe="union")
        frame = scores_to_frame(scores, aggregator, "all_sets", coverage)
        frame.to_csv(OUT / "all_sets" / f"{aggregator}.csv", index=False)
        all_frames.append(frame)
    all_sets = pd.concat(all_frames, ignore_index=True)
    all_sets.to_csv(OUT / "all_sets/all_sets_ensembles.csv", index=False)
    comparison = pd.concat(
        [compare_with_58(summary, within), compare_all_sets(all_sets)],
        ignore_index=True,
        sort=False,
    )
    comparison.to_csv(OUT / "validation/comparison_with_58_ensemble.csv", index=False)
    checks = validate(within, inventory, synthetic, all_sets)
    checks["complete_matrix_union_matches_intersection"] = checks_match
    (OUT / "validation/validation_checks.json").write_text(json.dumps(checks, indent=2))
    (OUT / "validation/input_checksums.json").write_text(json.dumps(checksums, indent=2))
    if not all(
        [
            checks["combined_is_intersection"],
            checks["union_keeps_every_measured_feature"],
            checks["dataset_only_score_matches_its_own_lists"],
            checks["worst_rank_fill_changes_the_score"],
            checks["within_a_keeps_dataset_only_feature"],
            checks["unmeasured_feature_has_no_score"],
            checks["repeat_identical"],
            checks["stage_a_counts_match_expected"],
            checks["all_scores_finite"],
            checks["direction_consistent"],
            checks["every_measured_feature_retained"],
            checks["all_sets_ranks_every_original_identifier"],
            checks["all_sets_scores_finite"],
            checks["all_sets_direction_consistent"],
            checks["coverage_counts_match_inventory"],
            checks["complete_matrix_union_matches_intersection"],
        ]
    ):
        raise SystemExit(f"Validation failed: {json.dumps(checks, indent=2)}")
    print("FULL_UNIVERSE_DONE", json.dumps(identity_pack["summary"]))


if __name__ == "__main__":
    main()
