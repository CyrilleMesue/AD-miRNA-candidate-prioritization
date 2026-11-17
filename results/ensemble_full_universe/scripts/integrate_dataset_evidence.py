#!/usr/bin/env python
"""Coverage-aware integration of the three dataset-level ensemble summaries.

The primary score is the mean of the available dataset-level median percentiles.
Unmeasured datasets stay blank. This rule is the constant PRIMARY_RULE.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from ensemble_common import (
    AGGREGATORS,
    CANDIDATES,
    DATASETS,
    ENSEMBLE_58,
    EXPECTED,
    FEATURE_RANKERS,
    HISTORICAL_TOP20,
    INDIVIDUAL,
    OUT,
    PRIMARY_RULE,
)


def dataset_tables() -> dict[str, pd.DataFrame]:
    tables = {}
    for dataset in DATASETS:
        path = OUT / "dataset_ensembles" / dataset / "dataset_ensemble_summary.csv"
        tables[dataset] = pd.read_csv(path)
    return tables


def resolved_long(tables: dict[str, pd.DataFrame]) -> pd.DataFrame:
    rows = []
    for dataset, frame in tables.items():
        kept = frame[frame["identifier_resolution_status"] == "resolved mature"].copy()
        if kept["resolved_mature_mirna_id"].duplicated().any():
            duplicated = kept.loc[kept["resolved_mature_mirna_id"].duplicated(keep=False), "resolved_mature_mirna_id"]
            raise SystemExit(f"Ambiguous duplicate mature IDs in {dataset}: {sorted(set(duplicated))}")
        for record in kept.itertuples(index=False):
            rows.append(
                {
                    "resolved_mature_mirna_id": record.resolved_mature_mirna_id,
                    "dataset": dataset,
                    "original_mirna_id": record.original_mirna_id,
                    "dataset_summary_rank": int(record.dataset_summary_rank),
                    "n_features_in_dataset": EXPECTED[dataset],
                    "median_rank_percentile": float(record.median_rank_percentile),
                    "median_ensemble_rank": float(record.median_ensemble_rank),
                    "best_ensemble_rank": int(record.best_ensemble_rank),
                    "worst_ensemble_rank": int(record.worst_ensemble_rank),
                    "top10_method_count": int(record.top10_method_count),
                    "top20_method_count": int(record.top20_method_count),
                    "top50_method_count": int(record.top50_method_count),
                }
            )
    return pd.DataFrame(rows)


def competition_rank(values: pd.Series) -> pd.Series:
    return values.rank(method="min", ascending=True).astype(int)


def integrate(long: pd.DataFrame) -> pd.DataFrame:
    records = []
    for mature, group in long.groupby("resolved_mature_mirna_id", sort=False):
        observed = group.set_index("dataset")
        percentiles = {dataset: np.nan for dataset in DATASETS}
        originals = {dataset: "" for dataset in DATASETS}
        for dataset, row in observed.iterrows():
            percentiles[dataset] = float(row["median_rank_percentile"])
            originals[dataset] = row["original_mirna_id"]
        available = [percentiles[dataset] for dataset in DATASETS if percentiles[dataset] == percentiles[dataset]]
        records.append(
            {
                "resolved_mature_mirna_id": mature,
                "n_datasets": len(available),
                "datasets": "|".join(dataset for dataset in DATASETS if percentiles[dataset] == percentiles[dataset]),
                "original_id_ROSMAP": originals["ROSMAP"],
                "original_id_GSE120584": originals["GSE120584"],
                "original_id_GSE215789": originals["GSE215789"],
                "percentile_ROSMAP": percentiles["ROSMAP"],
                "percentile_GSE120584": percentiles["GSE120584"],
                "percentile_GSE215789": percentiles["GSE215789"],
                "mean_available_percentile": float(np.mean(available)),
                "median_available_percentile": float(np.median(available)),
                "best_available_percentile": float(np.min(available)),
                "worst_available_percentile": float(np.max(available)),
                "primary_rule": PRIMARY_RULE,
                "unmeasured_left_blank": True,
            }
        )
    frame = pd.DataFrame(records)
    frame["primary_rank"] = competition_rank(frame["mean_available_percentile"])
    frame["sensitivity_median_rank"] = competition_rank(frame["median_available_percentile"])
    frame["sensitivity_best_rank"] = competition_rank(frame["best_available_percentile"])
    frame["sensitivity_worst_rank"] = competition_rank(frame["worst_available_percentile"])
    frame["stratum_rank"] = frame.groupby("n_datasets")["mean_available_percentile"].transform(
        lambda values: values.rank(method="min", ascending=True)
    ).astype(int)
    complete = frame["n_datasets"] == 3
    frame["complete_case_rank"] = np.nan
    frame.loc[complete, "complete_case_rank"] = competition_rank(frame.loc[complete, "mean_available_percentile"])
    return frame.sort_values(["primary_rank", "resolved_mature_mirna_id"]).reset_index(drop=True)


def coverage_summary(frame: pd.DataFrame, tables: dict[str, pd.DataFrame]) -> pd.DataFrame:
    identity_rows = []
    for dataset, table in tables.items():
        counts = table["identifier_resolution_status"].value_counts()
        identity_rows.append(
            {
                "dataset": dataset,
                "original_identifiers": int(table["original_mirna_id"].nunique()),
                "resolved_mature": int(counts.get("resolved mature", 0)),
                "ambiguous": int(counts.get("ambiguous", 0)),
                "unresolved": int(counts.get("unresolved", 0)),
                "non_mirna_or_control": int(counts.get("non-miRNA/control", 0)),
            }
        )
    summary = pd.DataFrame(identity_rows)
    extra = pd.DataFrame(
        [
            {
                "dataset": "union_resolved",
                "original_identifiers": int(sum(table["original_mirna_id"].nunique() for table in tables.values())),
                "resolved_mature": int(frame["resolved_mature_mirna_id"].nunique()),
                "ambiguous": int(summary["ambiguous"].sum()),
                "unresolved": int(summary["unresolved"].sum()),
                "non_mirna_or_control": int(summary["non_mirna_or_control"].sum()),
                "resolved_in_one_dataset": int((frame["n_datasets"] == 1).sum()),
                "resolved_in_two_datasets": int((frame["n_datasets"] == 2).sum()),
                "resolved_in_three_datasets": int((frame["n_datasets"] == 3).sum()),
            }
        ]
    )
    return pd.concat([summary, extra], ignore_index=True, sort=False)


def candidate_tables(tables: dict[str, pd.DataFrame], integrated: pd.DataFrame) -> None:
    folder = OUT / "candidates"
    folder.mkdir(parents=True, exist_ok=True)
    individual_rows = []
    ensemble_rows = []
    for dataset in DATASETS:
        summary = tables[dataset]
        for candidate in CANDIDATES:
            hits = summary[
                (summary["original_mirna_id"] == candidate) | (summary["resolved_mature_mirna_id"] == candidate)
            ]
            if hits.empty:
                individual_rows.append(
                    {
                        "candidate": candidate,
                        "dataset": dataset,
                        "present": False,
                        "original_mirna_id": "",
                        "identifier_resolution_status": "absent",
                    }
                )
                continue
            for _, hit in hits.iterrows():
                for method in FEATURE_RANKERS:
                    source = pd.read_csv(INDIVIDUAL / f"{dataset}__{method}.csv")
                    matched = source[source["feature_id"] == hit.original_mirna_id]
                    if matched.empty:
                        continue
                    row = matched.iloc[0]
                    n_features = len(source)
                    individual_rows.append(
                        {
                            "candidate": candidate,
                            "dataset": dataset,
                            "present": True,
                            "original_mirna_id": hit.original_mirna_id,
                            "resolved_mature_mirna_id": hit.resolved_mature_mirna_id,
                            "identifier_resolution_status": hit.identifier_resolution_status,
                            "method": method,
                            "score": float(row["score"]),
                            "rank_stable": int(row["rank_stable"]),
                            "tie_group_size": int(row["tie_group_size"]),
                            "n_features": n_features,
                            "rank_percentile": (float(row["rank_stable"]) - 1.0) / (n_features - 1.0),
                        }
                    )
                for aggregator in AGGREGATORS:
                    ensemble_rows.append(
                        {
                            "candidate": candidate,
                            "dataset": dataset,
                            "present": True,
                            "original_mirna_id": hit.original_mirna_id,
                            "resolved_mature_mirna_id": hit.resolved_mature_mirna_id,
                            "identifier_resolution_status": hit.identifier_resolution_status,
                            "aggregator": aggregator,
                            "ensemble_score": float(hit[f"{aggregator}__ensemble_score"]),
                            "ensemble_rank": int(hit[f"{aggregator}__ensemble_rank"]),
                            "rank_percentile": float(hit[f"{aggregator}__rank_percentile"]),
                            "n_features": EXPECTED[dataset],
                            "median_ensemble_rank": float(hit.median_ensemble_rank),
                            "best_ensemble_rank": int(hit.best_ensemble_rank),
                            "worst_ensemble_rank": int(hit.worst_ensemble_rank),
                            "median_rank_percentile": float(hit.median_rank_percentile),
                            "dataset_summary_rank": int(hit.dataset_summary_rank),
                            "top10_method_count": int(hit.top10_method_count),
                            "top20_method_count": int(hit.top20_method_count),
                            "top50_method_count": int(hit.top50_method_count),
                        }
                    )
    pd.DataFrame(individual_rows).to_csv(folder / "candidate_individual_method_ranks.csv", index=False)
    pd.DataFrame(ensemble_rows).to_csv(folder / "candidate_dataset_ensemble_ranks.csv", index=False)

    historical = pd.read_csv(ENSEMBLE_58 / "ensemble_rankings.csv")
    cross_rows = []
    for candidate in CANDIDATES:
        match = integrated[
            (integrated["resolved_mature_mirna_id"] == candidate)
            | (integrated["original_id_ROSMAP"] == candidate)
            | (integrated["original_id_GSE120584"] == candidate)
            | (integrated["original_id_GSE215789"] == candidate)
        ]
        if match.empty:
            cross_rows.append({"candidate": candidate, "present_in_resolved_union": False})
            continue
        row = match.iloc[0]
        historical_match = historical[historical["feature_id"].str.lower() == candidate.lower()]
        for aggregator in AGGREGATORS:
            prior = historical_match[historical_match["aggregator"] == aggregator]
            cross_rows.append(
                {
                    "candidate": candidate,
                    "present_in_resolved_union": True,
                    "resolved_mature_mirna_id": row["resolved_mature_mirna_id"],
                    "n_datasets": int(row["n_datasets"]),
                    "datasets": row["datasets"],
                    "coverage_stratum": int(row["n_datasets"]),
                    "mean_available_percentile": float(row["mean_available_percentile"]),
                    "primary_rank": int(row["primary_rank"]),
                    "stratum_rank": int(row["stratum_rank"]),
                    "sensitivity_median_rank": int(row["sensitivity_median_rank"]),
                    "sensitivity_best_rank": int(row["sensitivity_best_rank"]),
                    "sensitivity_worst_rank": int(row["sensitivity_worst_rank"]),
                    "complete_case_rank": float(row["complete_case_rank"]) if row["complete_case_rank"] == row["complete_case_rank"] else np.nan,
                    "percentile_ROSMAP": row["percentile_ROSMAP"],
                    "percentile_GSE120584": row["percentile_GSE120584"],
                    "percentile_GSE215789": row["percentile_GSE215789"],
                    "aggregator_in_58_analysis": aggregator,
                    "rank_in_58_ensemble": int(prior["ensemble_rank"].iloc[0]) if len(prior) else np.nan,
                    "n_features_in_58_ensemble": 58 if len(prior) else np.nan,
                    "percentile_in_58_ensemble": (
                        (int(prior["ensemble_rank"].iloc[0]) - 1) / 57.0 if len(prior) else np.nan
                    ),
                }
            )
    pd.DataFrame(cross_rows).to_csv(folder / "candidate_cross_dataset_summary.csv", index=False)

    printed = json.loads(HISTORICAL_TOP20.read_text())
    comparison_rows = []
    for candidate in CANDIDATES:
        for aggregator in AGGREGATORS:
            names = [name.lower() for name in printed.get(aggregator.upper(), [])]
            prior = historical[(historical["aggregator"] == aggregator) & (historical["feature_id"].str.lower() == candidate.lower())]
            comparison_rows.append(
                {
                    "candidate": candidate,
                    "aggregator": aggregator,
                    "in_historical_printed_top20": candidate.lower() in names,
                    "historical_printed_position": names.index(candidate.lower()) + 1 if candidate.lower() in names else np.nan,
                    "rank_in_recomputed_58": int(prior["ensemble_rank"].iloc[0]) if len(prior) else np.nan,
                    "reference_universe": "58-name complete-case ensemble",
                }
            )
    pd.DataFrame(comparison_rows).to_csv(folder / "candidate_historical_comparison.csv", index=False)


def compare_top20(tables: dict[str, pd.DataFrame], integrated: pd.DataFrame) -> None:
    historical = json.loads(HISTORICAL_TOP20.read_text())
    recomputed = pd.read_csv(ENSEMBLE_58 / "ensemble_rankings.csv")
    rows = []
    for dataset, table in tables.items():
        top = table.nsmallest(20, "dataset_summary_rank")
        resolved_top = [name.lower() for name in top["resolved_mature_mirna_id"].fillna("") if name]
        for aggregator in AGGREGATORS:
            printed = [name.lower() for name in historical.get(aggregator.upper(), [])]
            prior = recomputed[recomputed["aggregator"] == aggregator]
            shared = table[table["resolved_mature_mirna_id"].str.lower().isin(prior["feature_id"].str.lower())]
            if len(shared) > 2:
                prior_small = prior.copy()
                prior_small["key"] = prior_small["feature_id"].str.lower()
                shared = shared.copy()
                shared["key"] = shared["resolved_mature_mirna_id"].str.lower()
                joined = shared.merge(prior_small, on="key", how="inner")
                rho = float(spearmanr(joined["median_rank_percentile"], joined["ensemble_rank"]).statistic)
            else:
                rho = np.nan
            rows.append(
                {
                    "comparison": "dataset_median_percentile_top20_vs_historical_printed_top20",
                    "dataset": dataset,
                    "aggregator": aggregator,
                    "new_universe_n": EXPECTED[dataset],
                    "reference_universe_n": 58,
                    "top20_overlap": len(set(resolved_top) & set(printed)),
                    "spearman_of_shared_resolved_ids": rho,
                }
            )
    primary_top = [
        name.lower() for name in integrated.nsmallest(20, "primary_rank")["resolved_mature_mirna_id"]
    ]
    for aggregator in AGGREGATORS:
        printed = [name.lower() for name in historical.get(aggregator.upper(), [])]
        prior = recomputed[recomputed["aggregator"] == aggregator].copy()
        prior["key"] = prior["feature_id"].str.lower()
        complete = integrated[integrated["n_datasets"] == 3].copy()
        complete["key"] = complete["resolved_mature_mirna_id"].str.lower()
        joined = complete.merge(prior, on="key", how="inner")
        rho = float(spearmanr(joined["mean_available_percentile"], joined["ensemble_rank"]).statistic) if len(joined) > 2 else np.nan
        rows.append(
            {
                "comparison": "primary_top20_vs_historical_printed_top20",
                "dataset": "coverage_aware_union",
                "aggregator": aggregator,
                "new_universe_n": int(integrated.shape[0]),
                "reference_universe_n": 58,
                "top20_overlap": len(set(primary_top) & set(printed)),
                "spearman_of_shared_resolved_ids": rho,
            }
        )
    pd.DataFrame(rows).to_csv(OUT / "validation" / "historical_comparison.csv", index=False)


def main() -> None:
    if PRIMARY_RULE != "mean_of_available_dataset_median_percentiles":
        raise SystemExit("Primary rule was changed after the specification was frozen")
    tables = dataset_tables()
    long = resolved_long(tables)
    integrated = integrate(long)
    folder = OUT / "cross_dataset"
    folder.mkdir(parents=True, exist_ok=True)
    integrated.to_csv(folder / "primary_integration.csv", index=False)
    integrated[
        [
            "resolved_mature_mirna_id",
            "n_datasets",
            "datasets",
            "mean_available_percentile",
            "primary_rank",
            "stratum_rank",
            "percentile_ROSMAP",
            "percentile_GSE120584",
            "percentile_GSE215789",
        ]
    ].to_csv(folder / "coverage_stratified_rankings.csv", index=False)
    integrated[
        [
            "resolved_mature_mirna_id",
            "n_datasets",
            "mean_available_percentile",
            "primary_rank",
            "median_available_percentile",
            "sensitivity_median_rank",
            "best_available_percentile",
            "sensitivity_best_rank",
            "worst_available_percentile",
            "sensitivity_worst_rank",
            "stratum_rank",
            "complete_case_rank",
        ]
    ].to_csv(folder / "sensitivity_rankings.csv", index=False)
    long.to_csv(folder / "dataset_level_resolved_evidence.csv", index=False)
    coverage_summary(integrated, tables).to_csv(OUT / "validation" / "identifier_coverage_summary.csv", index=False)
    candidate_tables(tables, integrated)
    compare_top20(tables, integrated)
    counts = integrated["n_datasets"].value_counts().sort_index().to_dict()
    print("INTEGRATION_DONE", PRIMARY_RULE, json.dumps({str(key): int(value) for key, value in counts.items()}))


if __name__ == "__main__":
    main()
