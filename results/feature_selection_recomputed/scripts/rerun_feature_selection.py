#!/usr/bin/env python
"""New recomputation of the 13 individual rankers and 12 ensemble aggregators.

This is not a recovery of the missing original scores. It calls the
existing DifferentiatingFactorAnalysis and EnsembleFeatureRanker
implementations. Each method–dataset result is written as soon as that
method returns, so a later failure does not discard earlier results.
"""

from __future__ import annotations

import hashlib
import json
import os
import sys
import time
import traceback
import warnings
from pathlib import Path

# Headless plotting. The original functions still build figures.
# Agg avoids a display requirement and does not change the stored scores.
os.environ.setdefault("MPLBACKEND", "Agg")

ROOT = Path(__file__).resolve().parents[3]
OUT = Path(__file__).resolve().parents[1]
os.chdir(OUT)
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

import matplotlib

matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Boruta 0.3 still uses NumPy 1 aliases removed in NumPy 2.
# Restoring the aliases does not change the Boruta algorithm.
# See specification/ANALYSIS_SPECIFICATION.md.
if not hasattr(np, "int"):
    np.int = int
if not hasattr(np, "float"):
    np.float = float
if not hasattr(np, "bool"):
    np.bool = bool

from modules.differentialAnalysis.differential_analysis import DifferentiatingFactorAnalysis
from modules.featureSelection.ensemble_feature_selection import EnsembleFeatureRanker
from prioritization.original import (
    AGGREGATORS,
    DECREASING_IMPORTANCE_SUFFIXES,
    FEATURE_RANKERS,
    sort_dict_features,
)
from prioritization.matrices import load_ranking_datasets

INDIVIDUAL = OUT / "individual"
ENSEMBLE = OUT / "ensembles"
VALIDATION = OUT / "validation"
REFERENCE = OUT / "reference"
for path in (INDIVIDUAL, ENSEMBLE, VALIDATION, REFERENCE):
    path.mkdir(parents=True, exist_ok=True)

DATASETS = ("ROSMAP", "GSE120584", "GSE215789")
SMALLER_IS_BETTER = {"t_test", "boruta", "mannwhitneyu", "svm_rfe"}
# Original run_all_analyses order. Later methods can see numpy state,
# and two importance scores reuse models fitted by SHAP and LIME.
ORIGINAL_ORDER = [
    "shap",
    "lime",
    "t_test",
    "randomforest_feature_importance",
    "xgb_feature_importance",
    "rf_permutation_feature_importance",
    "xgb_permutation_feature_importance",
    "lasso",
    "ridge",
    "elasticnet",
    "boruta",
    "mannwhitneyu",
    "svm_rfe",
]
# Original analysis notebook (parent project, not part of this repository).
NOTEBOOK = Path(os.environ.get("ORIGINAL_PROJECT", "../Alzheimers-Multi-Omics-Analysis")) / (
    "notebooks/11.BIOMARKER SELECTION FOR DOWNSTREAM VALIDATION.ipynb"
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def result_paths(dataset: str, method: str) -> tuple[Path, Path]:
    stem = f"{dataset}__{method}"
    return INDIVIDUAL / f"{stem}.csv", INDIVIDUAL / f"{stem}.meta.json"


def is_complete(dataset: str, method: str) -> bool:
    csv_path, meta_path = result_paths(dataset, method)
    if not csv_path.is_file() or not meta_path.is_file():
        return False
    meta = json.loads(meta_path.read_text())
    return meta.get("status") == "complete" and meta.get("n_features", 0) > 0


def stable_ranks(scores: dict, larger_is_better: bool) -> dict:
    """Consecutive ranks, ties broken by insertion order. Same rule as EnsembleFeatureRanker.rank_dict."""
    ordered = sorted(scores.items(), key=lambda item: item[1], reverse=larger_is_better)
    return {key: rank + 1 for rank, (key, _) in enumerate(ordered)}


def tie_sizes(scores: dict) -> dict:
    counts: dict = {}
    for value in scores.values():
        counts[value] = counts.get(value, 0) + 1
    return {key: counts[value] for key, value in scores.items()}


def save_ranking(dataset: str, method: str, scores: dict, expected: list[str], meta: dict) -> None:
    larger = method not in SMALLER_IS_BETTER
    ranks = stable_ranks(scores, larger_is_better=larger)
    ties = tie_sizes(scores)
    rows = []
    for feature, score in scores.items():
        try:
            numeric = float(score)
            finite = bool(np.isfinite(numeric))
        except (TypeError, ValueError):
            numeric = np.nan
            finite = False
        rows.append(
            {
                "feature_id": feature,
                "score": numeric,
                "rank_stable": ranks[feature],
                "tie_group_size": ties[feature],
                "score_is_finite": finite,
            }
        )
    frame = pd.DataFrame(rows)
    csv_path, meta_path = result_paths(dataset, method)
    frame.to_csv(csv_path, index=False)
    expected_set = set(expected)
    got = list(scores)
    direction_ok = True
    if frame["score_is_finite"].all() and len(frame):
        best = frame.loc[frame["rank_stable"].idxmin(), "score"]
        if larger:
            direction_ok = bool(best == frame["score"].max())
        else:
            direction_ok = bool(best == frame["score"].min())
    meta.update(
        {
            "status": "complete",
            "n_features": int(len(frame)),
            "n_expected_features": int(len(expected)),
            "n_missing_features": int(len(expected_set - set(got))),
            "n_unexpected_features": int(len(set(got) - expected_set)),
            "n_duplicate_identifiers": int(len(got) - len(set(got))),
            "n_nonfinite_scores": int((~frame["score_is_finite"]).sum()),
            "n_tied_scores": int((frame["tie_group_size"] > 1).sum()),
            "ranking_direction": "larger_is_more_important" if larger else "smaller_is_more_important",
            "tie_handling": "stable_insertion_order_consecutive_ranks",
            "direction_check_passed": bool(direction_ok),
            "coverage_check_passed": bool(set(got) == expected_set and len(got) == len(set(got))),
        }
    )
    meta_path.write_text(json.dumps(meta, indent=2, default=str))
    print(
        f"METHOD_DONE {dataset} {method} n={meta['n_features']} "
        f"missing={meta['n_missing_features']} nonfinite={meta['n_nonfinite_scores']} "
        f"seconds={meta.get('runtime_seconds')}",
        flush=True,
    )


def save_failure(dataset: str, method: str, meta: dict) -> None:
    _, meta_path = result_paths(dataset, method)
    meta["status"] = "failed"
    meta_path.write_text(json.dumps(meta, indent=2, default=str))
    print(f"METHOD_FAILED {dataset} {method}: {meta.get('error_type')}: {meta.get('error_message')}", flush=True)


def call_method(analyzer, method: str) -> None:
    if method == "shap":
        analyzer.perform_shap_analysis()
    elif method == "lime":
        analyzer.perform_lime_analysis()
    elif method == "t_test":
        analyzer.perform_t_test()
    elif method == "randomforest_feature_importance":
        analyzer.perform_randomforest_feature_importance_analysis()
    elif method == "xgb_feature_importance":
        analyzer.perform_xgb_feature_importance_analysis()
    elif method == "rf_permutation_feature_importance":
        analyzer.perform_permutation_feature_importance_analysis(model_name="Random Forest")
    elif method == "xgb_permutation_feature_importance":
        analyzer.perform_permutation_feature_importance_analysis(model_name="XGBClassifier")
    elif method == "lasso":
        analyzer.perform_lasso_analysis()
    elif method == "ridge":
        analyzer.perform_ridge_analysis()
    elif method == "elasticnet":
        analyzer.perform_elasticnet_analysis()
    elif method == "boruta":
        analyzer.perform_boruta_analysis()
    elif method == "mannwhitneyu":
        analyzer.perform_mannwhitneyu_analysis()
    elif method == "svm_rfe":
        analyzer.perform_svm_rfe_analysis()
    else:
        raise KeyError(method)


def methods_required_this_process(dataset: str) -> list[str]:
    """Replay ancestors when a dependent still has to be computed in this process."""
    missing = {method for method in ORIGINAL_ORDER if not is_complete(dataset, method)}
    if not missing:
        return []
    required = set(missing)
    if required & {"lime", "randomforest_feature_importance", "xgb_feature_importance"}:
        required.add("shap")
    if "randomforest_feature_importance" in required:
        required.add("lime")
        required.add("shap")
    if "elasticnet" in required:
        required.update({"t_test", "lasso", "ridge"})
    elif "ridge" in required:
        required.update({"t_test", "lasso"})
    elif "lasso" in required:
        required.add("t_test")
    return [method for method in ORIGINAL_ORDER if method in required]


def run_dataset(dataset: str, frame: pd.DataFrame, checksums: dict) -> None:
    required = methods_required_this_process(dataset)
    if not required:
        print(f"DATASET_SKIP {dataset} all methods complete", flush=True)
        return
    print(f"DATASET_START {dataset} shape={frame.shape} methods={required}", flush=True)
    expected = [column for column in frame.columns if column != "Diagnosis"]
    analyzer = DifferentiatingFactorAnalysis(
        frame.reset_index(),
        analyses=FEATURE_RANKERS,
        labels_column="Diagnosis",
        reference_class="Control",
        sample_id_column=frame.index.name,
        outdir="",
        feature_type="microRNA",
        test_size=0.2,
        n_folds=5,
        scoring="F1",
        save_results=False,
        show_plots=False,
        verbose=False,
    )
    for method in ORIGINAL_ORDER:
        if method not in required:
            continue
        if method == "xgb_feature_importance" and analyzer.fitted_model_xgb is None:
            save_failure(
                dataset,
                method,
                {
                    "dataset": dataset,
                    "method": method,
                    "error_type": "BlockedByMissingModel",
                    "error_message": "The original XGBoost importance reads the SHAP model. SHAP did not leave a fitted model, so the seeded training-split fallback was not run.",
                    "input_checksums": checksums,
                },
            )
            continue
        if method == "randomforest_feature_importance" and analyzer.fitted_model_rf is None:
            save_failure(
                dataset,
                method,
                {
                    "dataset": dataset,
                    "method": method,
                    "error_type": "BlockedByMissingModel",
                    "error_message": "The original Random Forest importance reads the LIME forest. LIME did not leave a fitted model, so the n_estimators=100 fallback was not run.",
                    "input_checksums": checksums,
                },
            )
            continue
        meta = {
            "dataset": dataset,
            "method": method,
            "analysis_label": "new_recomputed_run",
            "not_a_original_score_recovery": True,
            "input_checksums": checksums,
            "n_samples": int(frame.shape[0]),
            "class_counts": frame["Diagnosis"].value_counts().to_dict(),
            "label_encoding_note": "Model-based methods use LabelEncoder, which sorts AD=0 and Control=1. Boruta, Mann-Whitney, and SVM-RFE receive the diagnosis strings.",
            "started_at_unix": time.time(),
        }
        started = time.perf_counter()
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            try:
                call_method(analyzer, method)
                plt.close("all")
                scores = analyzer.feature_weights[method]
                meta["runtime_seconds"] = round(time.perf_counter() - started, 3)
                meta["warnings"] = sorted({str(item.message) for item in caught})[:40]
                save_ranking(dataset, method, scores, expected, meta)
            except Exception as exc:
                plt.close("all")
                meta["runtime_seconds"] = round(time.perf_counter() - started, 3)
                meta["warnings"] = sorted({str(item.message) for item in caught})[:40]
                meta["error_type"] = type(exc).__name__
                meta["error_message"] = str(exc)
                meta["traceback"] = traceback.format_exc()
                save_failure(dataset, method, meta)
    print(f"DATASET_DONE {dataset}", flush=True)


def load_individual() -> dict:
    loaded = {}
    for dataset in DATASETS:
        for method in ORIGINAL_ORDER:
            csv_path, meta_path = result_paths(dataset, method)
            if not is_complete(dataset, method):
                continue
            frame = pd.read_csv(csv_path)
            loaded[f"{dataset}:{method}"] = dict(zip(frame["feature_id"], frame["score"]))
    return loaded


def original_top20() -> dict:
    destination = REFERENCE / "cell23_original_top20.json"
    if destination.is_file():
        return json.loads(destination.read_text())
    notebook = json.loads(NOTEBOOK.read_text())
    text = []
    for output in notebook["cells"][23].get("outputs", []):
        if "text" in output:
            value = output["text"]
            text.append("".join(value) if isinstance(value, list) else value)
    blob = "\n".join(text)
    import re

    headers = re.findall(r"\*{5,}([A-Z0-9._]+)\*{5,}", blob)
    lists = re.findall(r"Top 20 MicroRNAs selected: (\[[^\]]+\])", blob)
    parsed = {header: json.loads(body.replace("'", '"')) for header, body in zip(headers, lists)}
    destination.write_text(json.dumps(parsed, indent=2))
    return parsed


def run_ensembles(checksums: dict) -> None:
    weights = load_individual()
    present = sorted(weights)
    (ENSEMBLE / "individual_keys_used.json").write_text(json.dumps(present, indent=2))
    if len(present) != 39:
        print(f"ENSEMBLE_PARTIAL keys={len(present)}", flush=True)
    decreasing = [key for key in weights if key.split(":")[-1] in DECREASING_IMPORTANCE_SUFFIXES]
    ranker = EnsembleFeatureRanker(
        feature_weights=weights,
        feature_type="microRNA",
        outdir="",
        decreasing_importance_cases=decreasing,
        save_results=False,
        show_plots=False,
    )
    aggregated = {}
    for aggregator in AGGREGATORS:
        method, kind = aggregator.split("_")
        started = time.perf_counter()
        if kind == "rank":
            aggregated[aggregator] = ranker.Rank_by_Ranks(plot=False, method=method)
        else:
            aggregated[aggregator] = ranker.Rank_by_Weights(plot=False, method=method)
        print(f"ENSEMBLE_DONE {aggregator} n={len(aggregated[aggregator])} seconds={time.perf_counter()-started:.3f}", flush=True)
    rows = []
    for name, scores in aggregated.items():
        ascending = "rank" in name
        ordered = sort_dict_features(scores, ascending=ascending, return_with_values=True)
        for position, (feature, score) in enumerate(ordered, start=1):
            rows.append(
                {
                    "aggregator": name,
                    "feature_id": feature,
                    "ensemble_score": float(score),
                    "ensemble_rank": position,
                    "sort_ascending": ascending,
                }
            )
        top = [feature.lower() for feature, _score in ordered[:20]]
        (ENSEMBLE / f"{name}_top20.json").write_text(json.dumps(top, indent=2))
    pd.DataFrame(rows).to_csv(ENSEMBLE / "ensemble_rankings.csv", index=False)
    (ENSEMBLE / "ensemble_run_meta.json").write_text(
        json.dumps(
            {
                "analysis_label": "new_recomputed_run",
                "not_a_original_score_recovery": True,
                "n_individual_keys": len(present),
                "n_aggregators": len(aggregated),
                "decreasing_importance_cases": decreasing,
                "input_checksums": checksums,
                "note": "Features absent from any supplied individual dictionary are dropped by DataFrame.dropna inside the original ensemble functions.",
            },
            indent=2,
        )
    )


def compare_top20() -> None:
    original = original_top20()
    rows = []
    for aggregator, original_names in original.items():
        key = aggregator.lower()
        path = ENSEMBLE / f"{key}_top20.json"
        if not path.is_file():
            rows.append(
                {
                    "aggregator": key,
                    "original_n": len(original_names),
                    "recomputed_n": 0,
                    "overlap": 0,
                    "jaccard": np.nan,
                    "status": "recomputed_list_missing",
                }
            )
            continue
        recomputed = json.loads(path.read_text())
        shared = [name for name in recomputed if name in original_names]
        union = len(set(recomputed) | set(original_names))
        both = [name for name in shared]
        if len(both) > 1:
            original_rank = {name: i + 1 for i, name in enumerate(original_names)}
            recomputed_rank = {name: i + 1 for i, name in enumerate(recomputed)}
            from scipy.stats import spearmanr

            rho = spearmanr(
                [original_rank[name] for name in both],
                [recomputed_rank[name] for name in both],
            ).statistic
        else:
            rho = np.nan
        rows.append(
            {
                "aggregator": key,
                "original_n": len(original_names),
                "recomputed_n": len(recomputed),
                "overlap": len(shared),
                "jaccard": len(shared) / union if union else np.nan,
                "spearman_of_shared_top20_positions": rho,
                "original_top20": "|".join(original_names),
                "recomputed_top20": "|".join(recomputed),
                "status": "compared",
                "interpretation": "Aggregate name-list check only. It does not validate the missing original individual scores.",
            }
        )
    pd.DataFrame(rows).to_csv(VALIDATION / "ensemble_top20_comparison.csv", index=False)


def individual_agreement() -> None:
    from scipy.stats import kendalltau, spearmanr

    rows = []
    for dataset in DATASETS:
        tables = {}
        for method in ORIGINAL_ORDER:
            if not is_complete(dataset, method):
                continue
            frame = pd.read_csv(result_paths(dataset, method)[0]).set_index("feature_id")
            sign = -1.0 if method in SMALLER_IS_BETTER else 1.0
            tables[method] = sign * frame["score"]
        methods = list(tables)
        for i, left in enumerate(methods):
            for right in methods[i + 1 :]:
                joined = pd.concat([tables[left], tables[right]], axis=1, join="inner").dropna()
                joined.columns = ["left", "right"]
                if len(joined) < 3:
                    rho = tau = np.nan
                else:
                    rho = spearmanr(joined["left"], joined["right"]).statistic
                    tau = kendalltau(joined["left"], joined["right"]).statistic
                left_top = set(pd.read_csv(result_paths(dataset, left)[0]).nsmallest(20, "rank_stable")["feature_id"])
                right_top = set(pd.read_csv(result_paths(dataset, right)[0]).nsmallest(20, "rank_stable")["feature_id"])
                rows.append(
                    {
                        "dataset": dataset,
                        "method_a": left,
                        "method_b": right,
                        "n_shared_features": int(len(joined)),
                        "spearman_average_ties": rho,
                        "kendall_tau_b": tau,
                        "top20_overlap": int(len(left_top & right_top)),
                        "top20_jaccard": len(left_top & right_top) / len(left_top | right_top),
                        "top20_definition": "stable rank <= 20; k=20 is the original SVM-RFE selection size and ensemble display length",
                        "score_orientation": "smaller-is-better scores were multiplied by -1 before correlation",
                        "analysis_label": "new_recomputed_run",
                    }
                )
    pd.DataFrame(rows).to_csv(VALIDATION / "individual_method_agreement.csv", index=False)


def cross_dataset() -> None:
    from scipy.stats import spearmanr

    identity_path = ROOT / "results/biological_eda/supporting_data/feature_identity_audit.tsv"
    identity = pd.read_csv(identity_path, sep="\t")
    resolved = identity[identity["class"] == "resolved mature"].copy()
    mapping = {}
    for dataset, group in resolved.groupby("dataset"):
        counts = group["resolved_mature_id"].value_counts()
        unique_ids = set(counts[counts == 1].index)
        group = group[group["resolved_mature_id"].isin(unique_ids)]
        mapping[dataset] = dict(zip(group["feature_id"], group["resolved_mature_id"]))
    rows = []
    for method in ORIGINAL_ORDER:
        ranks = {}
        for dataset in DATASETS:
            if not is_complete(dataset, method):
                continue
            frame = pd.read_csv(result_paths(dataset, method)[0])
            translated = []
            for feature, rank in zip(frame["feature_id"], frame["rank_stable"]):
                mature = mapping.get(dataset, {}).get(feature)
                if mature:
                    translated.append((mature, int(rank)))
            ranks[dataset] = dict(translated)
        available = [dataset for dataset in DATASETS if dataset in ranks]
        for i, left in enumerate(available):
            for right in available[i + 1 :]:
                shared = sorted(set(ranks[left]) & set(ranks[right]))
                if len(shared) > 2:
                    rho = spearmanr([ranks[left][name] for name in shared], [ranks[right][name] for name in shared]).statistic
                else:
                    rho = np.nan
                left_top = {name for name, rank in ranks[left].items() if rank <= 20}
                right_top = {name for name, rank in ranks[right].items() if rank <= 20}
                rows.append(
                    {
                        "method": method,
                        "dataset_a": left,
                        "dataset_b": right,
                        "resolved_ids_in_a": len(ranks[left]),
                        "resolved_ids_in_b": len(ranks[right]),
                        "comparison_universe": len(shared),
                        "spearman_of_full_stable_ranks": rho,
                        "top20_resolved_overlap": len(left_top & right_top),
                        "top20_resolved_in_a": len(left_top),
                        "top20_resolved_in_b": len(right_top),
                        "identifier_rule": "resolved_mature class from feature_identity_audit.tsv; duplicate mature IDs within a dataset excluded",
                        "analysis_label": "new_recomputed_run",
                    }
                )
    pd.DataFrame(rows).to_csv(VALIDATION / "cross_dataset_recurrence.csv", index=False)


def input_checksums() -> dict:
    config_path = ROOT / "config/analysis_config.yaml"
    import yaml

    config = yaml.safe_load(config_path.read_text())
    checksums = {}
    for key, relative in config["processed"].items():
        path = ROOT / relative
        checksums[key] = {"path": str(path), "sha256": sha256_file(path), "bytes": path.stat().st_size}
    return checksums


def main() -> None:
    stage = sys.argv[1] if len(sys.argv) > 1 else "all"
    checksums = input_checksums()
    (OUT / "input_checksums.json").write_text(json.dumps(checksums, indent=2))
    if stage in {"all", "individual"}:
        import yaml

        config = yaml.safe_load((ROOT / "config/analysis_config.yaml").read_text())
        datasets = load_ranking_datasets({key: str(ROOT / value) for key, value in config["processed"].items()})
        # Notebook order.
        for name in ("ROSMAP", "GSE120584", "GSE215789"):
            run_dataset(name, datasets[name], checksums)
    if stage in {"all", "ensemble"}:
        original_top20()
        run_ensembles(checksums)
        compare_top20()
        individual_agreement()
        cross_dataset()
    print("RERUN_FINISHED", flush=True)


if __name__ == "__main__":
    main()
