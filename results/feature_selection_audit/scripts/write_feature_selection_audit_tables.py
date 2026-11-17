#!/usr/bin/env python
"""Write the feature-selection results-audit tables.

This script does not fit models, does not read expression matrices, and does
not create rankings. It records the 13 method keys from the original call,
the absence of the 39 score dictionaries, and identifier-coverage counts
already published by the biological EDA.
"""

from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / "results" / "feature_selection_audit" / "tables"

DATASETS = [
    {
        "dataset": "ROSMAP",
        "expected_n_features": 309,
        "cohort_used_by_cell_14": "AD 209, Control 169",
        "resolved_mature": 95,
        "ambiguous": 8,
        "unresolved": 189,
        "non_human_or_non_mirna": 17,
    },
    {
        "dataset": "GSE120584",
        "expected_n_features": 2521,
        "cohort_used_by_cell_14": "AD 1021, Control 288",
        "resolved_mature": 2521,
        "ambiguous": 0,
        "unresolved": 0,
        "non_human_or_non_mirna": 0,
    },
    {
        "dataset": "GSE215789",
        "expected_n_features": 1137,
        "cohort_used_by_cell_14": "AD 23, Control 21",
        "resolved_mature": 1017,
        "ambiguous": 41,
        "unresolved": 79,
        "non_human_or_non_mirna": 0,
    },
]

METHODS = [
    {
        "method": "lasso",
        "display_name": "LASSO",
        "score_definition": "Absolute logistic-regression coefficient after an internal StandardScaler",
        "ranking_direction": "larger_is_more_important",
        "scores_comparable_across_methods": "no",
        "tie_behavior_in_source": "Ties are possible. No saved scores exist from which to count them.",
        "invalid_score_behavior_in_source": "The returned value is a float absolute coefficient.",
        "features_returned_by_source": "Every column of the ranking matrix",
    },
    {
        "method": "ridge",
        "display_name": "Ridge",
        "score_definition": "Absolute logistic-regression coefficient after an internal StandardScaler",
        "ranking_direction": "larger_is_more_important",
        "scores_comparable_across_methods": "no",
        "tie_behavior_in_source": "Ties are possible. No saved scores exist from which to count them.",
        "invalid_score_behavior_in_source": "The returned value is a float absolute coefficient.",
        "features_returned_by_source": "Every column of the ranking matrix",
    },
    {
        "method": "elasticnet",
        "display_name": "Elastic Net",
        "score_definition": "Absolute logistic-regression coefficient after an internal StandardScaler",
        "ranking_direction": "larger_is_more_important",
        "scores_comparable_across_methods": "no",
        "tie_behavior_in_source": "Ties are possible. No saved scores exist from which to count them.",
        "invalid_score_behavior_in_source": "The returned value is a float absolute coefficient.",
        "features_returned_by_source": "Every column of the ranking matrix",
    },
    {
        "method": "shap",
        "display_name": "SHAP",
        "score_definition": "Mean across samples of the absolute SHAP value",
        "ranking_direction": "larger_is_more_important",
        "scores_comparable_across_methods": "no",
        "tie_behavior_in_source": "Ties are possible. No saved scores exist from which to count them.",
        "invalid_score_behavior_in_source": "The returned value is a float mean absolute SHAP value.",
        "features_returned_by_source": "Every column of the ranking matrix",
    },
    {
        "method": "lime",
        "display_name": "LIME",
        "score_definition": "Absolute local LIME weight from the last shuffled sample that produced that parsed name",
        "ranking_direction": "larger_is_more_important",
        "scores_comparable_across_methods": "no",
        "tie_behavior_in_source": "Ties are possible. No saved scores exist from which to count them.",
        "invalid_score_behavior_in_source": "The returned value is a float absolute weight when a name is parsed.",
        "features_returned_by_source": "Parsed explanation names. Completeness of the matrix columns is not guaranteed by the source.",
    },
    {
        "method": "xgb_feature_importance",
        "display_name": "XGBoost feature importance",
        "score_definition": "feature_importances_ of the XGBoost model already fit for SHAP",
        "ranking_direction": "larger_is_more_important",
        "scores_comparable_across_methods": "no",
        "tie_behavior_in_source": "Ties are possible. No saved scores exist from which to count them.",
        "invalid_score_behavior_in_source": "The returned value is a float native importance.",
        "features_returned_by_source": "Every column of the ranking matrix",
    },
    {
        "method": "randomforest_feature_importance",
        "display_name": "Random Forest feature importance",
        "score_definition": "feature_importances_ of the Random Forest already fit for LIME",
        "ranking_direction": "larger_is_more_important",
        "scores_comparable_across_methods": "no",
        "tie_behavior_in_source": "Ties are possible. No saved scores exist from which to count them.",
        "invalid_score_behavior_in_source": "The returned value is a float native importance.",
        "features_returned_by_source": "Every column of the ranking matrix",
    },
    {
        "method": "rf_permutation_feature_importance",
        "display_name": "Random Forest permutation importance",
        "score_definition": "Sum of 20 permutation repeats on one stratified 20 percent holdout. Negative values are kept.",
        "ranking_direction": "larger_is_more_important",
        "scores_comparable_across_methods": "no",
        "tie_behavior_in_source": "Ties are possible. No saved scores exist from which to count them.",
        "invalid_score_behavior_in_source": "The returned value is a float sum and may be negative.",
        "features_returned_by_source": "Every column of the ranking matrix",
    },
    {
        "method": "xgb_permutation_feature_importance",
        "display_name": "XGBoost permutation importance",
        "score_definition": "Sum of 20 permutation repeats on one stratified 20 percent holdout. Negative values are kept.",
        "ranking_direction": "larger_is_more_important",
        "scores_comparable_across_methods": "no",
        "tie_behavior_in_source": "Ties are possible. No saved scores exist from which to count them.",
        "invalid_score_behavior_in_source": "The returned value is a float sum and may be negative.",
        "features_returned_by_source": "Every column of the ranking matrix",
    },
    {
        "method": "t_test",
        "display_name": "t-test",
        "score_definition": "Two-sample t-test p-value. The absolute t statistic is computed and discarded. No multiplicity correction.",
        "ranking_direction": "smaller_is_more_important",
        "scores_comparable_across_methods": "no",
        "tie_behavior_in_source": "Equal p-values would be ties. No saved scores exist from which to count them.",
        "invalid_score_behavior_in_source": "The stored score is a p-value.",
        "features_returned_by_source": "Every column of the ranking matrix",
    },
    {
        "method": "boruta",
        "display_name": "Boruta",
        "score_definition": "BorutaPy ranking_. Rank 1 is the selected class in the wrapper. The support mask is discarded.",
        "ranking_direction": "smaller_is_more_important",
        "scores_comparable_across_methods": "no",
        "tie_behavior_in_source": "Boruta ranking_ can assign the same rank to more than one miRNA. The number of ties was not saved.",
        "invalid_score_behavior_in_source": "The stored score is an integer rank.",
        "features_returned_by_source": "Every column of the ranking matrix",
    },
    {
        "method": "mannwhitneyu",
        "display_name": "Mann-Whitney U test",
        "score_definition": "Two-sided Mann-Whitney U p-value. The U statistic is discarded. No multiplicity correction.",
        "ranking_direction": "smaller_is_more_important",
        "scores_comparable_across_methods": "no",
        "tie_behavior_in_source": "Equal p-values would be ties. No saved scores exist from which to count them.",
        "invalid_score_behavior_in_source": "A miRNA with fewer than two values in a group is stored as NaN. The three ranking matrices have no missing values, so that branch is unused for these inputs.",
        "features_returned_by_source": "Every column of the ranking matrix",
    },
    {
        "method": "svm_rfe",
        "display_name": "SVM-RFE",
        "score_definition": "RFE ranking_ with n_features_to_select=20. The 20 selected miRNAs share rank 1.",
        "ranking_direction": "smaller_is_more_important",
        "scores_comparable_across_methods": "no",
        "tie_behavior_in_source": "Twenty miRNAs share rank 1 by construction. Further ties among later ranks are possible and were not saved.",
        "invalid_score_behavior_in_source": "The stored score is an integer RFE rank.",
        "features_returned_by_source": "Every column of the ranking matrix, including miRNAs beyond the 20 selected",
    },
]

ABSENT = "absent"
NOT_COMPUTED = "not_computed"


def main() -> None:
    inventory_rows = []
    for dataset in DATASETS:
        for method in METHODS:
            inventory_rows.append(
                {
                    "dataset": dataset["dataset"],
                    "method": method["method"],
                    "display_name": method["display_name"],
                    "expected_dictionary_key": f"{dataset['dataset']}:{method['method']}",
                    "expected_output_file": "results/rankings/individual_rankings.json",
                    "original_notebook_output": "Cell 14 printed the integer 39. The dictionaries were not written to the cell output.",
                    "output_present_in_this_repository": "no",
                    "n_ranked_features": "",
                    "expected_n_features_in_ranking_matrix": dataset["expected_n_features"],
                    "cohort_passed_to_cell_14": dataset["cohort_used_by_cell_14"],
                    "ranking_direction": method["ranking_direction"],
                    "score_definition": method["score_definition"],
                    "scores_comparable_across_methods": method["scores_comparable_across_methods"],
                    "ties_observed_in_saved_output": "not_observable",
                    "missing_or_invalid_scores_observed": "not_observable",
                    "all_expected_features_ranked": "not_observable",
                    "verification_status": ABSENT,
                    "notes": method["tie_behavior_in_source"],
                }
            )
    inventory = pd.DataFrame(inventory_rows)
    if len(inventory) != 39:
        raise SystemExit(f"Expected 39 inventory rows, got {len(inventory)}")

    method_summary = pd.DataFrame(METHODS)
    method_summary.insert(0, "n_datasets", 3)
    method_summary["saved_ranking_lists"] = 0
    method_summary["expected_ranking_lists"] = 3
    method_summary["score_scale"] = "method_specific"
    method_summary["common_scale_with_other_methods"] = "no"

    agreement_rows = []
    for dataset in DATASETS:
        for metric, k, tie_policy in [
            ("spearman_rank_correlation", "complete_ranking", "average_ranks_would_be_required"),
            ("kendall_rank_correlation", "complete_ranking", "tau-b_would_be_required"),
            ("top_k_overlap", "20", "svm_rfe_selection_size_only"),
            ("jaccard_similarity_of_top_k", "20", "svm_rfe_selection_size_only"),
        ]:
            agreement_rows.append(
                {
                    "dataset": dataset["dataset"],
                    "comparison": "within_dataset_method_agreement",
                    "metric": metric,
                    "k": k,
                    "tie_handling": tie_policy,
                    "n_method_pairs": "",
                    "value": "",
                    "status": NOT_COMPUTED,
                    "reason": "The 13 score dictionaries for this dataset were not saved. Rank correlation and top-k overlap were not calculated.",
                    "calculation_class": "not_a_original_result",
                }
            )
    agreement = pd.DataFrame(agreement_rows)

    coverage_source = "results/biological_eda/tables/mature_identity_summary.tsv and coverage_intersections.tsv"
    cross_rows = []
    for dataset in DATASETS:
        cross_rows.append(
            {
                "comparison": "feature_availability",
                "datasets": dataset["dataset"],
                "eligible_feature_count": dataset["expected_n_features"],
                "resolved_mature_count": dataset["resolved_mature"],
                "ambiguous_count": dataset["ambiguous"],
                "unresolved_count": dataset["unresolved"],
                "non_human_or_non_mirna_count": dataset["non_human_or_non_mirna"],
                "ranking_agreement": "",
                "status": "reported_from_existing_identity_audit",
                "source": coverage_source,
                "interpretation": "Feature availability and identifier resolution. This row is not a ranking comparison.",
            }
        )
    overlaps = [
        ("ROSMAP ∩ GSE120584", 94, 0.03727200634417129),
        ("ROSMAP ∩ GSE215789", 58, 0.05502846299810247),
        ("GSE120584 ∩ GSE215789", 1000, 0.39401103230890466),
        ("ROSMAP ∩ GSE120584 ∩ GSE215789", 58, ""),
    ]
    for label, count, jaccard in overlaps:
        cross_rows.append(
            {
                "comparison": "identifier_overlap_resolved_mature",
                "datasets": label,
                "eligible_feature_count": count,
                "resolved_mature_count": count,
                "ambiguous_count": "",
                "unresolved_count": "",
                "non_human_or_non_mirna_count": "",
                "ranking_agreement": "",
                "status": "reported_from_existing_identity_audit",
                "source": coverage_source,
                "interpretation": (
                    f"Resolved mature-identifier overlap. Jaccard={jaccard if jaccard != '' else 'not_applicable_for_three_way'}. "
                    "This count is eligible for a future ranking comparison. It is not itself ranking agreement."
                ),
            }
        )
    for label, _count, _jaccard in overlaps:
        cross_rows.append(
            {
                "comparison": "ranking_agreement",
                "datasets": label,
                "eligible_feature_count": "",
                "resolved_mature_count": "",
                "ambiguous_count": "",
                "unresolved_count": "",
                "non_human_or_non_mirna_count": "",
                "ranking_agreement": "",
                "status": "not_assessable",
                "source": "no saved individual ranking dictionaries",
                "interpretation": "Top-ranked miRNA recurrence was not calculated because the individual rankings were not saved.",
            }
        )
    cross = pd.DataFrame(cross_rows)

    excluded = pd.DataFrame(
        [
            {
                "path": "Alzheimers-Multi-Omics-Analysis/results/GSE215789_plasma_microRNA/feature_importances.json",
                "n_methods": 6,
                "methods": "shap, lime, t_test, randomforest_feature_importance, xgb_feature_importance, permutation_feature_importance",
                "n_features": 2743,
                "decision": "excluded",
                "reason": "Six methods, a single permutation key, and 2743 features. The original cell uses 13 methods and the ranking matrix has 1137 miRNAs.",
            },
            {
                "path": "Alzheimers-Multi-Omics-Analysis/results/GSE120584_serum_microRNA/feature_importances.json",
                "n_methods": 6,
                "methods": "shap, lime, t_test, randomforest_feature_importance, xgb_feature_importance, permutation_feature_importance",
                "n_features": 2521,
                "decision": "excluded",
                "reason": "Six methods and one permutation key. The original cell uses 13 methods, including separate Random Forest and XGBoost permutation keys.",
            },
            {
                "path": "Alzheimers-Multi-Omics-Analysis/results/ROSMAP_microRNA_exosome/feature_importances.json",
                "n_methods": 6,
                "methods": "shap, lime, t_test, randomforest_feature_importance, xgb_feature_importance, permutation_feature_importance",
                "n_features": 233,
                "decision": "excluded",
                "reason": "Six methods and 233 features. The original ROSMAP ranking matrix has 309 miRNAs.",
            },
            {
                "path": "Alzheimers-Multi-Omics-Analysis/results/ROSMAP_microRNA_test/feature_importances.json",
                "n_methods": 4,
                "methods": "t_test, randomforest_feature_importance, xgb_feature_importance, permutation_feature_importance",
                "n_features": 309,
                "decision": "excluded",
                "reason": "Four methods and one permutation key. The feature count matches ROSMAP, and the method inventory does not.",
            },
        ]
    )

    OUT.mkdir(parents=True, exist_ok=True)
    inventory.to_csv(OUT / "feature_selection_output_inventory.csv", index=False)
    method_summary.to_csv(OUT / "feature_selection_method_summary.csv", index=False)
    agreement.to_csv(OUT / "feature_selection_agreement_summary.csv", index=False)
    cross.to_csv(OUT / "feature_selection_cross_dataset_summary.csv", index=False)
    excluded.to_csv(OUT / "feature_selection_excluded_external_files.csv", index=False)
    print(f"inventory {len(inventory)}")
    print(f"methods {len(method_summary)}")
    print(f"agreement {len(agreement)}")
    print(f"cross {len(cross)}")


if __name__ == "__main__":
    main()
