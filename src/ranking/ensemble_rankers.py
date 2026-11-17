"""12 ensemble aggregators from Notebook 11 cell 16."""
import json
from pathlib import Path

from modules.featureSelection.ensemble_feature_selection import EnsembleFeatureRanker
from prioritization.original import (
    AGGREGATORS,
    DECREASING_IMPORTANCE_SUFFIXES,
)


def run_ensemble_rankers(single_rankers_ranked_list: dict, outdir: str) -> dict:
    decreasing = [
        ranker for ranker in single_rankers_ranked_list
        if ranker.split(":")[-1] in DECREASING_IMPORTANCE_SUFFIXES
    ]
    ensemble_aggregator = EnsembleFeatureRanker(
        feature_weights=single_rankers_ranked_list,
        feature_type="microRNA",
        outdir="",
        decreasing_importance_cases=decreasing,
        save_results=False,
        show_plots=False,
    )
    aggregator_rankers_ranked_list = {}
    for aggregator in AGGREGATORS:
        aggregation_method, aggregation_type = aggregator.split("_")
        if aggregation_type == "rank":
            aggregator_rankers_ranked_list[aggregator] = ensemble_aggregator.Rank_by_Ranks(
                plot=False, method=aggregation_method
            )
        else:
            aggregator_rankers_ranked_list[aggregator] = ensemble_aggregator.Rank_by_Weights(
                plot=False, method=aggregation_method
            )
    out = Path(outdir)
    out.mkdir(parents=True, exist_ok=True)
    # dict values may contain numpy scalars
    serializable = {
        name: {str(k): float(v) for k, v in ranks.items()}
        for name, ranks in aggregator_rankers_ranked_list.items()
    }
    (out / "ensemble_rankings.json").write_text(json.dumps(serializable))
    if len(aggregator_rankers_ranked_list) != 12:
        raise RuntimeError(f"Expected 12 ensemble rankings, got {len(aggregator_rankers_ranked_list)}")
    return aggregator_rankers_ranked_list
