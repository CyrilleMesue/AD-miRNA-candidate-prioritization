"""Run the 13 Notebook 11 rankers on each dataset.

Calls the original DifferentiatingFactorAnalysis class unchanged.
"""
import json
from pathlib import Path

from modules.differentialAnalysis.differential_analysis import DifferentiatingFactorAnalysis
from prioritization.original import FEATURE_RANKERS


def run_individual_rankers(datasets: dict, outdir: str) -> dict:
    out = Path(outdir)
    out.mkdir(parents=True, exist_ok=True)
    single_rankers_ranked_list = {}
    for dataset_name, data in datasets.items():
        analyzer = DifferentiatingFactorAnalysis(
            data.reset_index(),
            analyses=FEATURE_RANKERS,
            labels_column="Diagnosis",
            reference_class="Control",
            sample_id_column=data.index.name,
            outdir="",
            feature_type="microRNA",
            test_size=0.2,
            n_folds=5,
            scoring="F1",
            save_results=False,
            show_plots=False,
            verbose=False,
        )
        analyzer.run_all_analyses()
        for ranker, weights in analyzer.feature_weights.items():
            single_rankers_ranked_list[f"{dataset_name}:{ranker}"] = weights
    path = out / "individual_rankings.json"
    path.write_text(json.dumps(single_rankers_ranked_list))
    if len(single_rankers_ranked_list) != 39:
        raise RuntimeError(
            f"Expected 39 ranked lists (13 methods x 3 datasets), got {len(single_rankers_ranked_list)}"
        )
    return single_rankers_ranked_list
