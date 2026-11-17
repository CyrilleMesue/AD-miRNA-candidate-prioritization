#!/usr/bin/env python
"""Run the Notebook 11 workflow in order.

python scripts/run_pipeline.py --config config/analysis_config.yaml
python scripts/run_pipeline.py --config config/analysis_config.yaml --stage matrices
"""
import argparse
import json
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from prioritization.matrices import load_ranking_datasets
from prioritization.candidate_selection import (
    attach_logfc,
    export_workbook,
    load_annotation_tables,
    load_differential_tables,
    run_stage1,
    target_association_pass,
)


def resolve(root: Path, value: str) -> str:
    path = Path(value)
    if not path.is_absolute():
        path = root / path
    return str(path)


def require(path: str, kind: str) -> None:
    if not Path(path).is_file():
        raise SystemExit(
            f"Missing {kind}: {path}\n"
            "Place the file at that path or change config/analysis_config.yaml. "
            "See data/raw/README.md and docs/DATA_PROVENANCE.md. "
            "The pipeline does not substitute example data."
        )


def load_config(path: str) -> dict:
    with open(path) as handle:
        return yaml.safe_load(handle)


def stage_matrices(cfg, root):
    paths = {k: resolve(root, v) for k, v in cfg["processed"].items()}
    for key, path in paths.items():
        require(path, f"processed matrix '{key}'")
    datasets = load_ranking_datasets(paths)
    summary = {}
    for name, df in datasets.items():
        counts = df["Diagnosis"].value_counts().to_dict()
        summary[name] = {
            "n_samples": int(df.shape[0]),
            "n_columns_including_label": int(df.shape[1]),
            "n_mirna": int(df.shape[1] - 1),
            "counts": {str(k): int(v) for k, v in counts.items()},
        }
        print(f"{name}: {summary[name]}")
    out = root / cfg["outputs"]["candidates"]
    out.mkdir(parents=True, exist_ok=True)
    (out / "matrix_summary.json").write_text(json.dumps(summary, indent=2))
    return datasets


def stage_rank(cfg, root, datasets):
    from ranking.individual_rankers import run_individual_rankers
    return run_individual_rankers(datasets, resolve(root, cfg["outputs"]["rankings"]))


def stage_ensemble(cfg, root, rankings):
    from ranking.ensemble_rankers import run_ensemble_rankers
    return run_ensemble_rankers(rankings, resolve(root, cfg["outputs"]["ensemble"]))


def stage_prioritize(cfg, root, ensembles):
    ann = {k: resolve(root, v) for k, v in cfg["annotation"].items()}
    de = {k: resolve(root, v) for k, v in cfg["differential"].items()}
    for key, path in {**ann, **de}.items():
        require(path, key)
    dbs = load_annotation_tables(ann)
    stage1, keep_ranks = run_stage1(ensembles, dbs)
    de_tables = load_differential_tables(de)
    stage2 = attach_logfc(stage1, keep_ranks, de_tables)
    stage2_path = root / cfg["outputs"]["candidates"] / "stage2_logfc_filtered.csv"
    stage2_path.parent.mkdir(parents=True, exist_ok=True)
    stage2.to_csv(stage2_path, index=False)
    selection = target_association_pass(stage2, dbs)
    pd_path = root / cfg["outputs"]["candidates"] / "prioritized_mirnas.txt"
    pd_path.write_text("\n".join(selection["stage2_mirnas"]) + "\n")
    selection["selected_microRNA_ranks"].to_csv(
        root / cfg["outputs"]["candidates"] / "selected_microrna_ranks.csv", index=False
    )
    selection["literature"].to_csv(
        root / cfg["outputs"]["candidates"] / "Literature_Review_Of_Selected.csv", index=False
    )
    export_workbook(selection, de_tables, resolve(root, cfg["outputs"]["workbook"]))
    print("stage2 microRNAs:", selection["stage2_mirnas"])
    print("association-pass microRNAs:", selection["selected_miRs"])
    return selection


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(ROOT / "config" / "analysis_config.yaml"))
    parser.add_argument(
        "--stage",
        choices=["all", "matrices", "rank", "ensemble", "prioritize"],
        default="all",
    )
    args = parser.parse_args()
    cfg = load_config(args.config)
    root = ROOT
    if args.stage == "matrices":
        stage_matrices(cfg, root)
        return
    if args.stage == "all":
        datasets = stage_matrices(cfg, root)
        rankings = stage_rank(cfg, root, datasets)
        ensembles = stage_ensemble(cfg, root, rankings)
        stage_prioritize(cfg, root, ensembles)
        return
    raise SystemExit(
        "Stages rank, ensemble, and prioritize depend on earlier outputs. "
        "Use --stage all, or --stage matrices to check the three input matrices."
    )


if __name__ == "__main__":
    main()
