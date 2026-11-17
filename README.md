# AD miRNA candidate prioritization

Code for the computational part of the MSc thesis *Prioritization of Alzheimer's
Disease-Associated microRNAs by Ensemble Feature Ranking and Evidence-Based
Filtering, and Their Investigation in Amyloid-β1-42-Treated SH-SY5Y Cells*
(Istanbul Technical University, 2026).

The feature-selection and ensemble-ranking framework applied here was developed
and benchmarked in our published paper:

> Njume, C. M., Petracci, I., Bellini, S., Goljanek-Whysall, K., Quinlan, L. R.,
> Fiszer, A., Borroni, B., Ghidoni, R., Kumbasar, A., & Cakmak, A. (2026).
> When complexity does not pay: Benchmarking deep learning and ensemble methods
> for biomarker discovery. *Briefings in Bioinformatics*, 27(3), bbag211.
> https://doi.org/10.1093/bib/bbag211

## What the code does

1. Prepares three AD miRNA datasets: ROSMAP dorsolateral prefrontal cortex
   (NanoString), GSE120584 serum (microarray), and GSE215789 plasma (small-RNA
   sequencing).
2. Ranks miRNAs in each dataset with 13 feature-selection methods (LASSO, ridge,
   elastic net, SHAP, LIME, XGBoost and random-forest importance, RF and XGBoost
   permutation importance, t-test, Boruta, Mann–Whitney U, SVM-RFE), giving 39
   standalone rankings.
3. Aggregates the rankings with 12 ensemble methods (mean, minimum, median and
   geometric-mean rank, RRA, Stuart, MRA; mean, median, geometric-mean and
   maximum weight, Threshold Algorithm). Maximum-weight aggregation is excluded
   from prioritization because it assigns the same score to all shared miRNAs.
4. Filters the top-20 miRNAs of each ensemble ranking for limited prior AD
   evidence (HMDD), experimentally supported interactions with AD-associated
   genes (TargetScan, miRTarBase, CTD, GeneCards), and differential expression
   (limma / edgeR) in at least one dataset.

The original procedure used the 11 ensemble rankings of the 58 miRNAs shared by
all 39 standalone rankings. The revised procedure adds ensemble rankings within
each dataset (44 top-20 lists in total).

## Repository layout

| Path | Contents |
|---|---|
| `src/modules/` | Feature-selection methods and ensemble aggregators (RRA, MRA, TA, Stuart, weight-based) |
| `src/ranking/`, `src/prioritization/` | Wrappers that run the rankers and the candidate filters |
| `src/data/`, `scripts/01-03_*.py` | Dataset preparation |
| `src/differential/` | limma and edgeR scripts (R) |
| `src/annotation/` | Builds the HMDD, CTD and GeneCards tables |
| `scripts/run_pipeline.py` | Original prioritization pipeline (ranking → ensembles → filters) |
| `scripts/09_prioritize_48_lists.py` | Revised prioritization over shared and within-dataset ensemble rankings |
| `results/**/scripts/` | Scripts that recompute the rankings and produce the thesis tables and figures |
| `docs/` | Pipeline order, data provenance, implementation notes |
| `third_party/` | Plotting helpers from the ITU Bioinformatics lab (see `third_party/NOTICE.txt`) |

## Data

**No data are included in this repository.** Place the input files under
`data/` as described in `data/raw/README.md`, `data/external/README.md` and
`docs/DATA_PROVENANCE.md`.

- GSE120584 and GSE215789 are public on the
  [Gene Expression Omnibus](https://www.ncbi.nlm.nih.gov/geo/).
- ROSMAP data are controlled-access through the
  [AD Knowledge Portal](https://adknowledgeportal.synapse.org/) and must be
  requested from the data provider. They cannot be redistributed.
- HMDD, miRTarBase, TargetScan, CTD and GeneCards tables are downloaded from the
  respective resources.

## Running

```bash
conda env create -f environment.yml   # or: pip install -r requirements.txt
conda activate ad-mirna-prioritization

# check that the prepared matrices are present
python scripts/run_pipeline.py --config config/analysis_config.yaml --stage matrices

# 13 standalone rankings per dataset and 12 ensemble rankings of the shared miRNAs
python results/feature_selection_recomputed/scripts/rerun_feature_selection.py
# 12 ensemble rankings within each dataset
python results/feature_selection_recomputed/scripts/run_per_dataset_ensembles.py
# revised prioritization over the 44 top-20 lists
python scripts/09_prioritize_48_lists.py
```

`scripts/run_pipeline.py --config config/analysis_config.yaml` runs the original
procedure end to end. Every script stops with an error when an input file is missing; none of them
generates stand-in data. SHAP, LIME, Boruta and SVM-RFE are slow on the GSE120584
matrix (2,521 features).

The thesis analyses used Python 3.13.9 with scikit-learn 1.7.2, XGBoost 2.1.4,
SHAP 0.52.0, LIME 0.2.0.1 and Boruta 0.3, and R 4.4.3 with edgeR 4.4.0 and limma.

## Citation

If you use the ranking framework, please cite the paper above.
