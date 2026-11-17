# Pipeline

```text
Synapse / GEO downloads
    -> notebook 2.1 (and notebook 1.1 for miRBase names)
    -> ROSMAP_brain_norm.csv
    -> GSE120584_serum_norm.csv
    -> GSE215789_plasma_count.csv
    -> Notebook 11 cells 9-11
         drop covariates
         MinMax scale features inside the cohort
         ROSMAP: any label containing "AD" becomes AD
         keep AD and Control
         drop all-zero feature columns
    -> 13 rankers x 3 datasets = 39 lists
    -> 12 ensemble aggregators = 12 lists
    -> top 20 microRNAs from each list
    -> HMDD absence filter, target prediction, top 100 CTD/GeneCards genes,
       unvalidated miRTarBase pairs
    -> join limma/edgeR logFC
    -> keep |logFC| > 0.1
    -> literature table (manual)
```

The limma/edgeR tables are not produced by the 13 rankers. They come from
`val_notebooks/Run Differential Analysis.ipynb`, which calls
`src/differential/run_limma.R` and `src/differential/run_edgeR.R` on
matrices prepared by `src/differential/prepare_for_de.py`.

## Rankers

lasso, ridge, elasticnet, shap, lime, xgb_feature_importance,
randomforest_feature_importance, rf_permutation_feature_importance,
xgb_permutation_feature_importance, t_test, boruta, mannwhitneyu, svm_rfe.

Parameters passed from Notebook 11: `reference_class=Control`,
`test_size=0.2`, `n_folds=5`, `scoring=F1`, `save_results=False`.
Seeds inside the ranker classes are whatever those modules already set
(often 42). They were not changed.

## Ensembles

Rank methods: mean, min, median, rra, geom.mean, stuart, mra.
Weight methods: mean, max, median, geom.mean, ta.

There is no k-of-12 recurrence rule and no cross-dataset recurrence rule
in the notebook.

## Entry point

`scripts/run_pipeline.py --config config/analysis_config.yaml`

`--stage matrices` only checks the three ranking matrices.
`--stage all` continues through ranking. SHAP, LIME, Boruta, and SVM-RFE
on these matrices are expensive. The extraction did not re-run them.
