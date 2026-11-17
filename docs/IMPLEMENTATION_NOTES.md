# Implementation notes

These are behaviors of the historical code. They were preserved.

## GSE120584 downsample is not used

Notebook 11 cell 12 draws 300 AD samples with `random_state=42` and builds
`balanced_df`. Cell 14 ranks `gse120584_df`, the full AD/Control matrix
(1021 AD, 288 Control). `src/prioritization/matrices.py` keeps the
downsample function and does not call it.

## `gene_db_cutoff=1000` is ignored

`ad_mirna_target_pipeline` accepts `gene_db_cutoff` (default 100).
Notebook 11 calls it with `gene_db_cutoff=1000`. The body never reads the
argument. CTD and GeneCards are sliced with `[:100]` on whatever row
order the filtered frames have.

## GeneCards score sort is not saved

Notebook 8.1 calls `sort_values(by="Score")` and does not assign the
result. The top 100 GeneCards genes are the first 100 Alzheimer rows in
the export file, not the 100 highest relevance scores.

## CTD genes are not sorted by inference score before the slice

Notebook 8.1 keeps CTD download order, forces `Disease` to `AD`, drops
missing rows, and concatenates AD, then breast cancer, then PSP. After
Notebook 11 filters to `Disease == "AD"`, the first 100 remaining rows
are the candidates.

## HMDD filter keeps microRNAs that are absent from HMDD

The pipeline keeps top-20 microRNAs whose `-3p`/`-5p`/base names are not
in the HMDD AD table after `Score > 1`. This is a novelty filter, not a
confirmation filter.

## Unvalidated pairs

miRTarBase rows whose reporter assay, Western blot, and qPCR columns sum
to 0 are kept. In the function this uses `iloc[:, 2:5]`, so it depends on
column order. The later notebook cells name the three columns explicitly.
Both are preserved in the corresponding functions.

## Stage 2 is reloaded from a CSV

Cell 44's `to_csv` is commented out. Cell 45 then reads
`data/stage2-selected-miRNA-and-targets.csv` and replaces the table just
computed. The extracted pipeline writes the computed table and does not
silently swap in that checkpoint. To match the executed session, pass
that historical CSV into `target_association_pass`.

## ROSMAP ranking labels and ROSMAP limma labels differ

Ranking collapses every diagnosis containing `"AD"` to AD, so AD (184)
and AD+ (25) become 209 AD. The limma preparation keeps only labels that
are already exactly `AD` or `Control`, so AD+ is excluded there. The
logFC join is therefore not computed on the same ROSMAP sample set as
the rankers.

## GSE120584 limma has no p-value filter

The `P.Value < 0.05` line is commented out for GSE120584 only. The other
differential tables keep p < 0.05. Commented `|logFC|` thresholds in
those load cells are not applied. The live threshold is `|logFC| > 0.1`
after the join.

## GSE218605 control definition for edgeR

`Prepare Data for Differential Analysis.ipynb` recodes diagnosis `HY` to
`Control` before the two-group filter. That is only in the differential
input, not in the ranking matrices.

## Notebook 3.3 overwrites the ROSMAP mRNA matrix

Cells 6-22 write a samples-by-genes table with a Sample ID column.
Cell 39 writes a genes-by-samples table to the same path. The
differential-expression notebook ran against the cell-22 schema. The
extractor follows cells 6-22 and does not apply cell 39.

## Notebook 2.3 cell 28 may not run as stored

After `groupby("Sample ID")`, Sample ID is the index, and the next cell
selects a column named Sample ID. The extracted function keeps that
selection. If it raises `KeyError`, that is the notebook text, not a new
failure mode introduced here.

## Chained miRTarBase indexing

Cell 48 filters with two successive `[]` operations rather than a single
boolean mask. That pattern is copied.

## Literature table is not a selection rule

The table has 14 rows and all nine prioritized microRNAs. It does not
choose three experimental candidates.

## Logger side effect

`src/modules/logger.py` creates a `logs/` directory on import. That file
was copied unchanged from the original package.

## Ranker package imports

`differential_analysis.py` imports SHAP, LIME, Boruta, limma, and Cox
code even when a given method is not in the 13-method list. Those
imports are required to load the class. Limma-as-a-ranker is not one of
the 13 methods.

## Rank normalization in `Rank_by_Ranks` (changed 2026-11-06)

This is the one historical behavior that was changed. Ranks were assigned
over each complete input list (up to 309, 2,521, or 1,137 miRNAs), the
table was cut to the 58 miRNAs shared by all 39 lists, and the ranks were
divided by 58. Normalized ranks therefore ran from 0.017 to 43.4, outside
the [0, 1] range the aggregators assume. Mean-rank aggregation (normal CDF
centered at 0.5, sd sqrt(1/(12m))) returned 1.0 for all 58 miRNAs, RRA
returned 1.0 for 57 of them, Stuart scores reached 1e56, and larger
datasets weighed more in every rank-based method.

`Rank_by_Ranks` now re-ranks each list within the retained features
(`rank(axis=0, method="first")`, order preserved) before dividing by 58,
so every list ranks the same 58 miRNAs 1..58. Weight-based methods and the
`union` universe are unchanged.

Effect on outputs: the top-20 pool went from 45 to 41 miRNAs and the
|logFC|-filtered shortlist (`results/candidates/prioritized_mirnas.txt`)
is 11 miRNAs. hsa-miR-509-3p is no longer selected. hsa-miR-423-3p,
hsa-miR-450b-5p, and hsa-miR-409-5p are. hsa-miR-324-3p, hsa-miR-574-3p,
and hsa-miR-769-3p are retained. Before the change, the same code and
data gave 12 miRNAs; the nine printed by the notebook come from the
reloaded stage-2 CSV (see "Stage 2 is reloaded from a CSV").

Maximum-weight aggregation still gives all 58 miRNAs the same score. That
is not a normalization error: in GSE215789 XGBoost permutation importance,
57 of the 58 shared miRNAs score exactly 0 and hsa-miR-1299 scores -0.89,
so min-max scaling sets 57 of them to 1, and every miRNA reaches 1 in at
least one list. Removing maximum weight does not change the shortlist.

## miRTarBase lookup for miRNAs without an arm designation (changed 2026-11-06)

This is the second historical behavior that was changed. At the
target-association step (`ad_mirna_target_pipeline` step 5 and
`target_association_pass`), each miRNA was looked up in miRTarBase with
`get_mirna_versions(miR)`, which returns only the `-3p` and `-5p` forms.
miRNAs named without an arm (hsa-mir-1275, hsa-mir-346, hsa-mir-1246,
hsa-mir-1299, hsa-mir-543) are stored in miRTarBase under that plain name,
so the lookup never found them and they were removed as if they had no
untested interaction with an AD-associated gene. They have 15-29 such
interactions each. The earlier target-retrieval step already queried the
plain name (`include_base=True`), so the two steps disagreed.

The lookup now uses `mirtarbase_query_names`, which adds the plain name
only when the miRNA has no `-3p`/`-5p` designation. Arm-specific lookups
are unchanged. In the 12 ensemble lists on the 58 shared miRNAs, the
AD-target step now keeps all 25 miRNAs that pass the HMDD screen instead
of 20; miR-346 and miR-1246 then also pass |logFC| > 0.1 in GSE120584
(miR-1246 has p = 0.07 there, so it fails once p < 0.05 is required). In the
48-list prioritization (scripts/09_prioritize_48_lists.py) the fix also lets
un-armed miRNAs from the per-dataset lists reach the fold-change step
(e.g. miR-1538, miR-1471, miR-4279, miR-1260, miR-520e).
