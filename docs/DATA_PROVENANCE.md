# Data provenance

Machine-readable copy: `metadata/data_provenance.csv`.

## Ranking matrices

### ROSMAP

Original files, all external and access-controlled: NanoString GCT
(`ROSMAP_arraymiRNA.gct`), assay metadata, `ROSMAP_clinical.csv`, and
biospecimen metadata. Notebook 2.1 maps specimen IDs to individual IDs,
maps `cogdx` with the diagnosis dictionary (`4.0` -> AD, `5.0` -> AD+,
`1.0` -> Control), and writes `ROSMAP_brain_norm.csv` for the mapped
cohort (525 samples, including MCI). Notebook 11 then drops sex, race,
PMI, and Braak, scales features, collapses labels containing "AD", and
keeps 209 AD and 169 Control. No all-zero microRNA column remains.
Ranking matrix: 378 x 309 microRNAs, plus the label column (378 x 310).

No normalization step is applied in notebook 2.1. The GCT is already a
processed NanoString matrix. Platform name is not written in Notebook 11;
the assay metadata filename contains `nanostring`.

### GSE120584

GEO series matrix, `skiprows=39`. Clinical columns are a fixed slice
(age, sex, APOE4, diagnosis). Expression columns are a fixed slice.
miRBase accessions are renamed with the notebook 1.1 map and unmapped
columns are dropped. Notebook 11 drops age, sex, and APOE4, scales, and
keeps 1021 AD and 288 Control. Matrix: 1309 x 2521 microRNAs.
The GEO record describes a serum miRNA array. The notebook stores the
experiment type "array" and does not store a GPL accession.

### GSE215789

Plasma count file plus the series matrix. Only rows whose tissue field
contains "Plasma" are kept. Sample titles are joined through the saved
JSON map. Notebook 11 drops sex and age, scales, keeps AD and Control,
and drops all-zero columns: 23 AD, 21 Control, 1137 microRNAs.
The study also has serum and MCI samples. Those samples are in the
processed file and are removed by Notebook 11. The GEO record describes
high-throughput sequencing. A GPL accession is not stored in the notebook.

## Annotation

HMDD, GeneCards, and CTD tables are built by notebook 8.1 from provider
exports. Dates are embedded in the CTD filenames (2025-07-21 for AD genes).
`mirTarBase_cleaned.csv` and the TargetScan predictions file have no
writer anywhere in the project. They are irreducible inputs until a
producer is found.

## Log fold change tables

`val_notebooks/Prepare Data for Differential Analysis.ipynb` reshapes
processed matrices to features-by-samples and relabels Control as CTRL.
`Run Differential Analysis.ipynb` calls the R scripts. GSE218605,
GSE173955, and GSE184942, plus ROSMAP mRNA, are used only at this step.
Their preparation code is in `src/data/prepare_supporting_expression.py`
and `src/differential/`.

ROSMAP mRNA starts from `ROSMAP_RNAseq_FPKM_gene.tsv` (notebook 2.3),
then Ensembl symbols via mygene (notebook 3.3, cells 6-22).
