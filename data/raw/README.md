# Raw inputs

Do not commit these files. Several are controlled-access (ROSMAP / AD Knowledge Portal) or large third-party downloads.

The notebooks record filenames, not Synapse accession numbers. No `syn########` identifier was found in the producer notebooks, so none is invented here.

## Ranking matrices

| Expected processed file | Raw files the producer reads | Producer |
|---|---|---|
| `data/processed/ROSMAP_brain_norm.csv` | `ROSMAP_arraymiRNA.gct`, `ROSMAP_assay_miRNAarray_nanostring_metadata.csv`, `ROSMAP_clinical.csv`, `ROSMAP_biospecimen_metadata.csv` | `scripts/01_prepare_rosmap.py` |
| `data/processed/GSE120584_serum_norm.csv` | `GSE120584_series_matrix.txt.gz` plus `miRNA.dat` mapped to `mirBase_to_mirTARBase.json` | `scripts/02_prepare_gse120584.py` |
| `data/processed/GSE215789_plasma_count.csv` | `GSE215789_miR.Counts_plasma.csv.gz`, `GSE215789_series_matrix.txt.gz`, and `data/external/GSE215789_sample_name_mapping.json` | `scripts/03_prepare_gse215789.py` |

GEO files are the series-matrix and supplementary files from the accession in the filename. Download the same file the original notebook opened. A fresh GEO export can change column positions, and the GSE120584 parser uses fixed column slices (`33:2596` and `8:12`).

ROSMAP files come from the AD Knowledge Portal. Keep them local. The repository does not contain them.

## miRNA name map

`downloads/microRNA/miRNA.dat` from miRBase (https://www.mirbase.org/download/miRNA.dat), parsed by `src/data/prepare_mirbase_map.py` (notebook 1.1).

## Annotation downloads used to rebuild the tables Notebook 11 reads

From notebook 8.1:

- `GeneCards-SearchResults-Alzheimers.csv`
- `GeneCards-SearchResults-BreastCancer.csv`
- `GeneCards-SearchResults-PSP.csv`
- `CTD_D000544_genes_20250721065005.csv` (AD genes, downloaded 2025-07-21)
- `CTD_D001941_genes_20250904053237.csv`
- `CTD_D013494_genes_20250904052250.csv`
- `alldata_v4.xlsx` (HMDD v4). This workbook is referenced by notebook 8.1 and was not present under `downloads/val_data/` at extraction time.

## Files with no producer in the original project

- `mirTarBase_cleaned.csv`
- `Predicted_Targets_Info.default_predictions.txt` (TargetScan default predictions table; only a reader was found)

Place copies at the paths in `config/analysis_config.yaml`.
