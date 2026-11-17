"""Matrices used only for the logFC tables Notebook 11 reads.

GSE218605: notebook 2.1 cells 21-22.
GSE184942 and GSE173955: notebook 2.1 cells 102-113.
ROSMAP mRNA FPKM join: notebook 2.3 cells 4-29.
Gene-symbol mapping of that FPKM table: notebook 3.3 cells 6-22.
The later overwrite in notebook 3.3 cell 39 is not applied, because the
differential-expression notebook successfully read the cell-22 schema.
"""
import gzip
import re
import pandas as pd

from data.word_map import WORD_MAP


def prepare_gse218605(excel_path, series_matrix_gz, count_out):
    expression = pd.read_excel(excel_path)
    expression.columns = expression.iloc[20, :]
    expression.index = expression.Mature_ID
    expression = expression.iloc[20:, :].T.iloc[17:, :].drop("Mature_ID", axis=1)
    expression_count = expression.iloc[:24, :]
    with gzip.open(series_matrix_gz, "rt", encoding="utf-8") as file:
        df = pd.read_csv(file, sep="\t", skiprows=29)
        df.index = df.iloc[:, 0]
        df = df.iloc[:, 1:].T
        clinical = df.iloc[:, [8]]
        clinical.columns = ["Diagnosis"]
        clinical.index = list(map(lambda x: re.findall(r"\[(.*?)\]", x)[-1], clinical.index))
        clinical["Diagnosis"] = clinical["Diagnosis"].apply(lambda x: WORD_MAP["Diagnosis"][x.split(": ")[-1]])
        d = dict(zip(df.index, df["!Sample_geo_accession"]))
        d2 = {re.findall(r"\[(.*?)\]", x)[-1]: d[x] for x in d.keys()}
        data_count = pd.concat([clinical, expression_count], axis=1)
        data_count.index = list(map(lambda x: f"GSE218605_{d2[x]}", data_count.index))
        data_count.index.rename("Sample ID", inplace=True)
    data_count.to_csv(count_out)
    return data_count


def prepare_gse184942(series_matrix_gz, counts_tsv_gz, out_csv):
    with gzip.open(series_matrix_gz, "rt") as file:
        df = pd.read_csv(file, sep="\t", skiprows=27)
        df.index = df.iloc[:, 0]
        df = df.iloc[:, 1:].T
        clinical = df.iloc[:, [0, 8]]
        clinical.set_index("!Sample_geo_accession", inplace=True)
        clinical.index.name = None
        clinical.columns = ["Diagnosis"]
        clinical["Diagnosis"] = clinical["Diagnosis"].apply(lambda x: WORD_MAP["Diagnosis"][x.split(": ")[-1]])
    expression = pd.read_csv(counts_tsv_gz, sep="\t")
    expression.set_index("GeneID", inplace=True)
    expression = expression.T
    return clinical, expression


def prepare_gse173955(series_matrix_gz, counts_tsv_gz):
    with gzip.open(series_matrix_gz, "rt") as file:
        df = pd.read_csv(file, sep="\t", skiprows=31)
        df.index = df.iloc[:, 0]
        df = df.iloc[:, 1:].T
        clinical = df.iloc[:, [0, 8, 9]]
        labels = clinical.index
        clinical.set_index("!Sample_geo_accession", inplace=True)
        clinical.index.name = None
        clinical.columns = ["Gender", "Age"]
        clinical.Gender = clinical.Gender.apply(lambda x: x.split(": ")[-1])
        clinical.Age = clinical.Age.apply(lambda x: x.split(": ")[-1]).astype(int)
        clinical["Diagnosis"] = list(map(lambda x: "Control" if "non-Alzheimer" in x else "AD", labels))
    expression = pd.read_csv(counts_tsv_gz, sep="\t")
    expression.set_index("GeneID", inplace=True)
    expression = expression.T
    return clinical, expression


def map_entrez_symbols(expression, querymany):
    """Notebook 2.1 cells 104-105 and 111-112. querymany is mygene.MyGeneInfo().querymany."""
    entrez_ids = expression.columns
    results = querymany(entrez_ids, scopes="entrezgene", fields="symbol,name", species="human")
    mapper = {int(entry["query"]): entry["symbol"] for entry in results if "symbol" in entry}
    expression = expression[[column for column in expression.columns if column in mapper]]
    expression.columns = [mapper[column] for column in expression.columns]
    return expression


def combine_and_save(clinical, expression, out_csv):
    data = pd.concat([clinical.loc[expression.index], expression], axis=1)
    data.index.rename("Sample ID", inplace=True)
    data.to_csv(out_csv)
    return data


def prepare_rosmap_mrna_fpkm(fpkm_tsv, biospecimen_csv, clinical_csv, out_csv):
    rnaseq_met = pd.read_csv(biospecimen_csv)
    gene_expression = pd.read_csv(fpkm_tsv, sep="\t")
    gene_expression = gene_expression.iloc[:, 1:]
    gene_expression.index = gene_expression.gene_id
    gene_expression = gene_expression.drop("gene_id", axis=1).T
    clinical_data = pd.read_csv(clinical_csv)
    specimen_to_individual = dict(zip(rnaseq_met.specimenID, rnaseq_met.individualID))
    gene_expression.index = list(map(lambda x: "_".join(x.split("_")[:-1]), gene_expression.index))
    mappable = [s for s in gene_expression.index if s in specimen_to_individual]
    gene_expression = gene_expression.loc[mappable]
    gene_expression.index = list(map(lambda x: specimen_to_individual[x], gene_expression.index))
    gene_expression.columns.name = None
    gene_expression.index.name = "Sample ID"
    gene_expression = gene_expression.drop_duplicates()
    gene_expression = gene_expression[gene_expression.index.notna()]
    clinical_data.index = clinical_data.individualID
    clinical_data = clinical_data.loc[gene_expression.index]
    word_map = {
        "Gender": {"0.0": "Female", "1.0": "Male"},
        "Diagnosis": {"1.0": "Control", "2.0": "MCI", "3.0": "MCI+", "4.0": "AD", "5.0": "AD+", "6.0": "Other"},
    }
    clinical_data = clinical_data[["msex", "race", "pmi", "braaksc", "cogdx"]]
    clinical_data.columns = ["Gender", "Race", "PMI", "Braak", "Diagnosis"]
    clinical_data["Gender"] = clinical_data["Gender"].apply(lambda x: word_map["Gender"][str(x)])
    clinical_data["Diagnosis"] = clinical_data["Diagnosis"].apply(lambda x: word_map["Diagnosis"][str(x)])
    clinical_data["Braak"] = clinical_data["Braak"].astype(int)
    gene_expression_data = pd.merge(clinical_data.reset_index(), gene_expression.reset_index())
    gene_expression_data = gene_expression_data.drop_duplicates()
    numeric_columns = [f for f in gene_expression_data.columns if gene_expression_data[f].dtype != "O"]
    categorical_columns = [f for f in gene_expression_data.columns if gene_expression_data[f].dtype == "O"]
    gene_expression_data = gene_expression_data.groupby("Sample ID").agg({
        **{col: (lambda x: x.mode()[0] if not x.mode().empty else x.iloc[0]) for col in categorical_columns},
        **{col: "mean" for col in numeric_columns},
    })
    ordered_columns = ["Sample ID", "Gender", "Race", "PMI", "Braak", "Diagnosis"] + list(gene_expression.columns[6:])
    gene_expression_data = gene_expression_data[ordered_columns]
    gene_expression_data.to_csv(out_csv, index=False)
    return gene_expression_data


def map_rosmap_ensembl_symbols(rosmap_mrna_csv, querymany, out_csv):
    """Notebook 3.3 cells 6-22. Does not apply the cell 39 transpose overwrite."""
    df = pd.read_csv(rosmap_mrna_csv)
    ensembl_ids = [i.split(".")[0] for i in df.columns[6:]]
    results = querymany(ensembl_ids, scopes="ensembl.gene", fields="symbol", species="human")
    gene_id_to_symbol = {result.get("query"): result.get("symbol", "N/A") for result in results}
    gene_symbols = [gene_id_to_symbol[gene_id] for gene_id in ensembl_ids]
    df.columns = ["Sample ID", "Gender", "Race", "PMI", "Braak", "Diagnosis"] + gene_symbols
    new_columns = [column for column in list(df.columns) if column != "N/A"]
    new_columns_unique = list(dict.fromkeys(new_columns))
    df = df[new_columns_unique]
    df = df.loc[:, ~df.columns.duplicated()]
    df = df.loc[:, (df.sum() != 0)]
    df.to_csv(out_csv, index=False)
    return df
