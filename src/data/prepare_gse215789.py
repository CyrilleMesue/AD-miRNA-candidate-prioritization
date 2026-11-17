"""GSE215789 plasma counts. Notebook 2.1 cells 26-33.

Sample titles are joined to GEO accessions with the saved JSON map.
The HTML scrape that created that map is commented out in the notebook
and is not re-run here.
"""
import gzip
import pandas as pd

from data.word_map import WORD_MAP
from modules.utils import load_json


def prepare_gse215789(counts_gz: str, series_matrix_gz: str, sample_map_json: str, out_csv: str) -> pd.DataFrame:
    with gzip.open(counts_gz, "rt") as file:
        expression = pd.read_csv(file, sep=",")
        expression.index = expression.miRNA
        expression.drop("miRNA", axis=1, inplace=True)
        expression = expression.T
    sample_map = load_json(sample_map_json)
    with gzip.open(series_matrix_gz, "rt", encoding="utf-8") as file:
        df = pd.read_csv(file, sep="\t", skiprows=30)
        df.index = df.iloc[:, 0]
        df = df.iloc[:, 1:].T
        clinical = df.iloc[:, 7:11]
        clinical.columns = ["Tissue", "Gender", "Age", "Diagnosis"]
        clinical["Diagnosis"] = clinical["Diagnosis"].apply(lambda x: WORD_MAP["Diagnosis"][x.split(": ")[-1]])
        clinical["Gender"] = clinical["Gender"].apply(lambda x: WORD_MAP["Gender"][x.split(": ")[-1]])
        clinical["Age"] = clinical["Age"].apply(lambda x: x.split(": ")[-1] if "MCI" not in x else 0).astype("int")
        clinical["Tissue"] = clinical["Tissue"].apply(lambda x: x.split(": ")[-1])
        clinical_plasma = clinical[clinical.Tissue.str.contains("Plasma")].drop(columns=["Tissue"])
    expression.index = list(map(lambda x: sample_map[x], expression.index))
    data = pd.concat([clinical_plasma, expression], axis=1)
    data.index = list(map(lambda x: f"GSE215789_{x}", data.index))
    data.index.rename("Sample ID", inplace=True)
    data.to_csv(out_csv)
    return data
