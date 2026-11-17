"""GSE120584 serum matrix. Notebook 2.1 cells 14 and 18-19.

The series matrix is read with skiprows=39. Expression columns are
iloc[:, 33:2596] and clinical columns are iloc[:, 8:12]. miRNA accessions
are then renamed with artifacts/mirBase_to_mirTARBase.json. The second
save is the file Notebook 11 reads.
"""
import gzip
import pandas as pd

from data.word_map import WORD_MAP
from modules.utils import load_json


def prepare_gse120584(series_matrix_gz: str, mirbase_map_json: str, out_csv: str) -> pd.DataFrame:
    mirbase_to_mirtarbase = load_json(mirbase_map_json)
    with gzip.open(series_matrix_gz, "rt") as file:
        df = pd.read_csv(file, sep="\t", skiprows=39)
        df.index = df.iloc[:, 0]
        df = df.iloc[:, 1:].T
        expression_data = df.iloc[:, 33:2596]
        clinical_data = df.iloc[:, 8:12]
        clinical_data.columns = ["Diagnosis", "Age", "Gender", "APOE4"]
        clinical_data["Age"] = clinical_data["Age"].apply(lambda x: x.split(": ")[-1]).astype(int)
        clinical_data["Gender"] = clinical_data["Gender"].apply(lambda x: WORD_MAP["Gender"][x.split(": ")[-1]])
        clinical_data["APOE4"] = clinical_data["APOE4"].apply(lambda x: x.split(": ")[-1]).astype(int)
        clinical_data["Diagnosis"] = clinical_data["Diagnosis"].apply(lambda x: WORD_MAP["Diagnosis"][x.split(": ")[-1]])
        data = pd.concat([clinical_data, expression_data], axis=1)
        data.index = df["!Sample_geo_accession"].apply(lambda x: f"GSE120584_{x}")
        data.index.rename("Sample ID", inplace=True)
    listed = [miRNA for miRNA in data.columns[4:] if miRNA in mirbase_to_mirtarbase]
    data = data[["Diagnosis", "Age", "Gender", "APOE4"] + listed]
    data.columns = ["Diagnosis", "Age", "Gender", "APOE4"] + [mirbase_to_mirtarbase[m] for m in listed]
    data.to_csv(out_csv)
    return data
