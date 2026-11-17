"""ROSMAP NanoString miRNA matrix. Notebook 2.1 cells 77-97.

Writes the full cohort, including MCI and other cogdx labels.
Notebook 11 performs the AD/Control collapse itself.
"""
import pandas as pd

from data.word_map import WORD_MAP


def prepare_rosmap_mirna(gct: str, assay_metadata: str, clinical: str, biospecimen: str, out_csv: str) -> pd.DataFrame:
    expression = pd.read_csv(gct, sep="\t", skiprows=1)
    expression.columns = expression.iloc[0, :]
    expression = expression[1:]
    expression.index = expression.Genes
    expression.drop(["Genes", "Description"], axis=1, inplace=True)
    expression = expression.T

    metadata = pd.read_csv(assay_metadata)
    clinical_data = pd.read_csv(clinical)
    biospecimen_metadata = pd.read_csv(biospecimen)

    df = biospecimen_metadata.iloc[:, 0:2]
    df.dropna(inplace=True)
    specimen_to_individual = dict(zip(df.specimenID, df.individualID))
    df2 = metadata.iloc[:, 0:2]
    df2.dropna(inplace=True)
    sample_to_specimen = dict(zip(df2.mirna_id, df2.specimenID))

    mapped_samples = [s for s in expression.index if s in sample_to_specimen]
    mapped_specimens = [
        sample_to_specimen[s] for s in mapped_samples if sample_to_specimen[s] in specimen_to_individual
    ]
    mappable = [
        s for s in expression.index
        if s in mapped_samples and sample_to_specimen[s] in mapped_specimens
    ]
    expression = expression.loc[mappable]
    expression.index = [
        specimen_to_individual[sample_to_specimen[s]] for s in expression.index
    ]
    clinical_data.index = clinical_data.individualID
    clinical_data = clinical_data.loc[expression.index]
    clinical_data = clinical_data[["msex", "race", "pmi", "braaksc", "cogdx"]]
    clinical_data.columns = ["Gender", "Race", "PMI", "Braak", "Diagnosis"]
    clinical_data["Race"] = clinical_data["Race"].astype(int)
    clinical_data["Gender"] = clinical_data["Gender"].apply(lambda x: WORD_MAP["Gender"][str(x)])
    clinical_data["Diagnosis"] = clinical_data["Diagnosis"].apply(lambda x: WORD_MAP["Diagnosis"][str(x)])
    clinical_data["Braak"] = clinical_data["Braak"].astype(int)
    data = pd.merge(clinical_data.reset_index(), expression.reset_index())
    data.index = data["index"].apply(lambda x: f"ROSMAP_{x}")
    data.drop("index", axis=1, inplace=True)
    data.index.rename("Sample ID", inplace=True)
    data.to_csv(out_csv)
    return data
