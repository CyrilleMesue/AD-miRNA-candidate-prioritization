"""Build the annotation tables Notebook 11 reads.

Logic is from notebook 8.1. GeneCards sort_values() is not assigned
back to the frame, matching the notebook. CTD gene rows stay in
download order. Notebook 11 later keeps Disease == Alzheimer or AD
and, for genes, the first 100 rows of that filtered frame.
"""
import pandas as pd


def prepare_genecards(alzheimer_csv, breast_csv, psp_csv, out_csv):
    def _one(path, disease):
        df = pd.read_csv(path)
        df = df[["Gene Symbol", "Relevance score"]]
        df.rename(columns={"Gene Symbol": "Entity", "Relevance score": "Score"}, inplace=True)
        df.sort_values(by="Score", ascending=False)  # original no-op
        return pd.concat([pd.DataFrame({"Disease": [disease] * df.shape[0]}), df], axis=1)

    out = pd.concat([
        _one(alzheimer_csv, "Alzheimer"),
        _one(breast_csv, "BreastCancer"),
        _one(psp_csv, "LungCancer"),
    ], axis=0)
    out["Entity"] = out["Entity"].str.strip()
    out.to_csv(out_csv, index=False)
    return out


def prepare_ctd_genes(alzheimer_csv, breast_csv, psp_csv, out_csv):
    ad = pd.read_csv(alzheimer_csv)
    ad = ad[["Disease Name", "Gene Symbol", "Inference Score"]]
    ad.rename(columns={"Disease Name": "Disease", "Gene Symbol": "Entity", "Inference Score": "Score"}, inplace=True)
    ad["Disease"] = ad["Disease"].apply(lambda x: x.split(" ")[0])
    ad["Disease"] = ["AD"] * ad.shape[0]
    ad["Entity"] = ad["Entity"].str.strip()
    ad.dropna(inplace=True)

    br = pd.read_csv(breast_csv)
    br = br[["Disease Name", "Gene Symbol", "Inference Score"]]
    br.rename(columns={"Disease Name": "Disease", "Gene Symbol": "Entity", "Inference Score": "Score"}, inplace=True)
    br["Entity"] = br["Entity"].str.strip()
    br = br[br.Disease == "Breast Neoplasms"]
    br.Disease = ["BRCA"] * br.shape[0]
    br.dropna(inplace=True)

    psp = pd.read_csv(psp_csv)
    psp = psp[["Disease Name", "Gene Symbol", "Inference Score"]]
    psp.rename(columns={"Disease Name": "Disease", "Gene Symbol": "Entity", "Inference Score": "Score"}, inplace=True)
    psp["Entity"] = psp["Entity"].str.strip()
    psp = psp[psp.Disease == "Supranuclear Palsy, Progressive"]
    psp.dropna(inplace=True)
    psp.Disease = ["PSP"] * psp.shape[0]

    out = pd.concat([ad, br, psp], axis=0)
    out.to_csv(out_csv, index=False)
    return out


def prepare_hmdd(excel_path, out_csv):
    hmdd = pd.read_excel(excel_path)
    hmdd = hmdd[["disease", "miRNA"]]
    hmdd["association_count"] = hmdd.groupby(["disease", "miRNA"])["miRNA"].transform("count")
    hmdd = hmdd.drop_duplicates().sort_values("association_count", ascending=False)
    hmdd.rename(columns={"disease": "Disease", "miRNA": "Entity", "association_count": "Score"}, inplace=True)
    ad = hmdd[hmdd.Disease.isin(["Alzheimer Disease"])].copy()
    ad["Disease"] = ["AD"] * ad.shape[0]
    br = hmdd[hmdd.Disease.str.contains("Breast")].copy()
    br["Disease"] = ["BRCA"] * br.shape[0]
    psp = hmdd[hmdd.Disease.isin(["Supranuclear Palsy, Progressive"])].copy()
    psp["Disease"] = ["PSP"] * psp.shape[0]
    out = pd.concat([ad, br, psp], axis=0)
    out["Entity"] = out["Entity"].str.strip()
    out.to_csv(out_csv, index=False)
    return out
