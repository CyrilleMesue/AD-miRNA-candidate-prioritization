"""Load the three matrices exactly as Notebook 11 cells 9-11 do.

The GSE120584 downsample in cell 12 is intentionally not applied.
Cell 14 ranks `gse120584_df`, not `balanced_df`.
"""
import pandas as pd

from prioritization.original import scale_dataframe


def _read(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    if "Sample ID" not in df.columns:
        raise ValueError(f"{path} has no 'Sample ID' column")
    df = df.set_index("Sample ID")
    return df


def prepare_rosmap_for_ranking(path: str) -> pd.DataFrame:
    rosmap_df = _read(path)
    rosmap_df = rosmap_df.drop(columns=["Gender", "Race", "PMI", "Braak"])
    rosmap_df = scale_dataframe(rosmap_df, label_col="Diagnosis")
    rosmap_df["Diagnosis"] = rosmap_df["Diagnosis"].apply(lambda x: "AD" if "AD" in x else x)
    rosmap_df = rosmap_df[rosmap_df["Diagnosis"].isin(["AD", "Control"])]
    rosmap_df = rosmap_df.loc[:, (rosmap_df != 0).any(axis=0)]
    return rosmap_df


def prepare_gse215789_for_ranking(path: str) -> pd.DataFrame:
    gse215789_df = _read(path)
    gse215789_df = gse215789_df.drop(columns=["Gender", "Age"])
    gse215789_df = scale_dataframe(gse215789_df, label_col="Diagnosis")
    gse215789_df = gse215789_df[gse215789_df["Diagnosis"].isin(["AD", "Control"])]
    gse215789_df = gse215789_df.loc[:, (gse215789_df != 0).any(axis=0)]
    return gse215789_df


def prepare_gse120584_for_ranking(path: str) -> pd.DataFrame:
    gse120584_df = _read(path)
    gse120584_df = gse120584_df.drop(columns=["Gender", "Age", "APOE4"])
    gse120584_df = scale_dataframe(gse120584_df, label_col="Diagnosis")
    gse120584_df = gse120584_df[gse120584_df["Diagnosis"].isin(["AD", "Control"])]
    gse120584_df = gse120584_df.loc[:, (gse120584_df != 0).any(axis=0)]
    return gse120584_df


def downsample_gse120584_unused(gse120584_df: pd.DataFrame) -> pd.DataFrame:
    """Notebook 11 cell 12. Not used by the rankers. Preserved for provenance."""
    ad_df = gse120584_df[gse120584_df["Diagnosis"] == "AD"]
    control_df = gse120584_df[gse120584_df["Diagnosis"] == "Control"]
    ad_downsampled = ad_df.sample(n=300, random_state=42)
    balanced_df = pd.concat([control_df, ad_downsampled])
    balanced_df = balanced_df.sample(frac=1, random_state=42).reset_index(drop=True)
    return balanced_df


def load_ranking_datasets(paths: dict) -> dict:
    return {
        "ROSMAP": prepare_rosmap_for_ranking(paths["rosmap"]),
        "GSE120584": prepare_gse120584_for_ranking(paths["gse120584"]),
        "GSE215789": prepare_gse215789_for_ranking(paths["gse215789"]),
    }
