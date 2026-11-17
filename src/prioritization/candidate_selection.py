"""Candidate filtering from Notebook 11 cells 18-23 and 28-67.

Thresholds, top-20 cutoff, HMDD score filter, top-100 disease-gene
slice, unvalidated-pair rule, and |logFC| > 0.1 filter are unchanged.
"""
from pathlib import Path

import pandas as pd
from collections import defaultdict

from prioritization.original import (
    ad_mirna_target_pipeline,
    get_mirna_versions,
    mirtarbase_query_names,
    literature_review_table,
    sort_dict_features,
)


def load_annotation_tables(paths: dict) -> dict:
    hmdd = pd.read_csv(paths["hmdd"])
    hmdd = hmdd[hmdd.Disease == "AD"]
    hmdd = hmdd[hmdd.Score > 1]

    mirtarbase = pd.read_csv(paths["mirtarbase"])
    mirtarbase["miRNA"] = mirtarbase["miRNA"].apply(lambda x: x.lower())

    targetscan = pd.read_csv(paths["targetscan"], sep="\t")
    targetscan["miR Family"] = targetscan["miR Family"].apply(lambda x: "hsa-" + x.lower())

    genecards = pd.read_csv(paths["genecards"])
    genecards = genecards[genecards.Disease == "Alzheimer"]

    ctd = pd.read_csv(paths["ctd"])
    ctd = ctd[ctd.Disease == "AD"]
    return {
        "hmdd": hmdd,
        "mirtarbase": mirtarbase,
        "targetscan": targetscan,
        "genecards": genecards,
        "ctd": ctd,
    }


def run_stage1(aggregator_rankers_ranked_list: dict, dbs: dict) -> tuple:
    """Top 20 from each ensemble list, then ad_mirna_target_pipeline.

    gene_db_cutoff=1000 is what Notebook 11 passes. The function body
    ignores that argument and slices the first 100 rows.
    """
    stage1 = {aggregator: {"miR": set(), "Gene": set()} for aggregator in aggregator_rankers_ranked_list}
    keep_ranks = {}
    for aggregator, ranked_list in aggregator_rankers_ranked_list.items():
        ascending = True if "rank" in aggregator else False
        ranked_features = sort_dict_features(ranked_list, ascending=ascending, return_with_values=False)
        df = pd.DataFrame({"MicroRNAs": ranked_features, "Rank": list(range(1, len(ranked_features) + 1))})
        df["MicroRNAs"] = df["MicroRNAs"].apply(lambda x: x.lower())
        keep_ranks[aggregator] = df
        microRNAs, genes = ad_mirna_target_pipeline(
            df,
            dbs["hmdd"],
            dbs["mirtarbase"],
            dbs["targetscan"],
            dbs["ctd"],
            dbs["genecards"],
            gene_db_cutoff=1000,
        )
        stage1[aggregator]["miR"] = stage1[aggregator]["miR"].union(microRNAs)
        stage1[aggregator]["Gene"] = stage1[aggregator]["Gene"].union(genes)
    return stage1, keep_ranks


def _load_de_table(path, id_col, p_col, apply_p):
    df = pd.read_csv(path)
    df.rename(columns={"Unnamed: 0": id_col}, inplace=True)
    if apply_p:
        df = df[df[p_col] < 0.05]
    df[id_col] = df[id_col].apply(lambda x: str(x).lower())
    return df


def load_differential_tables(paths: dict) -> dict:
    """Notebook 11 cells 28-39.

    GSE120584 p-value filter is commented out in the notebook and stays off.
    The other five tables keep p < 0.05. |logFC| thresholds in those cells
    are commented out and are not applied here.
    """
    gse120584 = pd.read_csv(paths["gse120584_limma"])
    gse120584.rename(columns={"Unnamed: 0": "miRNA"}, inplace=True)
    gse120584["miRNA"] = gse120584["miRNA"].apply(lambda x: x.lower())

    gse218605 = _load_de_table(paths["gse218605_edger"], "miRNA", "PValue", True)
    rosmap_mir = _load_de_table(paths["rosmap_mirna_limma"], "miRNA", "P.Value", True)

    gse173955 = pd.read_csv(paths["gse173955_edger"])
    gse173955.rename(columns={"Unnamed: 0": "Gene"}, inplace=True)
    gse173955 = gse173955[gse173955["PValue"] < 0.05]

    gse184942 = pd.read_csv(paths["gse184942_edger"])
    gse184942.rename(columns={"Unnamed: 0": "Gene"}, inplace=True)
    gse184942 = gse184942[gse184942["PValue"] < 0.05]

    rosmap_mrna = pd.read_csv(paths["rosmap_mrna_limma"])
    rosmap_mrna.rename(columns={"Unnamed: 0": "Gene"}, inplace=True)
    rosmap_mrna = rosmap_mrna[rosmap_mrna["P.Value"] < 0.05]

    return {
        "GSE120584_miRNA": gse120584,
        "GSE218605_miRNA": gse218605,
        "ROSMAP_miRNA": rosmap_mir,
        "GSE173955_mRNA": gse173955,
        "GSE184942_mRNA": gse184942,
        "ROSMAP_mRNA": rosmap_mrna,
    }


def attach_logfc(stage1, keep_ranks, de_tables) -> pd.DataFrame:
    fold_change_data = {
        "miR": {
            "ROSMAP": dict(zip(de_tables["ROSMAP_miRNA"].miRNA, de_tables["ROSMAP_miRNA"].logFC)),
            "GSE120584": dict(zip(de_tables["GSE120584_miRNA"].miRNA, de_tables["GSE120584_miRNA"].logFC)),
            "GSE218605": dict(zip(de_tables["GSE218605_miRNA"].miRNA, de_tables["GSE218605_miRNA"].logFC)),
        },
        "Gene": {
            "ROSMAP": dict(zip(de_tables["ROSMAP_mRNA"].Gene, de_tables["ROSMAP_mRNA"].logFC)),
            "GSE173955": dict(zip(de_tables["GSE173955_mRNA"].Gene, de_tables["GSE173955_mRNA"].logFC)),
            "GSE184942": dict(zip(de_tables["GSE184942_mRNA"].Gene, de_tables["GSE184942_mRNA"].logFC)),
        },
    }
    rows = {"Feature": [], "Rank": [], "FeatureType": [], "FeatureRanker": [], "Dataset": [], "logFC": []}
    for ranker, d1 in stage1.items():
        ranker_rankers = keep_ranks[ranker]
        for feature_type, features in d1.items():
            feature_fc_data = fold_change_data[feature_type]
            for dataset_name, foldchanges in feature_fc_data.items():
                for feature in features:
                    versions = get_mirna_versions(feature, include_base=True) if feature_type == "miR" else [feature]
                    # Gene branch in the notebook still calls get_mirna_versions.
                    if feature_type != "miR":
                        versions = get_mirna_versions(feature, include_base=True)
                    for version in versions:
                        if version in foldchanges:
                            rows["FeatureRanker"].append(ranker)
                            rows["FeatureType"].append(feature_type)
                            rows["Dataset"].append(dataset_name)
                            rows["Feature"].append(feature)
                            rows["logFC"].append(foldchanges[version])
                            if feature_type == "miR":
                                rank = ranker_rankers[ranker_rankers.MicroRNAs == feature].Rank.values[0]
                            else:
                                rank = None
                            rows["Rank"].append(rank)
                            break
    stage2 = pd.DataFrame(rows)
    stage2 = stage2[stage2.logFC.abs() > 0.1]
    return stage2


def target_association_pass(stage2: pd.DataFrame, dbs: dict) -> dict:
    """Notebook 11 cells 45-63, without replacing stage2 from a saved CSV.

    The original notebook reloaded data/stage2-selected-miRNA-and-targets.csv
    at cell 45. Pass the computed stage2 here. To match that executed
    session, pass the original CSV instead.
    """
    df1 = stage2[stage2.FeatureType == "miR"].copy()
    unexplored = list(stage2[stage2.FeatureType == "miR"].Feature.unique())
    unexplored_mirs_targets = {"TargetScan": {}, "MirTarBase": {}}
    for miR in unexplored:
        versions = [mir.lower() for mir in mirtarbase_query_names(miR)]
        genes = set(dbs["targetscan"][dbs["targetscan"]["miR Family"].isin(versions)]["Gene Symbol"].to_list())
        targets = set(dbs["mirtarbase"][dbs["mirtarbase"].miRNA.isin(versions)].Gene.to_list())
        unexplored_mirs_targets["TargetScan"][miR] = list(genes)
        unexplored_mirs_targets["MirTarBase"][miR] = list(targets)

    all_potential_targets = set()
    for miR in df1.Feature:
        for db_data in unexplored_mirs_targets.values():
            all_potential_targets.update(set(db_data[miR]))

    all_gene_associations = set(dbs["ctd"].Entity.to_list()[:100]).union(set(dbs["genecards"].Entity.to_list()[:100]))
    associated_targets = [target for target in all_potential_targets if target in all_gene_associations]

    candidate_miRs = []
    for miR in df1.Feature:
        candidate_miRs += [mir.lower() for mir in mirtarbase_query_names(miR)]
    mirtarbase_subset = dbs["mirtarbase"][dbs["mirtarbase"].miRNA.isin(candidate_miRs)][dbs["mirtarbase"].Gene.isin(associated_targets)]
    mirtarbase_subset = mirtarbase_subset.reset_index(drop=True).drop_duplicates().reset_index(drop=True)

    ctd_map = dict(zip(dbs["ctd"].Entity, dbs["ctd"].Score))
    genecards_map = dict(zip(dbs["genecards"].Entity, dbs["genecards"].Score))
    features_list = mirtarbase_subset.Gene
    df1["Feature"] = df1["Feature"].apply(lambda x: x.lower())
    df1_map = dict(zip(df1.Feature, df1.Rank))
    mirtarbase_subset["CTD-AD-AS"] = [ctd_map.get(f, 0) for f in features_list]
    mirtarbase_subset["GeneCards-AD-AS"] = [genecards_map.get(f, 0) for f in features_list]
    mirtarbase_subset["miRNA-Rank"] = [df1_map.get(f, 0) for f in mirtarbase_subset.miRNA]
    mirtarbase_subset = mirtarbase_subset.sort_values(
        by=["GeneCards-AD-AS", "CTD-AD-AS", "Sum"], ascending=[False, False, False]
    )
    selected_associations = mirtarbase_subset.copy()
    selected_associations["Val_Sum"] = selected_associations[["Reporter assay", "Western blot", "qPCR"]].sum(axis=1)
    selected_associations = selected_associations[selected_associations["Val_Sum"] == 0]

    top_potential_miR_versions = set()
    for miR in mirtarbase_subset.miRNA.unique():
        top_potential_miR_versions.update(set(get_mirna_versions(miR, include_base=True)))
    selected_miRs = [miR for miR in top_potential_miR_versions if miR in df1.Feature.to_list()]
    potential_gene_targets = mirtarbase_subset.Gene.to_list()
    selected_microRNA_ranks = df1[df1.Feature.isin(selected_miRs)][["Feature", "Rank", "FeatureRanker", "Dataset", "logFC"]]
    return {
        "stage2_mirnas": unexplored,
        "selected_miRs": selected_miRs,
        "selected_associations": selected_associations,
        "selected_microRNA_ranks": selected_microRNA_ranks,
        "potential_gene_targets": potential_gene_targets,
        "top_potential_miR_versions": list(top_potential_miR_versions),
        "literature": literature_review_table(),
    }


def _sort_de(df, log_col, p_col):
    return df.sort_values(by=[log_col, p_col], ascending=[False, True])


def export_workbook(selection: dict, de_tables: dict, out_path: str) -> None:
    """Notebook 11 cell 67 workbook, including the full DE tables and subsets."""
    GSE120584_miRNA_Limma = _sort_de(de_tables["GSE120584_miRNA"].copy(), "logFC", "P.Value")
    ROSMAP_miRNA_Limma = _sort_de(de_tables["ROSMAP_miRNA"].copy(), "logFC", "P.Value")
    ROSMAP_mRNA_Limma = _sort_de(de_tables["ROSMAP_mRNA"].copy(), "logFC", "P.Value")
    GSE173955_mRNA_EdgeR = _sort_de(de_tables["GSE173955_mRNA"].copy(), "logFC", "PValue")
    GSE184942_mRNA_EdgeR = _sort_de(de_tables["GSE184942_mRNA"].copy(), "logFC", "PValue")
    GSE218605_miRNA_EdgeR = _sort_de(de_tables["GSE218605_miRNA"].copy(), "logFC", "PValue")

    versions = selection["top_potential_miR_versions"]
    genes = selection["potential_gene_targets"]
    GSE120584_miRNA_Selected = GSE120584_miRNA_Limma[GSE120584_miRNA_Limma.miRNA.isin(versions)]
    ROSMAP_miRNA_Selected = ROSMAP_miRNA_Limma[ROSMAP_miRNA_Limma.miRNA.isin(versions)]
    GSE218605_miRNA_Selected = GSE218605_miRNA_EdgeR[GSE218605_miRNA_EdgeR.miRNA.isin(versions)]
    GSE173955_mRNA_Selected = GSE173955_mRNA_EdgeR[GSE173955_mRNA_EdgeR.Gene.isin(genes)]
    ROSMAP_mRNA_Selected = ROSMAP_mRNA_Limma[ROSMAP_mRNA_Limma.Gene.isin(genes)]
    GSE184942_mRNA_Selected = GSE184942_mRNA_EdgeR[GSE184942_mRNA_EdgeR.Gene.isin(genes)]

    sheets = [
        ("SelectedMiRs", selection["selected_microRNA_ranks"]),
        ("LiteratureReview", selection["literature"]),
        ("SelectedAssociations", selection["selected_associations"]),
        ("GSE120584_miRNA_Limma", GSE120584_miRNA_Limma),
        ("ROSMAP_miRNA_Limma", ROSMAP_miRNA_Limma),
        ("ROSMAP_mRNA_Limma", ROSMAP_mRNA_Limma),
        ("GSE173955_mRNA_EdgeR", GSE173955_mRNA_EdgeR),
        ("GSE184942_mRNA_EdgeR", GSE184942_mRNA_EdgeR),
        ("GSE218605_miRNA_EdgeR", GSE218605_miRNA_EdgeR),
        ("GSE120584_miRNA_Selected", GSE120584_miRNA_Selected),
        ("ROSMAP_miRNA_Selected", ROSMAP_miRNA_Selected),
        ("GSE218605_miRNA_Selected", GSE218605_miRNA_Selected),
        ("GSE173955_mRNA_Selected", GSE173955_mRNA_Selected),
        ("ROSMAP_mRNA_Selected", ROSMAP_mRNA_Selected),
        ("GSE184942_mRNA_Selected", GSE184942_mRNA_Selected),
    ]
    descriptions = {
        "ROSMAP": "Religious Orders Study & Memory and Aging Project; post-mortem brain multi-omics.",
        "GSE120584": "miRNA expression dataset in Alzheimer’s context (check GEO for tissue/source).",
        "GSE173955": "mRNA-seq in AD vs non-AD (hippocampus; verify on GEO page for exact details).",
        "GSE184942": "mRNA-seq AD vs control (hippocampus/cortex; confirm specific region on GEO).",
        "GSE218605": "miRNA microarray / expression in AD, VCID, NVCID, HY groups.",
    }
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    summary_rows = []
    dataset_seen = defaultdict(bool)
    with pd.ExcelWriter(out, engine="openpyxl") as writer:
        for sheet_name, df in sheets:
            df.to_excel(writer, sheet_name=sheet_name, index=False)
            parts = sheet_name.split("_")
            selected = False
            if parts and parts[-1].lower() == "selected":
                selected = True
                parts = parts[:-1]
            dataset = parts[0] if parts else None
            omics = None
            for tok in parts[1:]:
                if tok.lower() in {"mirna", "mrna"}:
                    omics = "miRNA" if tok.lower() == "mirna" else "mRNA"
                    break
            dge_tool = None
            if parts and parts[-1].lower() in {"limma", "edger", "deseq2"}:
                dge_tool = parts[-1]
            category = sheet_name if sheet_name in {"SelectedMiRs", "LiteratureReview", "SelectedAssociations"} else None
            summary_rows.append({
                "Sheet": sheet_name,
                "Dataset": dataset,
                "Omics": omics,
                "DGE_Tool": dge_tool,
                "SelectedSubset": bool(selected),
                "Category": category,
                "Rows": len(df),
                "Columns": df.shape[1],
            })
            if dataset and not dataset_seen[dataset]:
                dataset_seen[dataset] = True
        summary_df = pd.DataFrame(summary_rows)
        summary_df["is_special"] = summary_df["Category"].notna()
        summary_df = summary_df.sort_values(
            ["is_special", "Dataset", "Omics", "SelectedSubset"], ascending=[True, True, True, False]
        ).drop(columns="is_special")
        summary_df.to_excel(writer, sheet_name="Summary", index=False)
        desc_rows = [{"Dataset": ds, "Description": descriptions.get(ds, "")} for ds, seen in dataset_seen.items() if seen]
        pd.DataFrame(desc_rows, columns=["Dataset", "Description"]).to_excel(
            writer, sheet_name="DatasetDescriptions", index=False
        )
