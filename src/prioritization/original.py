"""Notebook 11 helper functions, copied from the original notebook.

Source notebook:
    notebooks/11.BIOMARKER SELECTION FOR DOWNSTREAM VALIDATION.ipynb
Cells: scale_dataframe, sort_dict_features, get_mirna_versions,
ad_mirna_target_pipeline, and the manual Literature_Review table.

Do not change thresholds, weights, or filters in this file.
Known original quirks are documented in docs/IMPLEMENTATION_NOTES.md.
"""
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
from tqdm import tqdm

def scale_dataframe(df: pd.DataFrame, label_col: str = "labels") -> pd.DataFrame:
    """
    Applies MinMaxScaler (0–1 scaling) to all numeric columns in a DataFrame,
    except for the specified label column.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame with numeric features and a label column.
    label_col : str, default="labels"
        Column name that should not be scaled.

    Returns
    -------
    pd.DataFrame
        DataFrame with scaled numeric values and unchanged label column.
    """
    scaler = MinMaxScaler()

    # Separate features and labels
    features = df.drop(columns=[label_col])
    labels = df[label_col]

    # Scale features
    scaled_features = scaler.fit_transform(features)

    # Rebuild DataFrame
    scaled_df = pd.DataFrame(scaled_features, columns=features.columns, index=df.index)
    scaled_df[label_col] = labels.values

    return scaled_df


def sort_dict_features(feature_dict, ascending=True, return_with_values=False):
    """
    Sort features based on ranks or weights.

    Parameters
    ----------
    feature_dict : dict
        Dictionary mapping {feature: rank/weight}.
    ascending : bool, default=True
        - True: lower values are better (e.g., ranks).
        - False: higher values are better (e.g., weights).
    return_with_values : bool, default=False
        If True, return list of (feature, value) tuples.
        If False, return list of features only.

    Returns
    -------
    sorted_features : list
        Sorted features (or (feature, value) pairs).
    """
    sorted_items = sorted(
        feature_dict.items(),
        key=lambda x: x[1],
        reverse=not ascending
    )

    if return_with_values:
        return sorted_items
    else:
        return [k for k, v in sorted_items]


# find all versions of the top 20 mirs
def get_mirna_versions(x, include_base =False):
    """
    This function takes a string representing a miRNA or a complex (multiple miRNAs separated by underscores) 
    and returns a list containing all possible "-3p" and "-5p" versions for each miRNA.
    
    Parameters:
    - x (str): A string representing a single miRNA or a complex of miRNAs (joined by underscores).
    
    Return
    - versions (list): A list of strings containing both "-3p" and "-5p" variants for each miRNA.
    """
    
    # Split the input string by underscores to handle multiple miRNAs (if it's a complex miRNA)
    mirnas = x.split('_')  

    # Initialize an empty list to store all possible "-3p" and "-5p" versions of the miRNAs
    versions = []
    
    # Iterate over each miRNA in the split list
    for mirna in mirnas:
        # Check if the miRNA already contains "-3p" or "-5p" suffix
        if "3p" in mirna or "5p" in mirna:
            # Append the "-3p" or "-5p" version of the base miRNA to the list
            versions.append(mirna)
            if include_base:
                versions.append(mirna.split("-3p")[0].split("-5p")[0])
            continue
        else:
            # If the miRNA doesn't contain "-3p" or "-5p", use the miRNA name as it is
            base = mirna

        # Append the "-3p" version of the base miRNA to the list 
        if include_base:
            versions.append(base)
        versions.append(f"{base}-3p")
        # Append the "-5p" version of the base miRNA to the list
        versions.append(f"{base}-5p")
    
    # Return the list of both "-3p" and "-5p" versions for all miRNAs
    return versions


def mirtarbase_query_names(x):
    """Identifiers used to look up a miRNA in miRTarBase/TargetScan at the
    target-association step. Arm-specific names are queried as given by
    get_mirna_versions. A name without -3p/-5p (e.g. hsa-mir-1275) is also
    queried as written, because miRTarBase stores such miRNAs without an arm.
    Changed 2026-11-06; see docs/IMPLEMENTATION_NOTES.md."""
    versions = get_mirna_versions(x)
    if not any(arm in part for part in x.split("_") for arm in ("3p", "5p")):
        versions = versions + x.split("_")
    return versions


def ad_mirna_target_pipeline(
    df,                 # DataFrame with ranked microRNAs (with "MicroRNAs" and "Rank" columns)
    HMDD_db,            # miRNA–AD association db (Entity column with miRNAs)
    miRTarBase_db,      # miRTarBase predictions (miRNA, Gene)
    TargetScan_db,      # TargetScan predictions (miR Family, Gene Symbol)
    ctd_db,             # CTD Alzheimer associations (Entity, Score)
    genecards_db,       # GeneCards Alzheimer associations (Entity, Score)
    gene_db_cutoff:int = 100, # Means look into top 100 gene disease associations from genecards and ctd databases
):
    print("========== STEP 1: Select Top 20 MicroRNAs ==========")
    df1 = df.head(20).copy()
    initial_selection = df1["MicroRNAs"].to_list()
    top20miRs = df1.MicroRNAs.to_list()
    print(f"Top 20 MicroRNAs selected: {top20miRs}")

    # Expand to include all versions
    top20miRs_all_versions = set()
    for miR in top20miRs:
        top20miRs_all_versions.update(set(get_mirna_versions(miR)))
    print(f"Expanded to {len(top20miRs_all_versions)} total versions across databases")

    # Map miRs to their predicted targets in miRTarBase
    mir_to_targets = {}
    for miR in top20miRs_all_versions:
        mir_to_targets[miR] = miRTarBase_db[miRTarBase_db.miRNA == miR].Gene.to_list()

    print("\n========== STEP 2: Filter MicroRNAs without AD Associations ==========")
    unexplored_miR_list = []
    for miR in top20miRs:
        versions = get_mirna_versions(miR, include_base = True)
        present = False
        for v in versions:
            if v.lower() in HMDD_db.Entity.str.lower().to_list():
                present = True
                break
        if not present:
            unexplored_miR_list.append(miR)

    print(f"MicroRNAs without prior AD association (from HMDD): {unexplored_miR_list}")

    # Track predicted targets
    unexplored_data = {'miRNA': [], 'No TargetScan Preds': [], 'No MirTarBase': []}
    unexplored_mirs_targets = {'TargetScan': {}, 'miRTarBase': {}}

    print("\n========== STEP 3: Retrieve Predicted Targets ==========")
    for miR in unexplored_miR_list:
        versions = [mir.lower() for mir in get_mirna_versions(miR, include_base = True)]

        # TargetScan targets
        genes = set(TargetScan_db[TargetScan_db['miR Family'].isin(versions)]['Gene Symbol'].to_list())

        # miRTarBase targets
        targets = set(miRTarBase_db[miRTarBase_db.miRNA.isin(versions)].Gene.to_list())

        unexplored_mirs_targets['TargetScan'][miR] = list(genes)
        unexplored_mirs_targets['miRTarBase'][miR] = list(targets)
        unexplored_data['miRNA'].append(miR)
        unexplored_data['No TargetScan Preds'].append(len(genes))
        unexplored_data['No MirTarBase'].append(len(targets))

    df1 = df1[df1.MicroRNAs.isin(unexplored_miR_list)].copy()
    df1['No TargetScan Preds'] = unexplored_data['No TargetScan Preds']
    df1['No MirTarBase'] = unexplored_data['No MirTarBase']

    # Eliminate miRs with no predicted targets
    df1 = df1[df1[df1.columns[-2:]].sum(axis=1) != 0]
    print(f"Remaining MicroRNAs with at least 1 predicted target: {df1.MicroRNAs.tolist()}")

    print("\n========== STEP 4: AD Association Filtering ==========")
    all_potential_targets = set()
    for miR in df1.MicroRNAs:
        for db, db_data in unexplored_mirs_targets.items():
            all_potential_targets.update(set(db_data.get(miR, [])))

    # Merge CTD + GeneCards gene associations
    all_gene_associations = set(ctd_db.Entity.to_list()[:100]).union(
        set(genecards_db.Entity.to_list()[:100])
    )

    associated_targets = []
    for target in tqdm(all_potential_targets, desc="Checking AD associations"):
        if target in all_gene_associations:
            associated_targets.append(target)

    print(f"Number of AD-associated predicted targets: {len(associated_targets)}")

    print("\n========== STEP 5: Validation Filtering (miRTarBase) ==========")
    candidate_miRs = []
    for miR in df1.MicroRNAs:
        candidate_miRs += [mir.lower() for mir in mirtarbase_query_names(miR)]

    mirtarbase_subset = miRTarBase_db[
        miRTarBase_db.miRNA.isin(candidate_miRs) & miRTarBase_db.Gene.isin(associated_targets)
    ].drop_duplicates().reset_index(drop=True)

    print(f"All Predicted (miRTarBase) subset size: {mirtarbase_subset.shape}")

    # remove associations that have been validated
    mirtarbase_subset['Val_Sum'] = mirtarbase_subset.iloc[:,2:5].sum(axis=1)
    mirtarbase_subset =  mirtarbase_subset[ mirtarbase_subset["Val_Sum"] == 0]

    print(f"Unvalidated Associations(miRTarBase) subset size: {mirtarbase_subset.shape}")
    
    # Add association scores
    CTD_map = dict(zip(ctd_db.Entity, ctd_db.Score))
    Genecards_map = dict(zip(genecards_db.Entity, genecards_db.Score))

    features_list = mirtarbase_subset.Gene
    df1['MicroRNAs'] = df1['MicroRNAs'].str.lower()
    df1_map = dict(zip(df1.MicroRNAs, df1.Rank))

    df1_dict_keys = list(df1_map.keys())
    for miR in df1_dict_keys:
        versions = get_mirna_versions(miR, include_base = True)
        df1_version = ""
        for version in versions:
            if version in initial_selection:
                df1_version = version
        for version in versions:
            df1_map[version] = df1_map.get(df1_version, 0)

    mirtarbase_subset['CTD-AD-AS'] = [CTD_map.get(f, 0) for f in features_list]
    mirtarbase_subset['GeneCards-AD-AS'] = [Genecards_map.get(f, 0) for f in features_list]
    mirtarbase_subset['miRNA-Rank'] = [df1_map.get(f, 0) for f in mirtarbase_subset.miRNA]

    # Prioritize based on AD associations
    mirtarbase_subset['AD-AS-Sum'] = (
        mirtarbase_subset['CTD-AD-AS'] + mirtarbase_subset['GeneCards-AD-AS']
    )
    mirtarbase_subset = mirtarbase_subset.sort_values(
        by=['GeneCards-AD-AS', 'CTD-AD-AS', 'Sum'], ascending=[False, False, False]
    )

    print("\n========== FINAL OUTPUT ==========")
    print(f"Unique MicroRNAs retained: {len(mirtarbase_subset.miRNA.unique())}")
    print(f"Unique AD-associated targets retained: {len(mirtarbase_subset.Gene.unique())}")

    final_gene_set = set(mirtarbase_subset.Gene.unique())
    all_versions_of_final_miRs = set()
    for miR in set(mirtarbase_subset.miRNA.unique()):
        all_versions_of_final_miRs.update(get_mirna_versions(miR, include_base =True))
    final_microRNA_set = set([miR for miR in all_versions_of_final_miRs if miR in initial_selection])

    return final_microRNA_set , final_gene_set


def literature_review_table() -> pd.DataFrame:
    """Manual SH-SY5Y feasibility table from Notebook 11.

    This table is curated text. It does not drop the nine prioritized
    microRNAs down to three names.
    """
    # Literature_Review
    Literature_Review = pd.DataFrame({
        "microRNA": ['hsa-mir-324-3p',
                     'hsa-mir-324-3p',
                     'hsa-mir-324-3p',
                     'hsa-mir-324-3p',
                     "hsa-mir-509-3p",
                     'hsa-mir-769-3p',
                     'hsa-mir-769-3p',
                     'hsa-mir-299-5p',
                     'hsa-mir-885-5p',
                     'hsa-mir-296-5p',
                     "hsa-mir-574-3p", 
                     "hsa-mir-574-3p",
                     "hsa-mir-199b-5p",
                     'hsa-mir-362-3p'
                     
                     
                    ],
        "SH-SY5Y Mention":["SMA-miRs (miR-181a-5p, -324-5p, and -451a) are overexpressed in spinal muscular atrophy skeletal muscle and serum samples",
                           "Dexmedetomidine improves oxygen-glucose deprivation/reoxygenation (OGD/R) -induced neurological injury through regulating SNHG11/miR-324-3p/VEGFA axis",
                           "A High-Throughput Screening Identifies MicroRNA Inhibitors That Influence Neuronal Maintenance and/or Response to Oxidative Stress",
                           "Regulatory effects of the long non‑coding RNA RP11‑543N12.1 and microRNA‑324‑3p axis on the neuronal apoptosis induced by the inflammatory reactions of microglia",
                           "Transcriptomic and genetic studies identify NFAT5 as a candidate gene for cocaine dependence",
                           "miR-769-3p inhibits cellular proliferation of KSHV-infected SH-SY5Y cells through targeting mTOR",
                           "Identification of miRNAs contributing to neuroblastoma chemoresistance",
                           "MiR-299-5p regulates apoptosis through autophagy in neurons and ameliorates cognitive capacity in APPswe/PS1dE9 mice",
                           "EPO Mediates Neurotrophic, Neuroprotective, Anti-Oxidant, and Anti-Apoptotic Effects via Downregulation of miR-451 and miR-885-5p in SH-SY5Y Neuron-Like Cells",
                           "Long noncoding RNA KCNQ1OT1 promotes apoptosis in neuroblastoma cells by regulating miR-296-5p/Bax axis",
                           "RNAi Regulation Profile Comparison in SH-Sy5y and U87 Cells: Response upon ? Endorphin",
                           "TDP-43 regulates cancer-associated microRNAs",
                           "Down-regulation of the Lamin A/C in neuroblastoma triggers the expansion of tumor initiating cells",
                           "miR-362-5p inhibits proliferation and migration of neuroblastoma cells by targeting phosphatidylinositol 3-kinase-C2β"
                          ],
        "Link": ["https://doi.org/10.7554/eLife.68054", 
                 "https://doi.org/10.1080/21655979.2021.1957071",
                 "https://doi.org/10.1016/j.omtn.2019.06.007",
                 "https://doi.org/10.3892/ijmm.2018.3736",
                 "https://doi.org/10.1038/tp.2015.158",
                 "https://doi.org/10.7150/jca.93595",
                 "https://doi.org/10.1016/j.csbj.2015.04.003",
                 "https://doi.org/10.1038/srep24566",
                 "https://pubmed.ncbi.nlm.nih.gov/25324845/",
                 "https://doi.org/10.1111/febs.15047",
                 "https://doi.org/10.47739/2576-0092/1008",
                 "https://doi.org/10.1007/s13238-017-0480-9",
                 "https://doi.org/10.18632/oncotarget.5104",
                 "https://doi.org/10.1016/j.febslet.2015.05.056"
                ]
    })
    return Literature_Review


# Exact ranker and aggregator inventories from Notebook 11 cells 14 and 16.
FEATURE_RANKERS = [
    "lasso", "ridge", "elasticnet",
    "shap", "lime",
    "xgb_feature_importance", "randomforest_feature_importance",
    "rf_permutation_feature_importance", "xgb_permutation_feature_importance",
    "t_test", "boruta", "mannwhitneyu", "svm_rfe",
]

RANK_AGGREGATORS = ["mean", "min", "median", "rra", "geom.mean", "stuart", "mra"]
WEIGHT_AGGREGATORS = ["mean", "max", "median", "geom.mean", "ta"]
AGGREGATORS = [f"{m}_rank" for m in RANK_AGGREGATORS] + [f"{m}_weight" for m in WEIGHT_AGGREGATORS]

# Keys listed in Notebook 11 even though MORE/MOGONET rankers are not run.
DECREASING_IMPORTANCE_SUFFIXES = [
    "t_test", "boruta", "mannwhitneyu", "svm_rfe", "MOGONET:Ranker", "MORE:Ranker",
]
