"""Appendix tables: positions of the three experimental candidates in the
39 standalone rankings and in the 12 ensemble rankings (58 shared miRNAs and
per dataset). Writes thesis_tables/appendix_candidate_ranks.tex."""
from pathlib import Path
import pandas as pd

R = Path(__file__).resolve().parents[2]
OUT = R / "thesis_tables" / "appendix_candidate_ranks.tex"
C = ["hsa-miR-324-3p", "hsa-miR-574-3p", "hsa-miR-769-3p"]
DS = ["ROSMAP", "GSE120584", "GSE215789"]
M = ["lasso","ridge","elasticnet","shap","lime","xgb_feature_importance",
     "randomforest_feature_importance","rf_permutation_feature_importance",
     "xgb_permutation_feature_importance","t_test","boruta","mannwhitneyu","svm_rfe"]
MN = {"lasso":"LASSO","ridge":"Ridge","elasticnet":"Elastic net","shap":"SHAP","lime":"LIME",
      "xgb_feature_importance":"XGBoost importance","randomforest_feature_importance":"RF importance",
      "rf_permutation_feature_importance":"RF permutation",
      "xgb_permutation_feature_importance":"XGBoost permutation","t_test":"$t$-test",
      "boruta":"Boruta","mannwhitneyu":"Mann--Whitney U","svm_rfe":"SVM-RFE"}
A = ["mean_rank","min_rank","median_rank","rra_rank","geom.mean_rank","stuart_rank","mra_rank",
     "mean_weight","max_weight","median_weight","geom.mean_weight","ta_weight"]
AN = {"mean_rank":"Mean rank","min_rank":"Minimum rank","median_rank":"Median rank","rra_rank":"RRA",
      "geom.mean_rank":"Geometric-mean rank","stuart_rank":"Stuart","mra_rank":"MRA",
      "mean_weight":"Mean weight","max_weight":"Maximum weight","median_weight":"Median weight",
      "geom.mean_weight":"Geometric-mean weight","ta_weight":"Threshold algorithm"}
short = lambda c: c.replace("hsa-", "")


def cell(rank, tied):
    s = f"{int(rank):,}"
    if rank <= 20:
        s = r"\textbf{" + s + "}"
    return s + (r"$^{\dagger}$" if tied else "")


def header(groups):
    top = " & " + " & ".join(rf"\multicolumn{{3}}{{c}}{{\textbf{{{g}}}}}" for g in groups) + r" \\"
    rule = "".join(rf"\cline{{{2+3*i}-{4+3*i}}}" for i in range(len(groups)))
    sub = r"\textbf{Method} & " + " & ".join(
        rf"\textbf{{{short(c).replace('miR-','')}}}" for _ in groups for c in C) + r" \\"
    return [top, rule, sub]


# ---- standalone ----
s = pd.read_csv(R / "thesis_tables" / "candidate_standalone_ranks.csv")
s = s.set_index(["dataset", "method", "candidate"])
n = {d: int(s.loc[d]["n_ranked"].iloc[0]) for d in DS}
L = [r"\begin{table}[htbp]", r"\centering",
     r"\caption{Positions of the three miRNAs selected for experimental investigation in the 39 standalone rankings.}",
     r"\label{tab:app_standalone_ranks}", r"{\footnotesize\setlength{\tabcolsep}{4pt}",
     r"\begin{tabular}{l " + "r" * 9 + "}", r"\hline"]
L += header([f"{d} ({n[d]:,})" for d in DS]) + [r"\hline"]
for m in M:
    row = [MN[m]]
    for d in DS:
        for c in C:
            r = s.loc[(d, m, c)]
            row.append(cell(r["rank_stable"], r["tie_group_size"] > 1))
    L.append(" & ".join(row) + r" \\")
L += [r"\hline", r"\end{tabular}}", r"\par\smallskip",
      r"\parbox{\textwidth}{\footnotesize Columns give the rank of hsa-miR-324-3p, hsa-miR-574-3p, and "
      r"hsa-miR-769-3p in each dataset; the number of ranked miRNAs is given in parentheses. "
      r"None of the 39 rankings placed a candidate in its top 20. "
      r"$^{\dagger}$Tied with other miRNAs; the position within the tied group is arbitrary.}",
      r"\end{table}", ""]

# ---- ensembles ----
sh = pd.read_csv(R / "ensembles" / "ensemble_rankings.csv")
sh["tied"] = sh.groupby(["aggregator", "ensemble_score"])["feature_id"].transform("size") > 1
sh = sh.set_index(["aggregator", "feature_id"])
pd_ = pd.read_csv(R / "ensembles_per_dataset" / "candidate_ensemble_ranks_by_dataset.csv")
pd_ = pd_.set_index(["dataset", "aggregator", "feature_id"])
L += [r"\begin{table}[htbp]", r"\centering",
      r"\caption{Positions of the three miRNAs selected for experimental investigation in the ensemble rankings.}",
      r"\label{tab:app_ensemble_ranks}", r"{\scriptsize\setlength{\tabcolsep}{3pt}",
      r"\begin{tabular}{l " + "r" * 12 + "}", r"\hline"]
hd = header(["Shared (58)"] + [f"{d} ({n[d]:,})" for d in DS])
hd[2] = hd[2].replace(r"\textbf{Method}", r"\textbf{Ensemble method}")
L += hd + [r"\hline"]
for a in A:
    row = [AN[a]]
    for c in C:
        r = sh.loc[(a, c)]
        row.append(cell(r["ensemble_rank"], r["tied"]))
    for d in DS:
        for c in C:
            r = pd_.loc[(d, a, c)]
            row.append(cell(r["ensemble_rank"], r["n_tied"] > 1))
    L.append(" & ".join(row) + r" \\")
    if a == "mra_rank":
        L.append(r"\hline")
L += [r"\hline", r"\end{tabular}}", r"\par\smallskip",
      r"\parbox{\textwidth}{\footnotesize Columns give the rank of hsa-miR-324-3p, hsa-miR-574-3p, and "
      r"hsa-miR-769-3p. Shared, ensemble rankings of the 58 miRNAs shared by the three datasets; "
      r"ROSMAP, GSE120584, GSE215789, ensemble rankings of the 13 standalone rankings of each dataset "
      r"(revised prioritization). The number of ranked miRNAs is given in parentheses. Ranks within the "
      r"top 20 are shown in bold. $^{\dagger}$Tied with other miRNAs; the position within the tied group is arbitrary.}",
      r"\end{table}", ""]
OUT.write_text("\n".join(L))
print("wrote", OUT)
