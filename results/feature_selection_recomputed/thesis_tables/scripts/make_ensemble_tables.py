"""Build the thesis table for Results 3.1.3 from the 12 ensemble rankings.
Usage: python -I make_ensemble_tables.py <rankings dir> <out dir>"""
import pandas as pd, sys, os
base, out = sys.argv[1], sys.argv[2]; os.makedirs(out, exist_ok=True)
e = pd.read_csv(f"{base}/ensembles/ensemble_rankings.csv")
ORDER = ["mean_rank","min_rank","median_rank","rra_rank","geom.mean_rank","stuart_rank","mra_rank",
         "mean_weight","max_weight","median_weight","geom.mean_weight","ta_weight"]
NAME = {"mean_rank":"Mean rank","min_rank":"Minimum rank","median_rank":"Median rank","rra_rank":"RRA",
        "geom.mean_rank":"Geometric-mean rank","stuart_rank":"Stuart","mra_rank":"MRA",
        "mean_weight":"Mean weight","max_weight":"Maximum weight","median_weight":"Median weight",
        "geom.mean_weight":"Geometric-mean weight","ta_weight":"Threshold algorithm"}
CAND = {"hsa-miR-324-3p", "hsa-miR-574-3p", "hsa-miR-769-3p"}  # experimental candidates: bold in every table
short = lambda s: (r"\textbf{%s}" if s in CAND else "%s") % s.replace("hsa-", "")
rows, L = [], []
L.append(r"""\begin{table}[htbp]
\centering
\caption{Ten highest-ranked miRNAs in each ensemble ranking.}
\label{tab:ensemble_top10}
{\footnotesize
\begin{tabular}{>{\raggedright\arraybackslash}p{2.6cm} >{\raggedright\arraybackslash}p{1.3cm} >{\raggedright\arraybackslash}p{9.2cm}}
\hline
\textbf{Ensemble method} & \textbf{Distinct scores} & \textbf{Ten highest-ranked miRNAs, in rank order} \\
\hline""")
for i, a in enumerate(ORDER):
    g = e[e.aggregator == a].sort_values("ensemble_rank").reset_index(drop=True)
    nd = g.ensemble_score.nunique(); cnt = g.ensemble_score.map(g.ensemble_score.value_counts())
    if nd == 1:
        cell = f"All {len(g)} miRNAs received the same score"
    elif nd == 2 and cnt.iloc[1] == len(g) - 1:
        cell = f"{short(g.feature_id[0])}; remaining {len(g)-1} miRNAs received the same score"
    elif cnt.iloc[0] >= 10:
        cell = f"{int(cnt.iloc[0])} miRNAs tied for first place"
    else:
        cell = ", ".join(short(f) + ("$^{a}$" if c > 1 else "") for f, c in zip(g.feature_id[:10], cnt[:10]))
    rows.append(dict(aggregator=a, n=len(g), distinct_scores=nd, top10=";".join(g.feature_id[:10]), top20=";".join(g.feature_id[:20])))
    L.append(f"{NAME[a]} & {nd} & {cell}" + r" \\")
    if i == 6: L.append(r"\hline")
L.append(r"""\hline
\end{tabular}}
\par\smallskip
\parbox{\textwidth}{\footnotesize Rank-based methods are listed above the line and weight-based methods
below it. All 12 methods ranked the same 58 miRNAs. Distinct scores is the number of different
aggregated score values among the 58. The \textit{hsa-} prefix is omitted. $^{a}$Score tied with at
least one other miRNA. The three miRNAs selected for experimental investigation are shown in bold.}
\end{table}
""")
pd.DataFrame(rows).to_csv(f"{out}/ensemble_top_ranked.csv", index=False)
open(f"{out}/ensemble_results_table.tex", "w").write("\n".join(L))
