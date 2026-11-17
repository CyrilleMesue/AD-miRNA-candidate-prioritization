"""Build thesis tables for Results 3.1.2 from the standalone rankings.
Usage: python -I make_tables.py <rankings dir> <out dir>"""
import pandas as pd, sys, os
base, out = sys.argv[1], sys.argv[2]; os.makedirs(out, exist_ok=True)
D=["ROSMAP","GSE120584","GSE215789"]
M=["lasso","ridge","elasticnet","shap","lime","xgb_feature_importance","randomforest_feature_importance","rf_permutation_feature_importance","xgb_permutation_feature_importance","t_test","boruta","mannwhitneyu","svm_rfe"]
NAME={"lasso":"LASSO","ridge":"Ridge","elasticnet":"Elastic net","shap":"SHAP","lime":"LIME","xgb_feature_importance":"XGBoost importance","randomforest_feature_importance":"RF importance","rf_permutation_feature_importance":"RF permutation","xgb_permutation_feature_importance":"XGBoost permutation","t_test":"$t$-test","boruta":"Boruta","mannwhitneyu":"Mann--Whitney U","svm_rfe":"SVM-RFE"}
SMALLER={"t_test","boruta","mannwhitneyu","svm_rfe"}
C=["hsa-miR-324-3p","hsa-miR-574-3p","hsa-miR-769-3p"]
def load(d,m):
    df=pd.read_csv(f"{base}/individual/{d}__{m}.csv"); asc=m in SMALLER
    df["rmin"]=df.score.rank(method="min",ascending=asc).astype(int)
    df["rmax"]=df.score.rank(method="max",ascending=asc).astype(int)
    return df.sort_values("rank_stable")
short=lambda s:s.replace("hsa-","").replace("_","\\_")
rows=[]; top=[]
for d in D:
    for m in M:
        df=load(d,m); n=len(df)
        for c in C:
            r=df[df.feature_id==c].iloc[0]
            rows.append(dict(dataset=d,method=m,candidate=c,n_ranked=n,rank_stable=int(r.rank_stable),
                             rank_tie_min=int(r.rmin),rank_tie_max=int(r.rmax),tie_group_size=int(r.rmax-r.rmin+1),
                             rank_mid=(r.rmin+r.rmax)/2))
        n1=int((df.rmin==1).sum())
        if n1>1: cell=f"{n1} tied at rank 1"
        else:
            t=df.head(3); parts=[]
            for _,x in t.iterrows():
                parts.append(short(x.feature_id)+("$^{a}$" if x.rmax>x.rmin else ""))
            cell=", ".join(parts)
        top.append(dict(dataset=d,method=m,cell=cell,n_rank1=n1,top3=";".join(df.feature_id.head(3))))
R=pd.DataFrame(rows); R["pct_mid"]=R.rank_mid/R.n_ranked*100
R.to_csv(f"{out}/candidate_standalone_ranks.csv",index=False)
T=pd.DataFrame(top); T.to_csv(f"{out}/top_ranked_by_dataset_method.csv",index=False)
S=[]
for c in C:
    for d in D+["All"]:
        g=R[R.candidate==c] if d=="All" else R[(R.candidate==c)&(R.dataset==d)]
        b=g.loc[g.rank_tie_max.idxmin()]
        S.append(dict(candidate=c,dataset=d,n=len(g),best=int(b.rank_tie_max),best_min=int(b.rank_tie_min),best_method=b.method,best_dataset=b.dataset,best_n=int(b.n_ranked),
                      median_pct=round(g.pct_mid.median(),1),top10=int((g.rank_tie_max<=10).sum()),top20=int((g.rank_tie_max<=20).sum()),top50=int((g.rank_tie_max<=50).sum())))
S=pd.DataFrame(S); S.to_csv(f"{out}/candidate_rank_summary.csv",index=False); print(S.to_string())
# --- LaTeX
L=[]
L.append(r"""\begin{table}[htbp]
\centering
\caption{Top-ranked miRNAs in each standalone ranking.}
\label{tab:fs_top_ranked}
{\footnotesize
\begin{tabular}{>{\raggedright\arraybackslash}p{2.2cm} >{\raggedright\arraybackslash}p{3.4cm} >{\raggedright\arraybackslash}p{3.4cm} >{\raggedright\arraybackslash}p{3.4cm}}
\hline
\textbf{Method} & \textbf{ROSMAP} & \textbf{GSE120584} & \textbf{GSE215789} \\
\hline""")
for m in M:
    L.append(NAME[m]+" & "+" & ".join(T[(T.dataset==d)&(T.method==m)].cell.iloc[0] for d in D)+r" \\")
L.append(r"""\hline
\end{tabular}}
\par\smallskip
\parbox{\textwidth}{\footnotesize The three highest-priority miRNAs are listed in rank order, with the
\textit{hsa-} prefix omitted and identifiers given as stored in each matrix. Where more than one miRNA
shared rank~1, the number of tied miRNAs is given instead, because their order is not informative.
$^{a}$Score tied with at least one other miRNA.}
\end{table}
""")
def cell(r):
    return f"{r.rank_tie_min:,}--{r.rank_tie_max:,}" if r.rank_tie_max>r.rank_tie_min else f"{r.rank_tie_min:,}"
L.append(r"""\begin{table}[htbp]
\centering
\caption{Summary of standalone ranking positions of the three experimental candidates.}
\label{tab:fs_candidate_summary}
{\footnotesize
\begin{tabular}{>{\raggedright\arraybackslash}p{2.2cm} >{\raggedright\arraybackslash}p{1.8cm} >{\raggedright\arraybackslash}p{3.6cm} >{\raggedright\arraybackslash}p{1.5cm} >{\raggedright\arraybackslash}p{0.8cm} >{\raggedright\arraybackslash}p{0.8cm} >{\raggedright\arraybackslash}p{0.8cm}}
\hline
\textbf{Candidate} & \textbf{Dataset} & \textbf{Best position (method)} & \textbf{Median position (\%)} & \textbf{Top 10} & \textbf{Top 20} & \textbf{Top 50} \\
\hline""")
for c in C:
    for i,d in enumerate(D+["All"]):
        s=S[(S.candidate==c)&(S.dataset==d)].iloc[0]
        best=(f"{s.best_min:,}--{s.best:,}" if s.best>s.best_min else f"{s.best:,}")+f"/{s.best_n:,} ({NAME[s.best_method]}"+(f", {s.best_dataset})" if d=="All" else ")")
        lab=d if d!="All" else "All (39)"
        L.append(("\\textit{"+c+"}" if i==0 else "")+f" & {lab} & {best} & {s.median_pct:.1f} & {s.top10} & {s.top20} & {s.top50}"+r" \\")
    L.append(r"\hline")
L.append(r"""\end{tabular}}
\par\smallskip
\parbox{\textwidth}{\footnotesize Positions are ranks among all miRNAs scored in that dataset
(ROSMAP 309; GSE120584 2,521; GSE215789 1,137). Where the candidate's score was tied, the range of
tied positions is shown. Median position is the median, across rankings, of the midpoint of the tied
range divided by the number of ranked miRNAs; lower values indicate higher priority. Top-$k$ counts
include a ranking only if the whole tied range lay within the top $k$. These are retrospective
descriptions and were not used as selection criteria.}
\end{table}
""")
L.append(r"""\begin{table}[htbp]
\centering
\caption{Positions of the three experimental candidates in all 39 standalone rankings.}
\label{tab:fs_candidate_all_ranks}
{\footnotesize
\renewcommand{\arraystretch}{0.95}
\begin{tabular}{>{\raggedright\arraybackslash}p{1.8cm} >{\raggedright\arraybackslash}p{2.9cm} >{\raggedright\arraybackslash}p{1.1cm} >{\raggedright\arraybackslash}p{2.0cm} >{\raggedright\arraybackslash}p{2.0cm} >{\raggedright\arraybackslash}p{2.0cm}}
\hline
\textbf{Dataset} & \textbf{Method} & \textbf{Ranked (n)} & \textbf{miR-324-3p} & \textbf{miR-574-3p} & \textbf{miR-769-3p} \\
\hline""")
for d in D:
    for i,m in enumerate(M):
        g=R[(R.dataset==d)&(R.method==m)].set_index("candidate")
        L.append((d if i==0 else "")+f" & {NAME[m]} & {g.n_ranked.iloc[0]:,} & "+" & ".join(cell(g.loc[c]) for c in C)+r" \\")
    L.append(r"\hline")
L.append(r"""\end{tabular}}
\par\smallskip
\parbox{\textwidth}{\footnotesize Rank 1 is the highest priority. A range indicates that the candidate's
score was tied with other miRNAs; positions within a tied group follow input order and carry no ranking
information.}
\end{table}
""")
open(f"{out}/fs_results_tables.tex","w").write("\n".join(L))
