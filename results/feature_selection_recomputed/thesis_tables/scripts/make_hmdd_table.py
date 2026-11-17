"""LaTeX table for Results 'Candidate miRNA prioritization' (HMDD screen).
Usage: python -I make_hmdd_table.py <thesis_tables dir>"""
import sys, pandas as pd
d = sys.argv[1]; per = pd.read_csv(f"{d}/hmdd_screen_mirnas.csv")
CAND = {"hsa-mir-324-3p", "hsa-mir-574-3p", "hsa-mir-769-3p"}  # experimental candidates: bold in every table
def name(m):
    s = m.replace("hsa-mir-", "miR-")
    return r"\textbf{%s}" % s if m in CAND else s
def med(x): return f"{x:g}"
L = [r"""\begin{table}[htbp]
\centering
\caption{Outcome of the HMDD screen for the miRNAs in the 11 ensemble top-20 lists.}
\label{tab:hmdd_screen}
{\footnotesize
\begin{tabular}{l c c c c}
\hline
\textbf{miRNA} & \textbf{Top-20 lists} & \textbf{Best rank} & \textbf{Median rank} & \textbf{HMDD AD records} \\
\hline"""]
POS = pd.read_csv(f"{d}/hmdd_screen_positions.csv")
for known, title in ((False, "Retained (%d miRNAs, %d ranking positions)" % ((~per.hmdd_known).sum(), (~POS.hmdd_known).sum())),
                     (True, "Removed (%d miRNAs, %d ranking positions)" % (per.hmdd_known.sum(), POS.hmdd_known.sum()))):
    g = per[per.hmdd_known == known].sort_values(["n_lists", "best"], ascending=[False, True])
    L.append(r"\multicolumn{5}{l}{\textit{%s}} \\" % title)
    for _, r in g.iterrows():
        L.append(f"{name(r.miRNA)} & {r.n_lists} & {r.best} & {med(r['median'])} & {r.hmdd_score}" + r" \\")
    L.append(r"\hline")
L.append(r"""\end{tabular}}
\par\smallskip
\parbox{\textwidth}{\footnotesize ``Top-20 lists'' counts how many of the 11 ensemble rankings placed the miRNA in the top 20. Best and median ranks are calculated across those lists. ``HMDD AD records'' reports the higher of the two record counts when both the mature and stem identifiers were queried. Candidates with more than one record for either queried identifier were removed. The \textit{hsa-} prefix is omitted. Bold identifies the three miRNAs selected for experimental investigation.}
\end{table}""")
open(f"{d}/hmdd_screen_table.tex", "w").write("\n".join(L) + "\n")
