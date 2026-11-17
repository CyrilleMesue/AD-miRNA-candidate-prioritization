"""LaTeX tables for Results 3.1.6-3.1.7.
  original_de_table.tex       - original fold-change filter for the 20 miRNAs (tab:original_de)
  revised_source_table.tex    - revised prioritization by entry ranking (tab:revised_by_source)
  revised_shortlist_table.tex - the 52 miRNAs of the revised shortlist (tab:revised_shortlist)
Inputs: prioritization_results.json (compute_prioritization_results.py) and
results/candidates_48_lists/mirna_fate.csv.
Usage: python make_prioritization_tables.py <thesis_tables dir>"""
import json
import sys
from pathlib import Path

import pandas as pd

d = Path(sys.argv[1]).resolve(); ROOT = Path(__file__).resolve().parents[4]
r = json.loads((d / "prioritization_results.json").read_text())
T = lambda v: v is True or v == "True"  # booleans may be serialised as strings
for a_ in r["A"]:
    a_["kept"] = T(a_["kept"])
    for k in ("ROSMAP", "GSE120584", "GSE215789"):
        if a_[k] is not None:
            a_[k]["passed"] = T(a_[k]["passed"]); a_[k]["avail"] = T(a_[k]["avail"])
CAND = {"hsa-mir-324-3p", "hsa-mir-574-3p", "hsa-mir-769-3p"}


def name(m):
    s = m.replace("hsa-mir-", "miR-")
    return r"\textbf{%s}" % s if m in CAND else s


def p(x):
    if x >= 0.001:
        return "1.0" if x >= 0.995 else f"{float(f'{x:.2g}'):g}"
    m, e = f"{x:.1e}".split("e")
    return rf"${m}\times10^{{{int(e)}}}$"


# ---------------------------------------------------------------- original DE (20 miRNAs)
rows = sorted(r["A"], key=lambda a: (not a["kept"], -abs(a["GSE120584"]["logFC"])))
L = [r"""\begin{table}[htbp]
\centering
\caption{Differential-expression evidence for the 20 miRNAs retained after the target analysis (original procedure).}
\label{tab:original_de}
{\footnotesize
\begin{tabular}{l r r r r r r l}
\hline
 & \multicolumn{2}{c}{\textbf{ROSMAP}} & \multicolumn{2}{c}{\textbf{GSE120584}} & \multicolumn{2}{c}{\textbf{GSE215789}} & \\
\cline{2-3}\cline{4-5}\cline{6-7}
\textbf{miRNA} & \textbf{$\log_2\mathrm{FC}$} & \textbf{Raw \textit{p}} & \textbf{$\log_2\mathrm{FC}$} & \textbf{Raw \textit{p}} & \textbf{$\log_2\mathrm{FC}$} & \textbf{Raw \textit{p}} & \textbf{Outcome} \\
\hline"""]
for a in rows:
    cells = []
    for k in ("ROSMAP", "GSE120584", "GSE215789"):
        x = a[k]
        if x is None:
            cells += ["--", "--"]; continue
        fc = f"${x['logFC']:+.3f}$" + (r"$^{\dagger}$" if x["passed"] else "")
        cells += [fc, p(x["P"])]
    L.append(f"{name(a['miRNA'])} & " + " & ".join(cells) + " & " + ("Retained" if a["kept"] else "Removed") + r" \\")
L.append(r"""\hline
\end{tabular}}
\par\smallskip
\parbox{\textwidth}{\footnotesize $\log_2\mathrm{FC}$ is the $\log_2$ fold change for AD versus control, and p is the raw $p$-value. The $^{\dagger}$ symbol marks results that met both filtering thresholds ($|\log_2\mathrm{FC}| > 0.1$ and $p < 0.05$). A miRNA was retained if it met both thresholds in at least one dataset. A dash indicates that no identifier matched the differential-expression results for that dataset. The \textit{hsa-} prefix is omitted. Bold indicates the three miRNAs selected for experimental investigation.}
\end{table}""")
(d / "original_de_table.tex").write_text("\n".join(L) + "\n")

# ---------------------------------------------------------------- revised, by entry ranking
SRC = {"shared58": "Shared miRNAs (11)", "ROSMAP": "ROSMAP (11)", "GSE120584": "GSE120584 (11)",
       "GSE215789": "GSE215789 (11)", "all": "All 44 rankings"}
L = [r"""\begin{table}[htbp]
\centering
\caption{Distinct miRNAs and top-20 ranking positions retained at each step of the revised prioritization, by ranking source.}
\label{tab:revised_by_source}
{\footnotesize
\begin{tabular}{l r r r r}
\hline
\textbf{Ranking source} & \textbf{Top-20 pool} & \textbf{After HMDD} & \shortstack[r]{\textbf{After target}\\\textbf{screening}} & \shortstack[r]{\textbf{After differential-}\\\textbf{expression filtering}} \\
\hline"""]
for t in r["source_table"]:
    if t["source"] == "all":
        L.append(r"\hline")
    L.append(f"{SRC[t['source']]} & {t['pool']} ({t['positions']}) & {t['after_hmdd']} ({t['pos_hmdd']}) & "
             f"{t['after_target']} ({t['pos_target']}) & {t['after_de']} ({t['pos_de']})" + r" \\")
L.append(r"""\hline
\end{tabular}}
\par\smallskip
\parbox{\textwidth}{\footnotesize The number in parentheses after each ranking source is the number of ensemble rankings. In the remaining columns, values outside parentheses count distinct miRNAs, and values in parentheses count top-20 ranking positions. A miRNA appearing in more than one ranking source is counted once per source and once in the overall distinct-miRNA total.}
\end{table}""")
(d / "revised_source_table.tex").write_text("\n".join(L) + "\n")

# ---------------------------------------------------------------- revised shortlist (52)
f = pd.read_csv(ROOT / "results/candidates_48_lists/mirna_fate.csv")
sh = f[f.fate.isin(["Shortlisted", "Selected"])].copy()
orig11 = set((ROOT / "results/candidates_uniform_de/prioritized_mirnas.txt").read_text().split())
sh["orig"] = sh.miRNA.isin(orig11)
sh["cand"] = sh.miRNA.isin(CAND)
sh = sh.sort_values(["orig", "cand", "n_lists", "best"], ascending=[False, False, False, True])
ROUTES = ["shared58", "ROSMAP", "GSE120584", "GSE215789"]
DOT = r"$\bullet$"
L = [r"""\begin{table}[p]
\centering
\caption{The 44 miRNAs of the revised shortlist.}
\label{tab:revised_shortlist}
{\scriptsize\setlength{\tabcolsep}{3.5pt}
\begin{tabular}{l r r c c c c l r r}
\hline
 & & & \multicolumn{4}{c}{\textbf{Entered through rankings of}} & \multicolumn{3}{c}{\textbf{Differential expression}} \\
\cline{4-7}\cline{8-10}
\textbf{miRNA} & \textbf{Lists} & \textbf{Best rank} & \textbf{Shared} & \textbf{ROSMAP} & \textbf{GSE120584} & \textbf{GSE215789} & \textbf{Dataset} & \textbf{$\log_2\mathrm{FC}$} & \textbf{Raw \textit{p}} \\
\hline"""]
for seg, title in ((True, "Members of the original shortlist"),
                   (False, "miRNAs added by the revised prioritization")):
    g = sh[sh.orig == seg]
    L.append(r"\multicolumn{10}{l}{\textit{%s (%d)}} \\" % (title, len(g)))
    for _, x in g.iterrows():
        src = x.sources.split(";")
        marks = " & ".join(DOT if r_ in src else "" for r_ in ROUTES)
        L.append(f"{name(x.miRNA)} & {x.n_lists} & {x.best} & {marks} & {x.de_dataset} & ${(f"{x.logFC:+.4f}" if round(abs(x.logFC), 3) == 0.1 else f"{x.logFC:+.3f}")}$ & {p(x.P)}" + r" \\")
    L.append(r"\hline")
L.append(r"""\end{tabular}}
\par\smallskip
\parbox{\textwidth}{\scriptsize ``Lists'' gives the number of the 44 top-20 lists containing each miRNA, and ``Best rank'' gives its highest position across those lists. ``Shared'' denotes the 11 ensemble rankings of the 58 miRNAs shared by all three datasets. The three dataset columns denote the 11 rankings generated within each dataset. All four sets exclude maximum-weight aggregation. The differential-expression columns report the qualifying dataset, $\log_2$ fold change, and raw $p$-value. When multiple results qualify, the table reports the one with the smallest raw $p$-value. The \textit{hsa-} prefix is omitted. Bold indicates the three miRNAs selected for experimental investigation.}
\end{table}""")
(d / "revised_shortlist_table.tex").write_text("\n".join(L) + "\n")

# ---------------------------------------------------------------- literature review (original 8 + 44 added)
order11 = ["hsa-mir-324-3p", "hsa-mir-574-3p", "hsa-mir-769-3p", "hsa-mir-885-5p",
           "hsa-mir-199b-5p", "hsa-mir-296-5p", "hsa-mir-362-3p", "hsa-mir-299-5p"]
assert set(order11) == orig11
added = sh[~sh.orig].miRNA.tolist()
COLS = (r">{\raggedright\arraybackslash}p{1.8cm} >{\raggedright\arraybackslash}p{3.5cm} "
        r">{\raggedright\arraybackslash}p{3.5cm} >{\raggedright\arraybackslash}p{2.0cm} "
        r">{\raggedright\arraybackslash}p{1.55cm} c")
HEAD = (r"\textbf{miRNA} & \textbf{Reported AD-related or neurological involvement} & "
        r"\textbf{Experimental evidence from neuronal and other cell models} & \textbf{Study identified in an AD-related "
        r"SH-SY5Y model} & \textbf{Key references} & \textbf{Selected} \\")
PEND = r"\pend & \pend & \pend & \pend"
LIT = json.loads((d / "literature_cells.json").read_text())  # literature review (Table 3.8), PubMed-verified


def lit_rows(ms, selected):
    rows = []
    for m in ms:
        c = LIT.get(m)
        if c is None:
            rows.append(f"{name(m)} & {PEND} & {selected(m)}" + r" \\"); continue
        refs = r"\cite{%s}" % ",".join(c["cites"]) if c["cites"] else "None found"
        nm = name(m) + (r"$^{\dagger}$" if c["dag"] else "")
        rows.append(f"{nm} & {c['inv']} & {c['exp']} & {c['model']} & {refs} & {selected(m)}" + r" \\")
    return rows


L = [r"""\providecommand{\pend}{\textcolor{red}{[pending]}}
\begingroup\scriptsize\setlength{\tabcolsep}{2.5pt}
\begin{longtable}{""" + COLS + r"""}
\caption{Literature assessment of the shortlisted miRNAs.}\label{tab:literature_review}\\
\hline
""" + HEAD + r"""
\hline
\endfirsthead
\caption[]{Literature assessment of the shortlisted miRNAs (continued).}\\
\hline
""" + HEAD + r"""
\hline
\endhead
\hline
\endfoot""", r"\multicolumn{6}{l}{\textit{Original shortlist (8)}} \\"]
L += lit_rows(order11, lambda m: "Yes" if m in CAND else "No")
L += [r"\hline", r"\multicolumn{6}{l}{\textit{Added by the revised prioritization (%d)}} \\" % len(added)]
L += lit_rows(added, lambda m: "--")
L += [r"""\end{longtable}
\endgroup
\vspace{-\baselineskip}
{\footnotesize\noindent The literature assessment follows the criteria in Section~\ref{subsec:experimental_candidate_selection}. The 36 miRNAs added by the revised prioritization are listed in the order shown in Table~\ref{tab:revised_shortlist}. A dash in the ``Selected'' column indicates that the miRNA was assessed after I had selected the experimental candidates. The $^{\dagger}$ symbol indicates that the cited studies did not specify the mature arm. A$\beta$ denotes amyloid-$\beta$, and KSHV denotes Kaposi's sarcoma-associated herpesvirus. The \textit{hsa-} prefix is omitted. Bold identifies the three miRNAs selected for experimental investigation.\par}"""]
(d / "literature_table.tex").write_text("\n".join(L) + "\n")
print("tables written", len(added))
