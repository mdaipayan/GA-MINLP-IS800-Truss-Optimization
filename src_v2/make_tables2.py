"""
make_tables2.py -- LaTeX tables for the manuscript, generated only from
results_v2/*.json so that every number in the paper is traceable.
"""
import json
import os

import numpy as np

import engine2 as e

R = "../results_v2"
OUT = "../manuscript/tables_v2"
os.makedirs(OUT, exist_ok=True)
S = json.load(open(f"{R}/stats_v2.json"))
SUM = json.load(open(f"{R}/summary_v2.json"))
MILP = {t: json.load(open(f"{R}/milp_{t.replace('-', '')}.json")) for t in S["trusses"]}
ORDER = ["GA-DNS", "GA", "DE", "PSO", "SA", "ACO", "BB-BC", "FSD", "SQP-R"]
NAME = {"GA-DNS": r"GA--DNS$^\star$"}


def w(path, s):
    open(f"{OUT}/{path}", "w").write(s)


# ---------------------------------------------------------------- statistics
rows = []
for t, T in S["trusses"].items():
    sm = SUM["summary"][t]
    Wst = min(v["best"] for v in sm.values())
    cert = MILP[t]["gap"] <= 1e-6
    rows.append(r"\midrule" + "\n" + rf"\multicolumn{{8}}{{@{{}}l}}{{\textbf{{{t}}} "
                rf"--- budget {S['budget'][t]:,} analyses; "
                rf"$W^\ast={Wst:.2f}$~kg"
                + (r" (certified global optimum)" if cert else r" (best known; not certified)")
                + r"}\\")
    for m in ORDER:
        v = sm[m]
        det = T["methods"][m]["type"] == "deterministic"
        b = lambda x: rf"\textbf{{{x:.2f}}}" if abs(x - Wst) < 1e-6 else f"{x:.2f}"
        if det:
            rows.append(rf"{NAME.get(m, m)} & {b(v['best'])} & --- & --- & --- & {v['hit']*100:.0f} & {v['nfe']:.0f} & {v['t']:.2f} \\")
        else:
            rows.append(rf"{NAME.get(m, m)} & {b(v['best'])} & {b(v['median'])} & {v['mean']:.2f} & "
                        rf"{v['std']:.2f} & {v['hit']*100:.0f} & {v['nfe']:,.0f} & {v['t']:.2f} \\")
w("tab_statistics.tex", r"""\begin{table*}[tbp]
\centering\small
\caption{Results over $N=20$ independent runs (seeds 0--19) with an equal budget of
structural analyses for every method. All final designs are IS~800:2007-feasible
(success rate 100\,\% for every method). Hit\,=\,share of runs that reach $W^\ast$
exactly; NFE\,=\,median number of analyses needed to find the final best design.
Bold\,=\,equal to $W^\ast$. FSD and SQP-R are deterministic (one run).}
\label{tab:statistics}
\begin{tabular}{@{}lrrrrrrr@{}}
\toprule
Method & Best (kg) & Median (kg) & Mean (kg) & Std (kg) & Hit (\%) & NFE & $\bar t$ (s)\\
""" + "\n".join(rows) + r"""
\bottomrule
\end{tabular}
\end{table*}
""")

# ---------------------------------------------------------------- MILP
rows = []
for t, o in MILP.items():
    status = "optimal (gap 0)" if o["gap"] <= 1e-6 else f"time limit (gap {o['gap']*100:.1f}\\,\\%)"
    rows.append(rf"{t} & {o['n_binary']} & {o['n_cont']:,} & {o['n_rows']:,} & {o['nodes']:,} & "
                rf"{o['runtime']:.1f} & {o['weight']:.2f} & {o['bound']:.2f} & {status}\\")
w("tab_milp.tex", r"""\begin{table*}[tbp]
\centering\footnotesize\setlength{\tabcolsep}{4pt}
\caption{Exact MILP (\cref{sec:milp}) solved with HiGHS branch-and-bound on one CPU core.
Upper bound\,=\,best feasible design; lower bound\,=\,proved bound on the global optimum.
The 72-bar run was stopped at the 5400\,s time limit.}
\label{tab:milp}
\begin{tabular}{@{}lrrrrrrrl@{}}
\toprule
Truss & Binary & Continuous & Rows & B\&B nodes & Time (s) & Upper (kg) & Lower (kg) & Status\\
\midrule
""" + "\n".join(rows) + r"""
\bottomrule
\end{tabular}
\end{table*}
""")

# ---------------------------------------------------------------- tests
rows, last = [], None
for r in SUM["tests"]:
    t = r["truss"] if r["truss"] != last else ""
    last = r["truss"]
    p = r["p_holm"]
    ps = r"$<$0.001" if p < 1e-3 else f"{p:.3f}"
    verdict = (r"GA--DNS lighter" if p < 0.05 else ("n.s." if r["A12"] >= 0.5 else r"n.s.; competitor better"))
    rows.append(rf"{t} & {r['vs']} & {ps} & {r['A12']:.2f} & {verdict}\\")
w("tab_tests.tex", r"""\begin{table}[tbp]
\centering\small
\caption{One-sided Mann--Whitney $U$ tests ($H_1$: GA--DNS gives lighter final designs),
Holm-corrected within each truss, and Vargha--Delaney effect size $A_{12}$
(probability that a GA--DNS run is lighter than a competitor run; 0.5\,=\,no difference).}
\label{tab:tests}
\begin{tabular}{@{}llrrl@{}}
\toprule
Truss & Competitor & $p_{\mathrm{Holm}}$ & $A_{12}$ & Conclusion\\
\midrule
""" + "\n".join(rows) + r"""
\bottomrule
\end{tabular}
\end{table}
""")

# ---------------------------------------------------------------- best designs
def design_table(t, label):
    p = e.BENCHMARKS[t]()
    idx = np.array(MILP[t]["idx"])
    c = p.constraints(idx)
    rows = []
    for g in range(p.n_grp):
        mem = np.where(p.groups == g)[0]
        F = c["F"][mem]
        Fmax, Fmin = F.max() / 1e3, F.min() / 1e3
        rows.append(rf"{g+1} & {len(mem)} & {e.CAT_LABEL[idx[g]].replace('ISA ', '').replace('x', r'$\times$')} & "
                    rf"{e.CAT_A[idx[g]]*1e4:.2f} & {Fmin:.1f} & {Fmax:.1f} & "
                    rf"{c['klr'][mem].max():.0f} & {c['stress'][mem].max():.3f} & {c['slender'][mem].max():.3f}\\")
    return (r"""\begin{table}[tbp]
\centering\footnotesize\setlength{\tabcolsep}{3.5pt}
\caption{""" + label + rf""" Weight {p.weight(idx):.2f}~kg; max.\ displacement ratio {c['disp']:.3f}.
$F$\,=\,range of member forces over all load cases (negative\,=\,compression);
DCR\,=\,max.\ demand/capacity (stress or Annex~D buckling); $\lambda$-ratio\,=\,$(KL/r)/\text{{limit}}$.}}
\label{{tab:design{t.replace('-', '')}}}
\begin{{tabular}}{{@{{}}rrlrrrrrr@{{}}}}
\toprule
Grp & $n$ & ISA section & $A$ (cm$^2$) & $F_{{\min}}$ (kN) & $F_{{\max}}$ (kN) & $KL/r$ & DCR & $\lambda$-ratio\\
\midrule
""" + "\n".join(rows) + r"""
\bottomrule
\end{tabular}
\end{table}
""")


w("tab_design6bar.tex", design_table("6-bar", "Certified optimal 6-bar design (one member per group)."))
w("tab_design25bar.tex", design_table("25-bar", "Certified optimal 25-bar design."))
w("tab_design72bar.tex", design_table("72-bar", "Best known 72-bar design (groups 1--4 = top storey; in each storey: verticals, face diagonals, horizontals, plan diagonals)."))
print("tables written to", OUT)
