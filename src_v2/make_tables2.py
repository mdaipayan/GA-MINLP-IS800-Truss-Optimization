"""
make_tables2.py -- LaTeX tables and number macros for the manuscript, generated
only from results_v2/*.json so that every number in the paper is traceable.
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
ORDER = ["GA-DNS", "DNS-ILS", "GA", "DE", "PSO", "SA", "ACO", "BB-BC", "FSD", "SQP-R"]
NAME = {"GA-DNS": r"GA--DNS$^\star$", "DNS-ILS": r"DNS--ILS$^\dagger$"}


def w(path, s):
    open(f"{OUT}/{path}", "w").write(s)


def best_run(t):
    return min((r for m in S["trusses"][t]["methods"].values() for r in m["runs"] if r["feasible"]),
               key=lambda r: r["weight"])


WST = {t: best_run(t)["weight"] for t in S["trusses"]}
CERT = {t: WST[t] - MILP[t]["bound"] <= 1e-4 * WST[t] for t in S["trusses"]}
GAP = {t: max(0.0, (WST[t] - MILP[t]["bound"]) / WST[t]) for t in S["trusses"]}

# ---------------------------------------------------------------- statistics
rows = []
for t, T in S["trusses"].items():
    sm = SUM["summary"][t]
    Wst = WST[t]
    rows.append(r"\midrule" + "\n" + rf"\multicolumn{{8}}{{@{{}}l}}{{\textbf{{{t}}} "
                rf"--- budget {S['budget'][t]:,} analyses; "
                rf"$W^\ast={Wst:.2f}$~kg"
                + (r" (certified global optimum)" if CERT[t] else r" (best known; not certified)")
                + r"}\\")
    for m in ORDER:
        v = sm[m]
        det = T["methods"][m]["type"] == "deterministic"
        b = lambda x: rf"\textbf{{{x:.2f}}}" if abs(x - Wst) < 1e-6 else f"{x:.2f}"
        if det:
            rows.append(rf"{NAME.get(m, m)} & {b(v['best'])} & --- & --- & --- & 100 & {v['hit']*100:.0f} & {v['nfe']:.0f} \\")
        else:
            rows.append(rf"{NAME.get(m, m)} & {b(v['best'])} & {b(v['median'])} & {v['mean']:.2f} & "
                        rf"{v['std']:.2f} & {v['sr']*100:.0f} & {v['hit']*100:.0f} & {v['nfe']:,.0f} \\")
w("tab_statistics.tex", r"""\begin{table*}[tbp]
\centering\small
\caption{Results over $N=20$ independent runs (seeds 0--19) with an equal budget of
structural analyses for every method. Statistics are over the feasible runs.
SR\,=\,share of runs that end with an IS~800-feasible design;
Hit\,=\,share of runs that reach $W^\ast$ exactly; NFE\,=\,median number of analyses
at which the final design was found. Bold\,=\,equal to $W^\ast$.
FSD and SQP-R are deterministic (one run).
$^\star$proposed; $^\dagger$ablation: the proposed method without the GA phase.}
\label{tab:statistics}
\begin{tabular}{@{}lrrrrrrr@{}}
\toprule
Method & Best (kg) & Median (kg) & Mean (kg) & Std (kg) & SR (\%) & Hit (\%) & NFE\\
""" + "\n".join(rows) + r"""
\bottomrule
\end{tabular}
\end{table*}
""")

# ---------------------------------------------------------------- MILP
rows = []
for t, o in MILP.items():
    status = "optimal" if CERT[t] else "time limit"
    rows.append(rf"{t} & {o['n_binary']:,} & {o['n_cont']:,} & {o['n_rows']:,} & {o['nodes']:,} & "
                rf"{o['runtime']:,.0f} & {o['bound']:.2f} & {WST[t]:.2f} & {GAP[t]*100:.1f} & {status}\\")
w("tab_milp.tex", r"""\begin{table*}[tbp]
\centering\footnotesize\setlength{\tabcolsep}{4pt}
\caption{Exact MILP (\cref{sec:milp}) solved with HiGHS branch-and-bound.
Lower bound\,=\,proved bound on the global optimum; best known\,=\,lightest feasible design
found by any method; gap\,=\,(best known $-$ lower bound)/best known.
The 72-bar run was stopped at the 5400\,s time limit. Times are wall-clock times on a
machine shared with other jobs and are indicative only.}
\label{tab:milp}
\begin{tabular}{@{}lrrrrrrrrl@{}}
\toprule
Truss & Binary & Continuous & Rows & B\&B nodes & Time (s) & Lower bound (kg) & Best known (kg) & Gap (\%) & Status\\
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
    if t and last is not None:
        rows.append(r"\addlinespace")
    last = r["truss"]
    p = r["p_holm"]
    ps = r"$<$0.001" if p < 1e-3 else f"{p:.3f}"
    if p < 0.05:
        verdict = "GA--DNS lighter" if r["A12"] > 0.5 else "competitor lighter"
    else:
        verdict = "no significant difference"
    rows.append(rf"{t} & {r['vs'].replace('DNS-ILS', 'DNS--ILS')} & {ps} & {r['A12']:.2f} & {verdict}\\")
w("tab_tests.tex", r"""\begin{table}[tbp]
\centering\small
\caption{Two-sided Mann--Whitney $U$ tests of the final weights of GA--DNS against
each other method, Holm-corrected within each truss, and Vargha--Delaney effect size
$A_{12}$ (probability that a GA--DNS run is lighter than a competitor run;
0.5\,=\,no difference). Infeasible runs are ranked as worse than every feasible run.}
\label{tab:tests}
\begin{tabular}{@{}llrrl@{}}
\toprule
Truss & Competitor & $p_{\mathrm{Holm}}$ & $A_{12}$ & Conclusion ($\alpha=0.05$)\\
\midrule
""" + "\n".join(rows) + r"""
\bottomrule
\end{tabular}
\end{table}
""")


# ---------------------------------------------------------------- best designs
def design_table(t, label):
    p = e.BENCHMARKS[t]()
    idx = np.array(best_run(t)["idx"])
    c = p.constraints(idx)
    rows = []
    for g in range(p.n_grp):
        mem = np.where(p.groups == g)[0]
        F = c["F"][mem]
        Fmax, Fmin = F.max() / 1e3, F.min() / 1e3
        rows.append(rf"{g+1} & {len(mem)} & {e.CAT_LABEL[idx[g]].replace('ISA ', '').replace('x', r'$\times$')} & "
                    rf"{e.CAT_A[idx[g]]*1e4:.2f} & {e.CAT_R[idx[g]]*1e3:.1f} & {Fmin:.1f} & {Fmax:.1f} & "
                    rf"{c['klr'][mem].max():.0f} & {c['stress'][mem].max():.3f} & {c['slender'][mem].max():.3f}\\")
    return (r"""\begin{table}[tbp]
\centering\footnotesize\setlength{\tabcolsep}{3pt}
\caption{""" + label + rf""" Weight {p.weight(idx):.2f}~kg; max.\ displacement ratio {c['disp']:.3f}.
$F$\,=\,range of member forces over all load cases (negative\,=\,compression);
DCR\,=\,max.\ demand/capacity (yielding or buckling); $\lambda$-ratio\,=\,$(KL/r_v)/\text{{limit}}$.}}
\label{{tab:design{t.replace('-', '')}}}
\begin{{tabular}}{{@{{}}rrlrrrrrrr@{{}}}}
\toprule
Grp & $n$ & Angle & $A$ (cm$^2$) & $r_v$ (mm) & $F_{{\min}}$ (kN) & $F_{{\max}}$ (kN) & $KL/r_v$ & DCR & $\lambda$-ratio\\
\midrule
""" + "\n".join(rows) + r"""
\bottomrule
\end{tabular}
\end{table}
""")


w("tab_design6bar.tex", design_table("6-bar", "Certified optimal 6-bar design (one member per group)."))
w("tab_design25bar.tex", design_table("25-bar", "Certified optimal 25-bar design."))
w("tab_design72bar.tex", design_table("72-bar", "Best known 72-bar design (groups 1--4 = top storey; in each storey: verticals, face diagonals, horizontals, plan diagonals)."))

# ---------------------------------------------------------------- number macros
m25, m72 = MILP["25-bar"], MILP["72-bar"]
w("numbers.tex", "\n".join([
    rf"\newcommand{{\LBsvt}}{{{m72['bound']:.2f}}}",
    rf"\newcommand{{\GAPsvt}}{{{GAP['72-bar']*100:.0f}}}",
    rf"\newcommand{{\NODEStf}}{{{m25['nodes']:,}}}".replace(",", r"\,"),
    rf"\newcommand{{\TIMEtf}}{{{m25['runtime']/60:.0f}}}",
    rf"\newcommand{{\BINsvt}}{{{m72['n_binary']:,}}}".replace(",", r"\,"),
]) + "\n")
print("tables written to", OUT, "| certified:", CERT, "| gaps:", {k: round(v, 4) for k, v in GAP.items()})
