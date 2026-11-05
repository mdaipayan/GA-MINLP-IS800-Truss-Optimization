"""
analysis2.py -- tables, statistics and colour figures from stats_v2.json
"""
import json
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import mannwhitneyu

import engine2 as e

R = "../results_v2"
FIG = "../figures_v2"
os.makedirs(FIG, exist_ok=True)

# Okabe-Ito colour-blind-safe palette; the proposed method is always vermilion
COL = {"GA-DNS": "#D55E00", "GA": "#0072B2", "DE": "#009E73", "PSO": "#CC79A7",
       "SA": "#E69F00", "ACO": "#56B4E9", "BB-BC": "#7F7F7F",
       "FSD": "#000000", "SQP-R": "#999933"}
ORDER = ["GA-DNS", "GA", "DE", "PSO", "SA", "ACO", "BB-BC"]
plt.rcParams.update({"font.size": 9, "axes.spines.top": False,
                     "axes.spines.right": False, "savefig.dpi": 300})


def load():
    S = json.load(open(f"{R}/stats_v2.json"))
    ref = {}
    for t in S["trusses"]:
        f = f"{R}/milp_{t.replace('-', '')}.json"
        if os.path.exists(f):
            ref[t] = json.load(open(f))
    return S, ref


def best_known(S, ref, t):
    ws = [r["weight"] for m in S["trusses"][t]["methods"].values()
          for r in m["runs"] if r["feasible"]]
    if t in ref and "weight" in ref[t]:
        ws.append(ref[t]["weight"])
    return min(ws)


def curve(hist, grid):
    """best feasible weight as a step function of NFE."""
    out = np.full(len(grid), np.nan)
    for nfe, w in hist:
        out[grid >= nfe] = w
    return out


def summary_table(S, ref):
    lines = []
    for t, T in S["trusses"].items():
        Wst = best_known(S, ref, t)
        cert = t in ref and ref[t].get("gap", 1) <= 1e-6
        lines.append(f"\n{t}  (budget {S['budget'][t]} analyses; best known {Wst:.2f} kg"
                     f"{', certified global optimum by MILP' if cert else ''})")
        lines.append(f"{'method':8s} {'best':>8s} {'median':>8s} {'mean':>8s} {'std':>7s} "
                     f"{'worst':>8s} {'SR':>5s} {'hit%':>5s} {'NFE_med':>8s} {'t(s)':>6s}")
        rows = {}
        for m, M in T["methods"].items():
            w = np.array([r["weight"] for r in M["runs"] if r["feasible"]])
            hit = np.mean([r["feasible"] and r["weight"] <= Wst * (1 + 1e-9) for r in M["runs"]])
            nfe = np.median([r["nfe_best"] for r in M["runs"]])
            tt = np.mean([r["runtime"] for r in M["runs"]])
            sr = len(w) / len(M["runs"])
            rows[m] = dict(best=w.min(), median=np.median(w), mean=w.mean(), std=w.std(ddof=1) if len(w) > 1 else 0,
                           worst=w.max(), sr=sr, hit=hit, nfe=nfe, t=tt)
            lines.append(f"{m:8s} {w.min():8.2f} {np.median(w):8.2f} {w.mean():8.2f} "
                         f"{rows[m]['std']:7.2f} {w.max():8.2f} {sr:5.0%} {hit:5.0%} {nfe:8.0f} {tt:6.2f}")
        if t in ref:
            lines.append(f"{'MILP':8s} {ref[t].get('weight', np.nan):8.2f}   status={ref[t]['status']}, "
                         f"gap={ref[t]['gap']:.2e}, time={ref[t]['runtime']:.1f}s")
        T["summary"] = rows
    return "\n".join(lines)


def tests(S):
    """Mann-Whitney U (independent runs), one-sided H1: GA-DNS lighter, Holm-corrected."""
    out = []
    for t, T in S["trusses"].items():
        g = [r["weight"] for r in T["methods"]["GA-DNS"]["runs"]]
        pv = []
        for m in ORDER[1:]:
            x = [r["weight"] if r["feasible"] else np.inf for r in T["methods"][m]["runs"]]
            if np.allclose(g, x):
                pv.append((m, np.nan, 1.0)); continue
            U, pval = mannwhitneyu(g, x, alternative="less")
            pv.append((m, U, pval))
        k = len(pv); order = np.argsort([q[2] for q in pv]); adj = np.empty(k); run = 0
        for rank, i in enumerate(order):
            run = max(run, min(1, (k - rank) * pv[i][2])); adj[i] = run
        for (m, U, pval), a in zip(pv, adj):
            # Vargha-Delaney A12: prob. GA-DNS run lighter than competitor run
            x = np.array([r["weight"] if r["feasible"] else np.inf for r in T["methods"][m]["runs"]])
            a12 = np.mean([(gi < xi) + 0.5 * (gi == xi) for gi in g for xi in x])
            out.append(dict(truss=t, vs=m, U=U, p=pval, p_holm=a, A12=a12))
    return out


def fig_convergence(S, ref):
    fig, axs = plt.subplots(1, 3, figsize=(7.2, 2.6), constrained_layout=True)
    for ax, (t, T) in zip(axs, S["trusses"].items()):
        B = S["budget"][t]; grid = np.linspace(1, B, 400); Wst = best_known(S, ref, t)
        for m in ORDER:
            C = np.array([curve(r["history"], grid) for r in T["methods"][m]["runs"]])
            med = np.nanmedian(C, 0)
            ax.plot(grid, med, color=COL[m], lw=2.2 if m == "GA-DNS" else 1.2,
                    label=m, zorder=5 if m == "GA-DNS" else 2)
            if m in ("GA-DNS", "GA"):
                ax.fill_between(grid, np.nanpercentile(C, 25, 0), np.nanpercentile(C, 75, 0),
                                color=COL[m], alpha=0.15, lw=0)
        ax.axhline(Wst, color="k", ls=":", lw=1)
        ax.text(B, Wst, " W*", va="center", fontsize=8)
        ax.set_ylim(Wst * 0.97, Wst * 1.45)
        ax.set_title(t); ax.set_xlabel("structural analyses (NFE)")
    axs[0].set_ylabel("best feasible weight (kg)\nmedian of 20 runs")
    axs[2].legend(fontsize=7, frameon=False, loc="upper right")
    fig.savefig(f"{FIG}/fig_convergence_nfe.pdf"); fig.savefig(f"{FIG}/fig_convergence_nfe.png")


def fig_data_profile(S, ref, tau=0.01):
    """Fraction of all runs (3 trusses x 20) that reach W <= (1+tau) W*
    within a given fraction of the budget."""
    fig, axs = plt.subplots(1, 2, figsize=(7.2, 2.7), constrained_layout=True)
    grid = np.linspace(0, 1, 201)
    for ax, tt in zip(axs, (0.0, tau)):
        for m in ORDER:
            reach = []
            for t, T in S["trusses"].items():
                Wst = best_known(S, ref, t); B = S["budget"][t]
                for r in T["methods"][m]["runs"]:
                    hit = [n for n, w in r["history"] if w <= Wst * (1 + tt) + 1e-9]
                    reach.append(hit[0] / B if hit else np.inf)
            frac = [(np.array(reach) <= g).mean() for g in grid]
            ax.step(grid, frac, where="post", color=COL[m], lw=2.2 if m == "GA-DNS" else 1.2, label=m)
        ax.set_xlabel("fraction of analysis budget used")
        ax.set_title("exact best-known weight" if tt == 0 else f"within {tau:.0%} of best-known")
        ax.set_ylim(0, 1.02)
    axs[0].set_ylabel("fraction of runs solved\n(3 trusses × 20 runs)")
    axs[1].legend(fontsize=7, frameon=False, loc="lower right")
    fig.savefig(f"{FIG}/fig_data_profile.pdf"); fig.savefig(f"{FIG}/fig_data_profile.png")


def fig_box(S, ref):
    fig, axs = plt.subplots(1, 3, figsize=(7.2, 2.8), constrained_layout=True)
    for ax, (t, T) in zip(axs, S["trusses"].items()):
        Wst = best_known(S, ref, t)
        data = [np.array([r["weight"] for r in T["methods"][m]["runs"] if r["feasible"]]) / Wst for m in ORDER]
        bp = ax.boxplot(data, vert=False, patch_artist=True, widths=0.6,
                        medianprops=dict(color="k"), flierprops=dict(ms=2))
        for b, m in zip(bp["boxes"], ORDER):
            b.set_facecolor(COL[m]); b.set_alpha(0.75)
        ax.axvline(1, color="k", ls=":", lw=1)
        ax.set_yticks(range(1, len(ORDER) + 1)); ax.set_yticklabels(ORDER if t == "6-bar" else [])
        ax.invert_yaxis(); ax.set_title(t); ax.set_xlabel("W / W*")
        ax.set_xscale("log")
    fig.savefig(f"{FIG}/fig_boxplot_ratio.pdf"); fig.savefig(f"{FIG}/fig_boxplot_ratio.png")


def fig_geometry():
    fig = plt.figure(figsize=(7.2, 2.9))
    cmap = plt.get_cmap("tab20")
    for k, (t, mk) in enumerate(e.BENCHMARKS.items()):
        p = mk(); ax = fig.add_subplot(1, 3, k + 1, projection="3d")
        for m, (i, j) in enumerate(p.conn):
            xyz = p.nodes[[i, j]]
            ax.plot(*xyz.T, color=cmap(p.groups[m] % 20), lw=1.6)
        sup = sorted(set(np.array([d // 3 for d in range(3 * len(p.nodes))
                                   if d not in set(p.free)])))
        ax.scatter(*p.nodes[sup].T, marker="^", color="k", s=18)
        loaded = np.unique(np.nonzero(np.abs(p.P).sum(1))[0])
        ln = sorted(set(p.free[loaded] // 3))
        ax.scatter(*p.nodes[ln].T, color="#D55E00", s=22)
        ax.set_title(f"{t}: {p.n_mem} members, {p.n_grp} groups, {p.n_lc} LC", fontsize=8)
        ax.set_box_aspect(np.ptp(p.nodes, 0) + 1e-9); ax.set_axis_off()
    fig.text(0.5, 0.02, "colours = design groups;  ▲ supports;  ● loaded nodes", ha="center", fontsize=8)
    fig.savefig(f"{FIG}/fig_benchmarks_v2.pdf", bbox_inches="tight")
    fig.savefig(f"{FIG}/fig_benchmarks_v2.png", bbox_inches="tight")


if __name__ == "__main__":
    S, ref = load()
    txt = summary_table(S, ref)
    T = tests(S)
    txt += "\n\nMann-Whitney U, one-sided (GA-DNS lighter), Holm-corrected; A12 = P(GA-DNS run lighter)\n"
    for r in T:
        txt += f"{r['truss']:7s} vs {r['vs']:6s} p={r['p']:.2e} p_holm={r['p_holm']:.2e} A12={r['A12']:.2f}\n"
    open(f"{R}/summary_v2.txt", "w").write(txt); print(txt)
    json.dump({"tests": T, "summary": {t: S["trusses"][t]["summary"] for t in S["trusses"]}},
              open(f"{R}/summary_v2.json", "w"), indent=1, default=float)
    fig_geometry(); fig_convergence(S, ref); fig_data_profile(S, ref); fig_box(S, ref)
