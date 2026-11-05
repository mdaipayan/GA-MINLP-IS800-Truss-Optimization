"""
verify_v2.py -- independent re-check of the reported best designs.

Uses a separate, deliberately simple implementation (element-by-element
assembly with full 6x6 matrices, numpy.linalg.solve, explicit IS 800 formulas
written out again) and compares it with engine2. Also checks static
equilibrium (sum of reactions = applied loads) for every load case.
"""
import json
import math
import sys

import numpy as np

import engine2 as e


def independent_check(p, idx):
    nn = len(p.nodes); nd = 3 * nn
    K = np.zeros((nd, nd))
    A = e.CAT_A[np.asarray(idx)][p.groups]; r = e.CAT_R[np.asarray(idx)][p.groups]
    for m, (i, j) in enumerate(p.conn):
        d = p.nodes[j] - p.nodes[i]; L = math.sqrt(d @ d); c = d / L
        kl = e.E * A[m] / L * np.outer(c, c)
        ke = np.block([[kl, -kl], [-kl, kl]])
        dof = [3 * i, 3 * i + 1, 3 * i + 2, 3 * j, 3 * j + 1, 3 * j + 2]
        K[np.ix_(dof, dof)] += ke
    worst = dict(dcr=0, slender=0, disp=0, eq_err=0)
    for l in range(p.n_lc):
        Pfull = np.zeros(nd); Pfull[p.free] = p.P[:, l]
        U = np.zeros(nd)
        U[p.free] = np.linalg.solve(K[np.ix_(p.free, p.free)], Pfull[p.free])
        R = K @ U - Pfull                                  # reactions at supports
        worst["eq_err"] = max(worst["eq_err"],
                              np.abs(R.reshape(-1, 3).sum(0) + Pfull.reshape(-1, 3).sum(0)).max()
                              / np.abs(Pfull).max())
        for m, (i, j) in enumerate(p.conn):
            d = p.nodes[j] - p.nodes[i]; L = math.sqrt(d @ d); c = d / L
            N = e.E * A[m] / L * c @ (U[3 * j:3 * j + 3] - U[3 * i:3 * i + 3])
            lam = L / r[m]
            if N >= 0:
                cap = A[m] * 250e6 / 1.10; lim = 400
            else:
                fcc = math.pi ** 2 * 200e9 / lam ** 2
                lb = math.sqrt(250e6 / fcc)
                phi = 0.5 * (1 + 0.49 * (lb - 0.2) + lb ** 2)
                fcd = min(250e6 / 1.10, (250e6 / 1.10) / (phi + math.sqrt(phi ** 2 - lb ** 2)))
                cap = A[m] * fcd; lim = 180
            worst["dcr"] = max(worst["dcr"], abs(N) / cap)
            worst["slender"] = max(worst["slender"], lam / lim)
        if p.disp_limit:
            worst["disp"] = max(worst["disp"],
                                np.abs(U[p.free][p.disp_pos]).max() / p.disp_limit)
    W = sum(7850 * A[m] * np.linalg.norm(p.nodes[j] - p.nodes[i])
            for m, (i, j) in enumerate(p.conn))
    return W, worst


if __name__ == "__main__":
    S = json.load(open("../results_v2/stats_v2.json"))
    ok_all = True
    for t, T in S["trusses"].items():
        p = e.BENCHMARKS[t]()
        best = min((r for m in T["methods"].values() for r in m["runs"] if r["feasible"]),
                   key=lambda r: r["weight"])
        W, w = independent_check(p, best["idx"])
        ok = (abs(W - best["weight"]) < 1e-6 and w["dcr"] <= 1 + 1e-9 and
              w["slender"] <= 1 + 1e-9 and w["disp"] <= 1 + 1e-9 and w["eq_err"] < 1e-8)
        ok_all &= ok
        print(f"{t}: reported {best['weight']:.4f} kg, independent {W:.4f} kg | "
              f"max DCR {w['dcr']:.4f}, max slenderness ratio {w['slender']:.4f}, "
              f"max disp ratio {w['disp']:.4f}, equilibrium error {w['eq_err']:.1e} -> "
              f"{'PASS' if ok else 'FAIL'}")
        print("   sections:", [e.CAT_LABEL[i] for i in best["idx"]])
    sys.exit(0 if ok_all else 1)
