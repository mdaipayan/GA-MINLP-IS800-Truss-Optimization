"""
milp_exact.py -- Certified global optimum of the discrete IS 800 sizing problem
===============================================================================

Exact MILP for linear-elastic trusses with a fixed section catalogue
(idea: Rasmussen & Stolpe 2008; Stolpe 2016). For a given member m and
catalogue section k, the IS 800 tension capacity A_k fy/gm0, the Annex-D
compression capacity A_k fcd(KL_m/r_k) and the slenderness limit are all
CONSTANTS. The discrete problem is therefore exactly an MILP, and
branch-and-bound gives a certified global optimum (or a certified gap).

Disaggregated (convex-hull) formulation -- no big-M constants:
  x[g,k] in {0,1}     group g uses section k,  sum_k x[g,k] = 1
  u[:,l]              free-DOF displacements in load case l
  e[m,k,l]            elongation of member m carried "through" section k
  B_m u_l = sum_k e[m,k,l]                                 (compatibility)
  sum_m B_m^T sum_k s_mk e[m,k,l] = P_l, s_mk = E A_k/L_m  (equilibrium)
  -(L_m/E) fcd_mk x[g,k] <= e[m,k,l] <= (L_m/E)(fy/gm0) x[g,k]
       (stress, buckling;  fcd_mk = 0 when KL/r_k > 180 -> no compression)
  x[g,k] = 0 if KL/r_k > 400 for a member of g                 (slenderness)
  |u_d,l| <= d_lim on displacement-limited DOFs
Only the chosen section has a non-zero e[m,k,l], so the member force is
s_mk * e[m,k,l] with the chosen k: the model is exact.
Incumbent pruning: x[g,k] = 0 if rho A_k L_g + sum_{h != g} rho A_min L_h > W_inc.
"""
import time

import numpy as np
from scipy.sparse import coo_matrix

import engine2 as e


def solve_exact(p, time_limit=1800.0, mip_rel_gap=1e-6, verbose=False,
                warm_idx=None, threads=None):
    import highspy
    t0 = time.perf_counter()
    nG, nM, K, nL = p.n_grp, p.n_mem, e.N_CAT, p.n_lc
    nF = len(p.free)
    klr = p.K_eff * p.L[:, None] / e.CAT_R[None, :]               # (nM, K)
    e_t = (p.L * e.FY / (e.GAMMA_M0 * e.E))[:, None].repeat(K, 1)  # max +elongation
    e_c = p.L[:, None] / e.E * e.fcd(klr)                           # max -elongation
    e_c[klr > e.SLEND_COMP] = 0.0
    s = e.E * e.CAT_A[None, :] / p.L[:, None]

    ix = lambda g, k: g * K + k
    nx = nG * K
    iu = lambda d, l: nx + l * nF + d
    nu = nF * nL
    ie = lambda m, k, l: nx + nu + (l * nM + m) * K + k
    nvar = nx + nu + nM * K * nL

    c = np.zeros(nvar)
    for g in range(nG):
        c[ix(g, 0):ix(g, 0) + K] = e.RHO * e.CAT_A * p.L_grp[g]
    lb = np.full(nvar, -np.inf); ub = np.full(nvar, np.inf)
    lb[:nx] = 0; ub[:nx] = 1
    w_min = e.RHO * e.CAT_A[0] * p.L_grp
    W_inc = p.weight(warm_idx) if warm_idx is not None else np.inf
    for g in range(nG):
        mem = np.where(p.groups == g)[0]
        bad = np.any(klr[mem] > e.SLEND_TENS, axis=0)
        too_heavy = e.RHO * e.CAT_A * p.L_grp[g] + (w_min.sum() - w_min[g]) > W_inc + 1e-6
        ub[ix(g, 0):ix(g, 0) + K][bad | too_heavy] = 0
    for l in range(nL):
        for d in p.disp_pos:
            lb[iu(d, l)] = -p.disp_limit; ub[iu(d, l)] = p.disp_limit
    integrality = np.zeros(nvar, bool); integrality[:nx] = True

    rows, cols, vals, lo, hi = [], [], [], [], []
    r = 0

    def add(ci, cv, l_, h_):
        nonlocal r
        rows.extend([r] * len(ci)); cols.extend(ci); vals.extend(cv)
        lo.append(l_); hi.append(h_); r += 1

    for g in range(nG):
        add([ix(g, k) for k in range(K)], [1.0] * K, 1.0, 1.0)
    Bnz = [np.nonzero(p.B[m])[0] for m in range(nM)]
    for l in range(nL):
        for d in range(nF):                                         # equilibrium
            ci, cv = [], []
            for m in np.nonzero(p.B[:, d])[0]:
                for k in range(K):
                    ci.append(ie(m, k, l)); cv.append(p.B[m, d] * s[m, k])
            add(ci, cv, p.P[d, l], p.P[d, l])
        for m in range(nM):
            g = p.groups[m]
            add([iu(d, l) for d in Bnz[m]] + [ie(m, k, l) for k in range(K)],
                list(p.B[m, Bnz[m]]) + [-1.0] * K, 0.0, 0.0)      # compatibility
            for k in range(K):
                if ub[ix(g, k)] == 0:
                    lb[ie(m, k, l)] = 0.0; ub[ie(m, k, l)] = 0.0
                    continue
                add([ie(m, k, l), ix(g, k)], [1.0, -e_t[m, k]], -np.inf, 0.0)
                add([ie(m, k, l), ix(g, k)], [1.0, e_c[m, k]], 0.0, np.inf)

    A = coo_matrix((vals, (rows, cols)), shape=(r, nvar)).tocsc()
    inf = highspy.kHighsInf
    fin = lambda v, s_: np.where(np.isfinite(v), v, s_ * inf)
    lp = highspy.HighsLp()
    lp.num_col_, lp.num_row_ = nvar, r
    lp.col_cost_ = c
    lp.col_lower_, lp.col_upper_ = fin(lb, -1), fin(ub, 1)
    lp.row_lower_, lp.row_upper_ = fin(np.array(lo), -1), fin(np.array(hi), 1)
    lp.a_matrix_.format_ = highspy.MatrixFormat.kColwise
    lp.a_matrix_.start_, lp.a_matrix_.index_, lp.a_matrix_.value_ = A.indptr, A.indices, A.data
    lp.integrality_ = [highspy.HighsVarType.kInteger if v else highspy.HighsVarType.kContinuous
                       for v in integrality]
    h = highspy.Highs()
    h.setOptionValue("output_flag", bool(verbose))
    h.setOptionValue("time_limit", float(time_limit))
    h.setOptionValue("mip_rel_gap", float(mip_rel_gap))
    if threads:
        h.setOptionValue("threads", int(threads))
    h.passModel(lp)
    if warm_idx is not None:
        U, _ = p.analyse(warm_idx)
        x0 = np.zeros(nvar)
        for g in range(nG):
            x0[ix(g, warm_idx[g])] = 1
        for l in range(nL):
            x0[[iu(d, l) for d in range(nF)]] = U[:, l]
            el = p.B @ U[:, l]
            for m in range(nM):
                x0[ie(m, warm_idx[p.groups[m]], l)] = el[m]
        sol = highspy.HighsSolution(); sol.col_value = list(x0); sol.value_valid = True
        h.setSolution(sol)
    h.run()
    info = h.getInfo()
    out = dict(status=h.modelStatusToString(h.getModelStatus()),
               runtime=time.perf_counter() - t0, n_binary=nx,
               n_cont=nvar - nx, n_rows=r, bound=info.mip_dual_bound,
               gap=info.mip_gap, nodes=info.mip_node_count,
               warm_weight=None if warm_idx is None else p.weight(warm_idx))
    xs = np.array(h.getSolution().col_value)
    if len(xs) == nvar and info.primal_solution_status > 0:
        idx = np.argmax(xs[:nx].reshape(nG, K), axis=1)
        p.reset()
        W, v, ok = p.evaluate(idx)                   # independent re-analysis
        out.update(idx=idx, weight=W, verified_feasible=bool(ok), verified_violation=v)
    return out


if __name__ == "__main__":
    import sys
    for name in (sys.argv[1:] or list(e.BENCHMARKS)):
        o = solve_exact(e.BENCHMARKS[name](), time_limit=600)
        print(name, {k: v for k, v in o.items() if k != "idx"})
