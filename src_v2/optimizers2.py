"""
optimizers2.py -- all optimizers on one footing
================================================

Every method:
  * works on the same discrete variables (one catalogue index per group),
  * ranks designs with the same Deb feasibility rule (Problem.fitness),
  * stops at the same budget of structural analyses (NFE),
  * reports the best FEASIBLE design and the NFE at which it was found.

Stochastic: SA, GA, PSO, ACO, BB-BC, DE.
Deterministic: FSD, continuous SLSQP + catalogue repair (SQP-R),
               and DNS (discrete neighbourhood search, Phase 2 of the hybrid).
Proposed: GA-DNS = GA exploration (Phase 1) + DNS refinement with a
          local-optimality certificate (Phase 2).
"""
import time

import numpy as np
from scipy.optimize import minimize

import engine2 as e
from engine2 import BudgetExhausted, Result, N_CAT


class Tracker:
    """Wraps Problem.fitness; remembers the best feasible design."""

    def __init__(self, p):
        self.p = p
        self.best_w = np.inf
        self.best_idx = None
        self.best_nfe = 0
        self.hist = []
        self.best_f = np.inf
        self.best_f_idx = None

    def __call__(self, idx):
        idx = np.clip(np.asarray(idx, int), 0, N_CAT - 1)
        f, W, ok = self.p.fitness(idx)
        if ok and W < self.best_w - 1e-9:
            self.best_w, self.best_idx, self.best_nfe = W, idx.copy(), self.p.nfe
            self.hist.append((self.p.nfe, W))
        if f < self.best_f:
            self.best_f, self.best_f_idx = f, idx.copy()
        return f

    def result(self, name, t0, note=""):
        ok = self.best_idx is not None
        idx = self.best_idx if ok else self.best_f_idx
        return Result(name, idx, self.p.weight(idx) if idx is not None else np.nan,
                      ok, self.p.nfe, self.best_nfe, self.hist,
                      time.perf_counter() - t0, note)


def _run(name, p, budget, body, seed):
    p.reset(budget)
    rng = np.random.default_rng(seed)
    tr = Tracker(p)
    t0 = time.perf_counter()
    note = ""
    try:
        note = body(p, tr, rng) or ""
    except BudgetExhausted:
        pass
    return tr.result(name, t0, note)


# ---------------------------------------------------------------- SA
def sa(p, budget, seed, T0=0.2, T1=1e-4):
    def body(p, tr, rng):
        x = rng.integers(0, N_CAT, p.n_grp); fx = tr(x)
        n = budget - 1
        for it in range(n):
            T = T0 * (T1 / T0) ** (it / n) * p.w_max
            y = x.copy()
            g = rng.integers(p.n_grp)
            y[g] = np.clip(y[g] + rng.choice([-3, -2, -1, 1, 2, 3]), 0, N_CAT - 1)
            fy = tr(y)
            if fy <= fx or rng.random() < np.exp(-(fy - fx) / T):
                x, fx = y, fy
    return _run("SA", p, budget, body, seed)


# ---------------------------------------------------------------- GA
def _ga_loop(p, tr, rng, npop, n_eval_max, pc=0.9, pm=None, keep_pop=False):
    nG = p.n_grp
    pm = pm if pm is not None else 1.0 / nG
    pop = rng.integers(0, N_CAT, (npop, nG))
    fit = np.array([tr(x) for x in pop])
    used = npop
    while used + npop <= n_eval_max:
        elite = pop[np.argmin(fit)].copy(); f_el = fit.min()
        a, b = rng.integers(0, npop, (2, npop))
        par = pop[np.where(fit[a] < fit[b], a, b)]
        child = par.copy()
        for k in range(0, npop - 1, 2):
            if rng.random() < pc:
                mask = rng.random(nG) < 0.5
                child[k, mask], child[k + 1, mask] = par[k + 1, mask], par[k, mask]
        mut = rng.random(child.shape) < pm
        step = rng.choice([-2, -1, 1, 2], child.shape)
        rnd = rng.integers(0, N_CAT, child.shape)
        big = rng.random(child.shape) < 0.2
        child = np.where(mut, np.where(big, rnd, child + step), child)
        child = np.clip(child, 0, N_CAT - 1)
        fit = np.array([tr(x) for x in child]); used += npop
        w = np.argmax(fit)
        pop = child; pop[w] = elite; fit[w] = f_el          # elitism
    return pop, fit


def ga(p, budget, seed, npop=40):
    def body(p, tr, rng):
        _ga_loop(p, tr, rng, npop, budget)
    return _run("GA", p, budget, body, seed)


# ---------------------------------------------------------------- PSO
def pso(p, budget, seed, npart=30, c1=2.0, c2=2.0):
    def body(p, tr, rng):
        nG = p.n_grp
        x = rng.uniform(0, N_CAT - 1, (npart, nG)); v = np.zeros_like(x)
        pb = x.copy(); pf = np.array([tr(np.rint(xi)) for xi in x])
        g = pb[np.argmin(pf)].copy(); gf = pf.min()
        n_it = budget // npart
        for it in range(n_it):
            w = 0.9 - 0.5 * it / max(n_it - 1, 1)
            r1, r2 = rng.random((2, npart, nG))
            v = w * v + c1 * r1 * (pb - x) + c2 * r2 * (g - x)
            v = np.clip(v, -10, 10)
            x = np.clip(x + v, 0, N_CAT - 1)
            for i in range(npart):
                f = tr(np.rint(x[i]))
                if f < pf[i]:
                    pf[i], pb[i] = f, x[i].copy()
                    if f < gf:
                        gf, g = f, x[i].copy()
    return _run("PSO", p, budget, body, seed)


# ---------------------------------------------------------------- ACO
def aco(p, budget, seed, n_ants=20, alpha=1.0, beta=0.5, rho=0.2):
    def body(p, tr, rng):
        nG = p.n_grp
        tau = np.ones((nG, N_CAT))
        eta = (1.0 / e.CAT_A)[None, :] ** beta
        best, bf = None, np.inf
        while True:
            prob = tau ** alpha * eta
            prob /= prob.sum(1, keepdims=True)
            ants = np.array([[rng.choice(N_CAT, p=prob[g]) for g in range(nG)]
                             for _ in range(n_ants)])
            f = np.array([tr(a) for a in ants])
            if f.min() < bf:
                bf, best = f.min(), ants[np.argmin(f)].copy()
            tau *= (1 - rho)
            for r, i in enumerate(np.argsort(f)[:3]):         # rank-based update
                tau[np.arange(nG), ants[i]] += (3 - r) / 3.0
            tau[np.arange(nG), best] += 2.0                   # elitist ant
            tau = np.clip(tau, 0.02, 20.0)
    return _run("ACO", p, budget, body, seed)


# ---------------------------------------------------------------- BB-BC
def bbbc(p, budget, seed, n=30):
    def body(p, tr, rng):
        nG = p.n_grp
        cand = rng.uniform(0, N_CAT - 1, (n, nG))
        k = 1
        while True:
            f = np.array([tr(np.rint(c)) for c in cand])
            w = 1.0 / f
            cm = (w[:, None] * cand).sum(0) / w.sum()
            best = cand[np.argmin(f)]
            centre = 0.5 * cm + 0.5 * best
            cand = centre + rng.standard_normal((n, nG)) * (N_CAT - 1) * 0.5 / k
            cand = np.clip(cand, 0, N_CAT - 1)
            cand[0] = best
            k += 1
    return _run("BB-BC", p, budget, body, seed)


# ---------------------------------------------------------------- DE
def de(p, budget, seed, npop=30, F=0.6, CR=0.9):
    def body(p, tr, rng):
        nG = p.n_grp
        X = rng.uniform(0, N_CAT - 1, (npop, nG))
        fX = np.array([tr(np.rint(x)) for x in X])
        while True:
            for i in range(npop):
                a, b, c = rng.choice([j for j in range(npop) if j != i], 3, replace=False)
                v = np.clip(X[a] + F * (X[b] - X[c]), 0, N_CAT - 1)
                mask = rng.random(nG) < CR; mask[rng.integers(nG)] = True
                u = np.where(mask, v, X[i])
                fu = tr(np.rint(u))
                if fu <= fX[i]:
                    X[i], fX[i] = u, fu
    return _run("DE", p, budget, body, seed)


# ---------------------------------------------------------------- FSD
def fsd(p, budget, seed=None, max_iter=50):
    """Stress-ratio resizing to the smallest adequate catalogue section."""
    def body(p, tr, rng):
        idx = np.full(p.n_grp, N_CAT // 2)
        seen = set()
        for _ in range(max_iter):
            tr(idx)
            c = p.constraints(idx)                 # same analysis, not re-counted
            new = idx.copy()
            for g in range(p.n_grp):
                mem = np.where(p.groups == g)[0]
                Fm = c["F"][mem]
                for k in range(N_CAT):
                    klr = p.K_eff * p.L[mem] / e.CAT_R[k]
                    ft = e.CAT_A[k] * e.FY / e.GAMMA_M0
                    fc = e.CAT_A[k] * e.fcd(klr)
                    comp = np.any(Fm < 0, axis=1)
                    ok = (np.all(np.max(np.maximum(Fm, 0), 1) <= ft)
                          and np.all(np.max(np.maximum(-Fm, 0), 1) <= fc)
                          and np.all(klr <= np.where(comp, 180, 400)))
                    if ok:
                        new[g] = k; break
                else:
                    new[g] = N_CAT - 1
            key = tuple(new)
            if key in seen:
                break
            seen.add(key); idx = new
        tr(idx)
        return "no displacement control (stress-ratio method)"
    return _run("FSD", p, budget, body, seed)


# ---------------------------------------------------------------- SQP-R
def sqp_r(p, budget, seed=None):
    """Continuous SLSQP on areas (r from a fitted r(A) law), then round each
    group up to the catalogue and repair by DNS-style step-ups."""
    lnA, lnr = np.log(e.CAT_A), np.log(e.CAT_R)
    b1, b0 = np.polyfit(lnA, lnr, 1)
    r_of = lambda A: np.exp(b0) * A ** b1

    def body(p, tr, rng):
        def cons(a_cm2):
            a = a_cm2 * 1e-4
            A = a[p.groups]; k = e.E * A / p.L
            K = p.B.T @ (k[:, None] * p.B)
            U = np.linalg.solve(K, p.P); F = k[:, None] * (p.B @ U)
            klr = p.K_eff * p.L / r_of(A)
            g = [1 - np.max(np.maximum(F, 0), 1) / (A * e.FY / e.GAMMA_M0),
                 1 - np.max(np.maximum(-F, 0), 1) / (A * e.fcd(klr)),
                 1 - klr / 400.0]
            if len(p.disp_pos):
                g.append(1 - np.abs(U[p.disp_pos]).ravel() / p.disp_limit)
            p.nfe += 1
            if p.budget is not None and p.nfe >= p.budget:
                raise BudgetExhausted
            return np.concatenate(g)
        obj = lambda a: e.RHO * 1e-4 * a @ p.L_grp
        a0 = np.full(p.n_grp, e.CAT_A[N_CAT // 2] * 1e4)
        res = minimize(obj, a0, method="SLSQP",
                       bounds=[(e.CAT_A[0] * 1e4, e.CAT_A[-1] * 1e4)] * p.n_grp,
                       constraints={"type": "ineq", "fun": cons},
                       options=dict(maxiter=200, ftol=1e-9))
        idx = np.array([np.searchsorted(e.CAT_A, a * 1e-4 - 1e-12) for a in res.x])
        idx = np.clip(idx, 0, N_CAT - 1)
        _repair(p, tr, idx)
        return "continuous optimum rounded up and repaired"
    return _run("SQP-R", p, budget, body, seed)


# ---------------------------------------------------------------- DNS
def _repair(p, tr, idx, max_steps=500):
    """Greedy feasibility repair: step up the group with the largest ratio."""
    idx = np.array(idx)
    for _ in range(max_steps):
        tr(idx)
        c = p.constraints(idx)
        if (c["stress"].max() <= 1 + 1e-9 and c["slender"].max() <= 1 + 1e-9
                and c["disp"] <= 1 + 1e-9):
            return idx, True
        worst = np.maximum(c["stress"], c["slender"])
        gr = np.zeros(p.n_grp)
        np.maximum.at(gr, p.groups, worst)
        if c["disp"] > 1 and gr.max() <= 1:
            # displacement only: step up the group with the best stiffness gain per kg
            gains = []
            for g in range(p.n_grp):
                if idx[g] == N_CAT - 1:
                    gains.append(-np.inf); continue
                t = idx.copy(); t[g] += 1
                tr(t)                                   # counted analysis
                gains.append((c["disp"] - p.constraints(t)["disp"]) /
                             (p.weight(t) - p.weight(idx) + 1e-12))
            g = int(np.argmax(gains))
        else:
            g = int(np.argmax(np.where(idx < N_CAT - 1, gr, -1)))
        idx[g] = min(idx[g] + 1, N_CAT - 1)
    return idx, False


def dns(p, tr, idx):
    """Deterministic discrete neighbourhood search (steepest descent).

    Neighbourhood N(k) = {k - e_i} U {k - e_i + e_j, i != j}.
    Stops when no feasible neighbour is lighter. The returned design is then
    CERTIFIED to be a local minimum with respect to N(k): every neighbour has
    been analysed and is either heavier or infeasible.
    Returns (idx, certified)."""
    idx, ok = _repair(p, tr, idx)
    if not ok:
        return idx, False
    W = p.weight(idx)
    nG = p.n_grp
    while True:
        cand = []
        for i in range(nG):
            if idx[i] == 0:
                continue
            t = idx.copy(); t[i] -= 1; cand.append(t)
            for j in range(nG):
                if j != i and idx[j] < N_CAT - 1:
                    t2 = t.copy(); t2[j] += 1
                    if p.weight(t2) < W - 1e-9:
                        cand.append(t2)
        cand.sort(key=p.weight)                    # lightest first
        moved = False
        for t in cand:
            f = tr(t)
            if f < e.np.inf and f == p.weight(t) and p.weight(t) < W - 1e-9:
                # feasible (fitness == weight only when feasible) and lighter
                idx, W, moved = t, p.weight(t), True
                break                              # steepest: lightest feasible
        if not moved:
            return idx, True


def ga_dns(p, budget, seed, npop=40, ga_share=0.5, n_elite=3):
    """Proposed hybrid. Phase 1: GA uses ga_share of the budget.
    Phase 2: DNS from the n_elite best distinct GA designs."""
    def body(p, tr, rng):
        pop, fit = _ga_loop(p, tr, rng, npop, int(ga_share * budget))
        order = np.argsort(fit)
        elites, seen = [], set()
        for i in order:
            key = tuple(pop[i])
            if key not in seen:
                seen.add(key); elites.append(pop[i].copy())
            if len(elites) == n_elite:
                break
        if tr.best_idx is not None and tuple(tr.best_idx) not in seen:
            elites.insert(0, tr.best_idx.copy())
        certified = {}
        for el in elites:
            idx, cert = dns(p, tr, el)
            if cert:
                certified[tuple(idx)] = p.weight(idx)
        # Iterated local search with the remaining budget: kick the best
        # certified design on 2 random groups, then descend again (DNS).
        best = min(certified, key=certified.get) if certified else None
        state["best_cert"] = certified.get(best) if best else None
        while best is not None:
            x = np.array(best)
            gs = rng.choice(p.n_grp, size=min(2, p.n_grp), replace=False)
            x[gs] = np.clip(x[gs] + rng.integers(-3, 4, len(gs)), 0, N_CAT - 1)
            idx, cert = dns(p, tr, x)
            if cert and p.weight(idx) < certified[best] - 1e-9:
                best = tuple(idx); certified[best] = p.weight(idx)
                state["best_cert"] = certified[best]
    state = {}
    r = _run("GA-DNS", p, budget, body, seed)
    bc = state.get("best_cert")
    r.note = ("certified local minimum" if bc is not None and abs(bc - r.weight) < 1e-6
              else "best design not certified")
    return r


STOCHASTIC = {"SA": sa, "GA": ga, "PSO": pso, "ACO": aco, "BB-BC": bbbc,
              "DE": de, "GA-DNS": ga_dns}
DETERMINISTIC = {"FSD": fsd, "SQP-R": sqp_r}
