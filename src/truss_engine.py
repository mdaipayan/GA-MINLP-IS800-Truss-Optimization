"""
truss_engine.py  (Numba-accelerated edition)
─────────────────────────────────────────────
Colab-safe, drop-in replacement for the original truss_engine.py.

Key Numba JIT kernels (compiled once, reused every call):
  _dsm_solve_nb      — Direct Stiffness Method solver (n_dof × n_dof)
  _fcd_nb            — IS 800:2007 Annex D buckling stress (scalar)
  _fcd_vec_nb        — Vectorised Annex D over all members
  _penalty_nb        — Stress + slenderness penalty accumulator
  _weight_nb         — Member weight summation
  _eval_nb           — Full fitness = weight + penalty (single call)
  _fsd_inner_nb      — FSD resizing inner loop
  _sa_inner_nb       — Simulated Annealing main loop
  _ga_fitness_nb     — GA population fitness batch evaluation
  _pso_inner_nb      — PSO velocity/position update loop
  _aco_prob_nb       — ACO pheromone probability computation

All other routines are pure Python / NumPy (they are not on the hot path).

Install (Colab):
    !pip install numba -q
    # then: from truss_engine import *

Numba AOT caching:
    Set NUMBA_CACHE_DIR env var to persist cache across Colab sessions.
    (Colab tips: mount Google Drive, set NUMBA_CACHE_DIR to a Drive path.)
"""

import os, time, math, random
import numpy as np
from numba import njit, prange, float64, int64, boolean
from numba.typed import List as NList
from scipy.optimize import linprog, minimize, differential_evolution
from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Dict


# ─────────────────────────────────────────────────────────────────
# Reproducible random seed handling
# ─────────────────────────────────────────────────────────────────
def _prepare_seed(seed: Optional[int] = None) -> Tuple[int, random.Random]:
    """Seed Python and NumPy RNGs and return a local Python RNG.

    If seed is None, a seed is drawn from the current NumPy RNG state.
    This preserves backwards compatibility with scripts that call
    np.random.seed(seed) before calling an optimizer, while also allowing
    fully explicit seed control through optimizer(..., seed=seed).
    """
    if seed is None:
        seed = int(np.random.randint(0, 2**31 - 1))
    seed = int(seed) % (2**31 - 1)
    random.seed(seed)
    np.random.seed(seed)
    return seed, random.Random(seed)

# ─────────────────────────────────────────────────────────────────
# Numba AOT cache (honours NUMBA_CACHE_DIR env var on Colab/Drive)
# ─────────────────────────────────────────────────────────────────
_CACHE = True   # set False to disable caching (useful for debugging)

# ══════════════════════════════════════════════════════════════════
# SP 6(1):1964  —  51 ISA equal-angle sections
# Each entry: (label, A_cm², r_min_cm, weight_kg/m)
# ══════════════════════════════════════════════════════════════════
SP6_CATALOG = [
    ("ISA 20x20x3",     1.11,  0.609,  0.87),
    ("ISA 25x25x3",     1.42,  0.762,  1.11),
    ("ISA 25x25x4",     1.85,  0.748,  1.45),
    ("ISA 30x30x3",     1.73,  0.922,  1.36),
    ("ISA 30x30x4",     2.27,  0.906,  1.78),
    ("ISA 35x35x4",     2.67,  1.065,  2.10),
    ("ISA 35x35x5",     3.25,  1.048,  2.55),
    ("ISA 40x40x4",     3.08,  1.225,  2.42),
    ("ISA 40x40x5",     3.79,  1.208,  2.97),
    ("ISA 45x45x4",     3.49,  1.384,  2.74),
    ("ISA 45x45x5",     4.30,  1.366,  3.37),
    ("ISA 45x45x6",     5.07,  1.349,  3.98),
    ("ISA 50x50x5",     4.80,  1.526,  3.77),
    ("ISA 50x50x6",     5.69,  1.508,  4.47),
    ("ISA 55x55x5",     5.30,  1.686,  4.16),
    ("ISA 55x55x6",     6.30,  1.667,  4.95),
    ("ISA 60x60x5",     5.82,  1.846,  4.57),
    ("ISA 60x60x6",     6.91,  1.826,  5.42),
    ("ISA 60x60x8",     9.03,  1.789,  7.09),
    ("ISA 65x65x6",     7.52,  1.987,  5.90),
    ("ISA 65x65x8",     9.86,  1.950,  7.74),
    ("ISA 70x70x6",     8.13,  2.147,  6.38),
    ("ISA 70x70x7",     9.40,  2.129,  7.38),
    ("ISA 75x75x6",     8.74,  2.306,  6.86),
    ("ISA 75x75x8",    11.40,  2.270,  8.96),
    ("ISA 75x75x10",   14.00,  2.234, 11.00),
    ("ISA 80x80x6",     9.35,  2.466,  7.34),
    ("ISA 80x80x8",    12.30,  2.428,  9.65),
    ("ISA 80x80x10",   15.20,  2.391, 11.90),
    ("ISA 90x90x6",    10.60,  2.785,  8.30),
    ("ISA 90x90x8",    13.90,  2.747, 10.90),
    ("ISA 90x90x10",   17.10,  2.709, 13.40),
    ("ISA 100x100x6",  11.80,  3.100,  9.26),
    ("ISA 100x100x8",  15.60,  3.060, 12.20),
    ("ISA 100x100x10", 19.20,  3.021, 15.10),
    ("ISA 100x100x12", 22.80,  2.983, 17.90),
    ("ISA 110x110x8",  17.20,  3.380, 13.50),
    ("ISA 110x110x10", 21.20,  3.340, 16.60),
    ("ISA 110x110x12", 25.10,  3.300, 19.70),
    ("ISA 120x120x8",  18.80,  3.699, 14.80),
    ("ISA 120x120x10", 23.20,  3.659, 18.20),
    ("ISA 120x120x12", 27.50,  3.618, 21.60),
    ("ISA 130x130x10", 25.20,  3.977, 19.80),
    ("ISA 130x130x12", 29.90,  3.937, 23.50),
    ("ISA 150x150x10", 29.20,  4.617, 22.90),
    ("ISA 150x150x12", 34.70,  4.575, 27.20),
    ("ISA 150x150x15", 42.90,  4.512, 33.70),
    ("ISA 200x200x16", 62.60,  6.153, 49.10),
    ("ISA 200x200x20", 77.60,  6.091, 60.90),
    ("ISA 200x200x25", 95.70,  6.010, 75.10),
    ("ISA 200x200x32",121.00,  5.910, 95.00),
]

# Contiguous NumPy arrays (used in all JIT kernels)
CAT_A = np.array([s[1] for s in SP6_CATALOG], dtype=np.float64) * 1e-4  # m²
CAT_r = np.array([s[2] for s in SP6_CATALOG], dtype=np.float64) * 1e-2  # m
CAT_w = np.array([s[3] for s in SP6_CATALOG], dtype=np.float64)          # kg/m
N_CAT = len(SP6_CATALOG)

# Material / code constants
E            = 200e9    # Pa
rho          = 7850.0   # kg/m³
fy           = 250e6    # Pa
gm0          = 1.10
KL_lim_comp  = 180
KL_lim_tens  = 400
_ALPHA_IS800 = 0.49     # buckling class c (angle sections)


# ══════════════════════════════════════════════════════════════════
# ██  NUMBA JIT KERNELS  ██
# ══════════════════════════════════════════════════════════════════

@njit(cache=_CACHE)
def _fcd_nb(KL_r: float64) -> float64:
    """IS 800:2007 Annex D design compressive stress fcd (Pa)."""
    if KL_r <= 0.0:
        return fy / gm0
    fcc = (math.pi ** 2) * E / (KL_r * KL_r)
    lam = math.sqrt(fy / fcc)
    phi = 0.5 * (1.0 + _ALPHA_IS800 * (lam - 0.2) + lam * lam)
    disc = phi * phi - lam * lam
    if disc < 1e-12:
        disc = 1e-12
    chi = 1.0 / (phi + math.sqrt(disc))
    if chi > 1.0:
        chi = 1.0
    return chi * fy / gm0


@njit(cache=_CACHE)
def _fcd_vec_nb(KL_r_arr: float64[:]) -> float64[:]:
    """Vectorised Annex D over an array of KL/r values."""
    n = KL_r_arr.shape[0]
    out = np.empty(n, dtype=np.float64)
    for i in range(n):
        out[i] = _fcd_nb(KL_r_arr[i])
    return out


@njit(cache=_CACHE)
def _weight_nb(A_arr: float64[:], L_arr: float64[:]) -> float64:
    """Total steel weight W = rho * sum(A_i * L_i)."""
    s = 0.0
    for i in range(A_arr.shape[0]):
        s += A_arr[i] * L_arr[i]
    return rho * s


@njit(cache=_CACHE)
def _penalty_nb(forces: float64[:],
                A_arr:  float64[:],
                L_arr:  float64[:],
                r_arr:  float64[:],
                K_eff:  float64,
                mu:     float64) -> float64:
    """Stress + slenderness constraint penalty (metaheuristic fitness)."""
    pen = 0.0
    fa_tens = fy / gm0
    for i in range(forces.shape[0]):
        F = forces[i]; A = A_arr[i]; r = r_arr[i]; L = L_arr[i]
        if A <= 0.0:
            pen += mu * 1e6
            continue
        sigma = abs(F) / A
        KLr   = K_eff * L / r if r > 0.0 else 1e9
        if F < 0.0:   # compression
            fa  = _fcd_nb(KLr)
            lim = 180.0
        else:         # tension
            fa  = fa_tens
            lim = 400.0
        vio = sigma / fa - 1.0
        if vio > 0.0:
            pen += mu * vio * vio
        sv = KLr / lim - 1.0
        if sv > 0.0:
            pen += mu * sv * sv
    return pen


@njit(cache=_CACHE)
def _dsm_solve_nb(nodes:  float64[:, :],
                  conn:   int64[:, :],
                  bc_dof: int64[:],
                  loads:  float64[:],
                  A_arr:  float64[:]):
    """
    3-D Direct Stiffness Method.
    Returns (U: float64[n_dof], forces: float64[n_mem], max_disp: float64).

    NOTE: Numba does not support named tuples; caller unpacks positionally.
    """
    n_nodes = nodes.shape[0]
    n_mem   = conn.shape[0]
    ndof    = 3 * n_nodes

    K = np.zeros((ndof, ndof), dtype=np.float64)

    for m in range(n_mem):
        ni = conn[m, 0]; nj = conn[m, 1]
        dx = nodes[nj, 0] - nodes[ni, 0]
        dy = nodes[nj, 1] - nodes[ni, 1]
        dz = nodes[nj, 2] - nodes[ni, 2]
        L  = math.sqrt(dx*dx + dy*dy + dz*dz)
        if L < 1e-12:
            continue
        l = dx / L; m_ = dy / L; n = dz / L
        k = E * A_arr[m] / L
        dc = np.array([l, m_, n, -l, -m_, -n], dtype=np.float64)
        dof = np.array([3*ni, 3*ni+1, 3*ni+2, 3*nj, 3*nj+1, 3*nj+2],
                       dtype=np.int64)
        for a in range(6):
            for b in range(6):
                K[dof[a], dof[b]] += k * dc[a] * dc[b]

    # Build free DOF list
    n_bc = bc_dof.shape[0]
    bc_set = np.zeros(ndof, dtype=np.int64)
    for i in range(n_bc):
        bc_set[bc_dof[i]] = 1

    n_free = ndof - n_bc
    free_dof = np.empty(n_free, dtype=np.int64)
    fi = 0
    for d in range(ndof):
        if bc_set[d] == 0:
            free_dof[fi] = d
            fi += 1

    # Extract K_ff and F_f
    K_ff = np.zeros((n_free, n_free), dtype=np.float64)
    F_f  = np.zeros(n_free, dtype=np.float64)
    for a in range(n_free):
        F_f[a] = loads[free_dof[a]]
        for b in range(n_free):
            K_ff[a, b] = K[free_dof[a], free_dof[b]]

    # Solve K_ff @ U_f = F_f  (Numba supports np.linalg.solve)
    try:
        U_f = np.linalg.solve(K_ff, F_f)
    except Exception:
        U_f = np.zeros(n_free, dtype=np.float64)

    U = np.zeros(ndof, dtype=np.float64)
    for i in range(n_free):
        U[free_dof[i]] = U_f[i]

    # Member forces
    forces_out = np.zeros(n_mem, dtype=np.float64)
    for m in range(n_mem):
        ni = conn[m, 0]; nj = conn[m, 1]
        dx = nodes[nj, 0] - nodes[ni, 0]
        dy = nodes[nj, 1] - nodes[ni, 1]
        dz = nodes[nj, 2] - nodes[ni, 2]
        L  = math.sqrt(dx*dx + dy*dy + dz*dz)
        if L < 1e-12:
            continue
        l = dx/L; m_ = dy/L; n = dz/L
        dU = (l  * (U[3*nj]   - U[3*ni]) +
              m_ * (U[3*nj+1] - U[3*ni+1]) +
              n  * (U[3*nj+2] - U[3*ni+2]))
        forces_out[m] = E * A_arr[m] / L * dU

    max_disp = 0.0
    for i in range(ndof):
        if abs(U[i]) > max_disp:
            max_disp = abs(U[i])

    return U, forces_out, max_disp


# Numba doesn't allow bare tuple type hint in @njit; patch after definition
# We use a workaround: return as separate arrays via a wrapper
# (Python callers just unpack; the JIT version already works)


@njit(cache=_CACHE)
def _eval_nb(cat_idx:  int64[:],
             nodes:    float64[:, :],
             conn:     int64[:, :],
             bc_dof:   int64[:],
             loads:    float64[:],
             L_arr:    float64[:],
             CAT_A_nb: float64[:],
             CAT_r_nb: float64[:],
             K_eff:    float64,
             mu:       float64):
    """
    Full fitness: weight + penalty.
    Returns (fitness: float, is_feasible: bool).
    """
    nm    = cat_idx.shape[0]
    A_arr = np.empty(nm, dtype=np.float64)
    r_arr = np.empty(nm, dtype=np.float64)
    for i in range(nm):
        k = cat_idx[i]
        if k < 0: k = 0
        if k >= CAT_A_nb.shape[0]: k = CAT_A_nb.shape[0] - 1
        A_arr[i] = CAT_A_nb[k]
        r_arr[i] = CAT_r_nb[k]

    _, forces, _ = _dsm_solve_nb(nodes, conn, bc_dof, loads, A_arr)
    W   = _weight_nb(A_arr, L_arr)
    pen = _penalty_nb(forces, A_arr, L_arr, r_arr, K_eff, mu)
    return W + pen, pen == 0.0


@njit(cache=_CACHE)
def _member_lengths_nb(nodes: float64[:, :],
                       conn:  int64[:, :]) -> float64[:]:
    """Pre-compute member lengths."""
    n_mem = conn.shape[0]
    L = np.empty(n_mem, dtype=np.float64)
    for m in range(n_mem):
        ni = conn[m, 0]; nj = conn[m, 1]
        dx = nodes[nj, 0] - nodes[ni, 0]
        dy = nodes[nj, 1] - nodes[ni, 1]
        dz = nodes[nj, 2] - nodes[ni, 2]
        L[m] = math.sqrt(dx*dx + dy*dy + dz*dz)
    return L


@njit(cache=_CACHE)
def _fsd_inner_nb(n_mem:    int64,
                  idx:      int64[:],
                  forces:   float64[:],
                  A_arr:    float64[:],
                  r_arr:    float64[:],
                  L_arr:    float64[:],
                  K_eff:    float64,
                  CAT_A_nb: float64[:],
                  CAT_r_nb: float64[:],
                  n_cat:    int64) -> int64[:]:
    """FSD resizing inner loop — returns updated section index array."""
    new_idx = idx.copy()
    fa_tens = fy / gm0
    for m in range(n_mem):
        F = forces[m]; A = A_arr[m]; r = r_arr[m]; L = L_arr[m]
        if abs(F) < 1e-3:
            # near-zero force: pick smallest slenderness-feasible
            for c in range(n_cat):
                rc  = CAT_r_nb[c]
                KLr = K_eff * L / rc if rc > 0.0 else 1e9
                if KLr <= 180.0:
                    new_idx[m] = c
                    break
            else:
                new_idx[m] = n_cat - 1
            continue

        if F < 0.0:
            fa = _fcd_nb(K_eff * L / r if r > 0.0 else 1e9)
        else:
            fa = fa_tens

        sigma = abs(F) / A if A > 0.0 else 1e12
        ratio = sigma / fa if fa > 0.0 else 1.0
        A_new = A * ratio

        # Find nearest catalog index
        best_c = 0
        best_d = abs(CAT_A_nb[0] - A_new)
        for c in range(1, n_cat):
            d = abs(CAT_A_nb[c] - A_new)
            if d < best_d:
                best_d = d
                best_c = c

        # Bump up if stress or slenderness violated
        lim = 180.0 if F < 0.0 else 400.0
        c = best_c
        while c < n_cat - 1:
            Ac  = CAT_A_nb[c]; rc = CAT_r_nb[c]
            if Ac <= 0.0:
                c += 1; continue
            sig_c = abs(F) / Ac
            fa_c  = _fcd_nb(K_eff * L / rc) if F < 0.0 else fa_tens
            KLr_c = K_eff * L / rc if rc > 0.0 else 1e9
            if sig_c <= fa_c and KLr_c <= lim:
                break
            c += 1
        new_idx[m] = c
    return new_idx


@njit(parallel=True, cache=_CACHE)
def _ga_fitness_nb(pop:      int64[:, :],
                   nodes:    float64[:, :],
                   conn:     int64[:, :],
                   bc_dof:   int64[:],
                   loads:    float64[:],
                   L_arr:    float64[:],
                   CAT_A_nb: float64[:],
                   CAT_r_nb: float64[:],
                   K_eff:    float64,
                   mu:       float64) -> float64[:]:
    """
    Parallel fitness evaluation for a GA population.
    Each individual evaluated independently — no data dependency.
    """
    npop = pop.shape[0]
    fitness = np.empty(npop, dtype=np.float64)
    for i in prange(npop):   # ← parallel loop
        f, _ = _eval_nb(pop[i], nodes, conn, bc_dof, loads, L_arr,
                        CAT_A_nb, CAT_r_nb, K_eff, mu)
        fitness[i] = f
    return fitness


@njit(parallel=True, cache=_CACHE)
def _bbbc_fitness_nb(pop:      int64[:, :],
                     nodes:    float64[:, :],
                     conn:     int64[:, :],
                     bc_dof:   int64[:],
                     loads:    float64[:],
                     L_arr:    float64[:],
                     CAT_A_nb: float64[:],
                     CAT_r_nb: float64[:],
                     K_eff:    float64,
                     mu:       float64) -> float64[:]:
    """Same parallel fitness batch for BB-BC."""
    npop = pop.shape[0]
    fs = np.empty(npop, dtype=np.float64)
    for i in prange(npop):
        f, _ = _eval_nb(pop[i], nodes, conn, bc_dof, loads, L_arr,
                        CAT_A_nb, CAT_r_nb, K_eff, mu)
        fs[i] = f
    return fs


@njit(cache=_CACHE)
def _sa_inner_nb(idx0:     int64[:],
                 nodes:    float64[:, :],
                 conn:     int64[:, :],
                 bc_dof:   int64[:],
                 loads:    float64[:],
                 L_arr:    float64[:],
                 CAT_A_nb: float64[:],
                 CAT_r_nb: float64[:],
                 K_eff:    float64,
                 mu:       float64,
                 T0:       float64,
                 Tend:     float64,
                 n_iter:   int64,
                 seed:     int64):
    """
    SA main loop — fully JIT compiled.
    Returns (best_idx, best_weight, weight_history).
    history sampled every 100 iterations.
    """
    np.random.seed(seed)
    nm       = idx0.shape[0]
    idx      = idx0.copy()
    best_idx = idx0.copy()

    curr_f, _ = _eval_nb(idx, nodes, conn, bc_dof, loads, L_arr,
                         CAT_A_nb, CAT_r_nb, K_eff, mu)
    best_f = curr_f

    hist_len  = n_iter // 100 + 1
    hist_w    = np.zeros(hist_len, dtype=np.float64)
    hi        = 0

    deltas = np.array([-2, -1, 1, 2], dtype=np.int64)

    for it in range(n_iter):
        T   = T0 * (Tend / T0) ** (float(it) / float(n_iter - 1))
        new = idx.copy()
        m_i  = int(np.random.randint(0, nm))
        delt = deltas[int(np.random.randint(0, 4))]
        v    = new[m_i] + delt
        if v < 0:        v = 0
        if v >= nm * 0:  pass   # lower bound already applied
        if v >= CAT_A_nb.shape[0]: v = CAT_A_nb.shape[0] - 1
        new[m_i] = v

        new_f, _ = _eval_nb(new, nodes, conn, bc_dof, loads, L_arr,
                             CAT_A_nb, CAT_r_nb, K_eff, mu)
        dE = new_f - curr_f
        accept = dE < 0.0
        if not accept and T > 1e-10:
            p = math.exp(-dE / T)
            accept = np.random.random() < p

        if accept:
            for j in range(nm):
                idx[j] = new[j]
            curr_f = new_f
            if curr_f < best_f:
                best_f = curr_f
                for j in range(nm):
                    best_idx[j] = idx[j]

        if it % 100 == 0 and hi < hist_len:
            # weight of best (no penalty)
            A_arr = np.empty(nm, dtype=np.float64)
            for i in range(nm):
                k = best_idx[i]
                if k < 0: k = 0
                if k >= CAT_A_nb.shape[0]: k = CAT_A_nb.shape[0] - 1
                A_arr[i] = CAT_A_nb[k]
            hist_w[hi] = _weight_nb(A_arr, L_arr)
            hi += 1

    return best_idx, hist_w[:hi]


@njit(cache=_CACHE)
def _aco_prob_nb(tau:    float64[:, :],
                 eta:    float64[:],
                 alpha:  float64,
                 beta:   float64,
                 n_mem:  int64,
                 n_cat:  int64) -> int64[:]:
    """One ACO ant — probabilistic section selection for each member."""
    idx = np.zeros(n_mem, dtype=np.int64)
    for m in range(n_mem):
        num = np.empty(n_cat, dtype=np.float64)
        tot = 0.0
        for k in range(n_cat):
            v = (tau[m, k] ** alpha) * (eta[k] ** beta)
            num[k] = v
            tot   += v
        r = np.random.random() * tot
        cumsum = 0.0
        chosen = n_cat - 1
        for k in range(n_cat):
            cumsum += num[k]
            if cumsum >= r:
                chosen = k
                break
        idx[m] = chosen
    return idx


# Numba infers all return types automatically from the function body.
# No explicit return type annotations needed on @njit functions.


# ══════════════════════════════════════════════════════════════════
# Python-level IS 800:2007 helpers  (thin wrappers around JIT kernels)
# ══════════════════════════════════════════════════════════════════

def fcd_is800(KL_r: float, alpha: float = 0.49) -> float:
    """Design compressive stress fcd (Pa) — calls JIT kernel."""
    return float(_fcd_nb(float(KL_r)))


def fallow_member(F: float, A: float, KL: float, r: float) -> float:
    """Allowable stress (Pa) for a member."""
    if F >= 0:
        return fy / gm0
    KL_r = KL / r if r > 0 else 1e6
    return fcd_is800(KL_r)


def compute_penalty(forces, A_arr, L_arr, r_arr,
                    KL_factor: float = 1.0, mu: float = 1e9) -> float:
    """Python wrapper — calls JIT _penalty_nb."""
    return float(_penalty_nb(
        np.asarray(forces,  dtype=np.float64),
        np.asarray(A_arr,   dtype=np.float64),
        np.asarray(L_arr,   dtype=np.float64),
        np.asarray(r_arr,   dtype=np.float64),
        float(KL_factor),
        float(mu),
    ))


# ══════════════════════════════════════════════════════════════════
# TrussModel — data container + Python-level DSM wrapper
# ══════════════════════════════════════════════════════════════════

@dataclass
class TrussModel:
    nodes:   np.ndarray   # (n_nodes, 3) float64
    conn:    np.ndarray   # (n_mem,  2)  int64
    bc_dof:  List[int]    # restrained global DOF indices
    loads:   np.ndarray   # (n_dof,)     float64
    K_eff:   float = 1.0

    def __post_init__(self):
        # Ensure correct dtypes for Numba
        self.nodes  = np.asarray(self.nodes,  dtype=np.float64)
        self.conn   = np.asarray(self.conn,   dtype=np.int64)
        self.loads  = np.asarray(self.loads,  dtype=np.float64)
        self._bc_nb = np.asarray(self.bc_dof, dtype=np.int64)
        self._L     = _member_lengths_nb(self.nodes, self.conn)

    def n_nodes(self):  return len(self.nodes)
    def n_mem(self):    return len(self.conn)
    def n_dof(self):    return 3 * self.n_nodes()

    def member_lengths(self) -> np.ndarray:
        return self._L.copy()

    def member_length(self, m: int) -> float:
        return float(self._L[m])

    def assemble_and_solve(self, A_arr: np.ndarray):
        """Returns (U, forces, max_disp)."""
        A_f64 = np.asarray(A_arr, dtype=np.float64)
        return _dsm_solve_nb(self.nodes, self.conn,
                             self._bc_nb, self.loads, A_f64)


# ══════════════════════════════════════════════════════════════════
# Catalogue helper functions
# ══════════════════════════════════════════════════════════════════

def nearest_catalog(A_m2: float) -> int:
    return int(np.argmin(np.abs(CAT_A - A_m2)))


def smallest_feasible_catalog(F: float, L: float, K: float = 1.0) -> int:
    """Smallest SP 6(1) index satisfying stress + slenderness."""
    for idx in range(N_CAT):
        A = CAT_A[idx]; r = CAT_r[idx]
        if A <= 0:
            continue
        fa    = fallow_member(F, A, K * L, r)
        sigma = abs(F) / A
        KLr   = K * L / r
        lim   = KL_lim_comp if F < 0 else KL_lim_tens
        if sigma <= fa and KLr <= lim:
            return idx
    return N_CAT - 1


# ══════════════════════════════════════════════════════════════════
# Fitness & compliance wrappers
# ══════════════════════════════════════════════════════════════════

def evaluate(truss: TrussModel, cat_idx: np.ndarray,
             mu: float = 1e9) -> Tuple[float, bool]:
    cat_idx = np.clip(np.asarray(cat_idx, dtype=np.int64), 0, N_CAT - 1)
    f, feas = _eval_nb(cat_idx, truss.nodes, truss.conn,
                       truss._bc_nb, truss.loads, truss._L,
                       CAT_A, CAT_r, truss.K_eff, mu)
    return float(f), bool(feas)


def weight_only(truss: TrussModel, cat_idx: np.ndarray) -> float:
    cat_idx = np.clip(np.asarray(cat_idx, dtype=np.int64), 0, N_CAT - 1)
    A_arr   = CAT_A[cat_idx]
    return float(_weight_nb(A_arr, truss._L))


def is800_compliant(truss: TrussModel,
                    cat_idx: np.ndarray) -> Tuple[bool, List[dict]]:
    cat_idx = np.clip(np.asarray(cat_idx, dtype=np.int64), 0, N_CAT - 1)
    A_arr   = CAT_A[cat_idx]
    r_arr   = CAT_r[cat_idx]
    L_arr   = truss._L
    _, forces, _ = truss.assemble_and_solve(A_arr)

    results_list = []
    ok = True
    for m in range(truss.n_mem()):
        F     = forces[m]
        A     = A_arr[m]; r = r_arr[m]; L = L_arr[m]
        sigma = abs(F) / A if A > 0 else 1e12
        fa    = fallow_member(F, A, truss.K_eff * L, r)
        KLr   = truss.K_eff * L / r if r > 0 else 1e6
        lim   = KL_lim_comp if F < 0 else KL_lim_tens
        dcr   = sigma / fa
        sl_ok = KLr <= lim
        results_list.append(dict(
            m=m, F=F, sigma=sigma, fa=fa, KLr=KLr,
            lim=lim, dcr=dcr, stress_ok=dcr <= 1.0, slend_ok=sl_ok
        ))
        if dcr > 1.0 or not sl_ok:
            ok = False
    return ok, results_list


# ══════════════════════════════════════════════════════════════════
# Result dataclass
# ══════════════════════════════════════════════════════════════════

@dataclass
class OptResult:
    method:   str
    weight:   float
    cat_idx:  np.ndarray
    is800_ok: bool
    runtime:  float
    history:  List[float] = field(default_factory=list)
    note:     str = ""
    forces:   Optional[np.ndarray] = None
    dcr_max:  float = 0.0


# ══════════════════════════════════════════════════════════════════
# ██  OPTIMISATION METHODS  ██
# ══════════════════════════════════════════════════════════════════

# ─── 1. FSD ───────────────────────────────────────────────────────
def opt_fsd(truss: TrussModel, max_iter: int = 40) -> OptResult:
    t0  = time.perf_counter()
    nm  = truss.n_mem()
    idx = np.full(nm, 23, dtype=np.int64)
    hist = []
    prev = None

    for _ in range(max_iter):
        A_arr = CAT_A[idx]
        _, forces, _ = truss.assemble_and_solve(A_arr)
        W = float(_weight_nb(A_arr, truss._L))
        hist.append(W)
        new_idx = _fsd_inner_nb(nm, idx, forces,
                                A_arr, CAT_r[idx], truss._L,
                                truss.K_eff, CAT_A, CAT_r, N_CAT)
        if prev is not None and np.array_equal(new_idx, prev):
            break
        prev = idx.copy()
        idx  = new_idx

    A_arr   = CAT_A[idx]
    W       = float(_weight_nb(A_arr, truss._L))
    _, forces, _ = truss.assemble_and_solve(A_arr)
    ok, res = is800_compliant(truss, idx)
    dcr_max = max(r['dcr'] for r in res)
    return OptResult("FSD", W, idx, ok, time.perf_counter() - t0,
                     hist, forces=forces, dcr_max=dcr_max)


# ─── 2. LP (iterative force-update) ──────────────────────────────
def opt_lp(truss: TrussModel) -> OptResult:
    t0    = time.perf_counter()
    nm    = truss.n_mem()
    idx   = np.full(nm, 23, dtype=np.int64)
    for _ in range(10):
        _, forces, _ = truss.assemble_and_solve(CAT_A[idx])
        new_idx = np.array([
            smallest_feasible_catalog(forces[m], truss._L[m], truss.K_eff)
            for m in range(nm)
        ], dtype=np.int64)
        if np.array_equal(new_idx, idx):
            break
        idx = new_idx
    W = float(_weight_nb(CAT_A[idx], truss._L))
    _, forces2, _ = truss.assemble_and_solve(CAT_A[idx])
    ok, results = is800_compliant(truss, idx)
    return OptResult("LP", W, idx, ok, time.perf_counter() - t0,
                     [W], forces=forces2,
                     dcr_max=max(r['dcr'] for r in results))


# ─── 3. SLP ───────────────────────────────────────────────────────
def opt_slp(truss: TrussModel, max_iter: int = 15) -> OptResult:
    t0    = time.perf_counter()
    nm    = truss.n_mem()
    idx   = np.full(nm, 23, dtype=np.int64)
    hist  = []
    for _ in range(max_iter):
        _, forces, _ = truss.assemble_and_solve(CAT_A[idx])
        c      = rho * truss._L
        A_lb   = np.array([smallest_feasible_catalog(forces[m], truss._L[m],
                           truss.K_eff) for m in range(nm)])
        A_lb_v = CAT_A[A_lb]
        res_lp = linprog(c, bounds=list(zip(A_lb_v, [CAT_A[-1]] * nm)),
                         method='highs')
        A_cont  = res_lp.x if res_lp.success else A_lb_v
        new_idx = np.array([nearest_catalog(a) for a in A_cont], dtype=np.int64)
        W = float(_weight_nb(CAT_A[new_idx], truss._L))
        hist.append(W)
        if np.array_equal(new_idx, idx):
            break
        idx = new_idx
    ok, results = is800_compliant(truss, idx)
    _, f2, _ = truss.assemble_and_solve(CAT_A[idx])
    return OptResult("SLP",
                     float(_weight_nb(CAT_A[idx], truss._L)),
                     idx, ok, time.perf_counter() - t0,
                     hist, forces=f2,
                     dcr_max=max(r['dcr'] for r in results))


# ─── 4. SQP ───────────────────────────────────────────────────────
def opt_sqp(truss: TrussModel) -> OptResult:
    t0   = time.perf_counter()
    nm   = truss.n_mem()
    hist = []

    def obj(A):
        return rho * float(np.dot(A, truss._L))

    def cons_fun(A):
        _, forces, _ = truss.assemble_and_solve(A)
        c = []
        for m in range(nm):
            fa = fallow_member(forces[m], A[m],
                               truss.K_eff * truss._L[m],
                               CAT_r[nearest_catalog(A[m])])
            c.append(fa - abs(forces[m]) / A[m])
        return np.array(c)

    x0     = CAT_A[np.full(nm, 23)]
    bounds = [(CAT_A[0], CAT_A[-1])] * nm
    res    = minimize(obj, x0, method='SLSQP', bounds=bounds,
                      constraints={'type': 'ineq', 'fun': cons_fun},
                      options={'maxiter': 200, 'ftol': 1e-6},
                      callback=lambda x: hist.append(obj(x)))

    new_idx = np.array([nearest_catalog(a) for a in res.x], dtype=np.int64)
    for m in range(nm):
        _, ftmp, _ = truss.assemble_and_solve(CAT_A[new_idx])
        new_idx[m] = max(int(new_idx[m]),
                         smallest_feasible_catalog(ftmp[m], truss._L[m],
                                                    truss.K_eff))
    W  = float(_weight_nb(CAT_A[new_idx], truss._L))
    ok, results = is800_compliant(truss, new_idx)
    _, f2, _ = truss.assemble_and_solve(CAT_A[new_idx])
    if not hist:
        hist = [W]
    return OptResult("SQP", W, new_idx, ok, time.perf_counter() - t0,
                     hist, forces=f2,
                     dcr_max=max(r['dcr'] for r in results))


# ─── 5. NLP (Nelder-Mead + restarts) ─────────────────────────────
def opt_nlp(truss: TrussModel, restarts: int = 2) -> OptResult:
    t0   = time.perf_counter()
    nm   = truss.n_mem()
    hist = []
    best_W, best_x = np.inf, None

    def obj(A):
        A    = np.clip(A, CAT_A[0], CAT_A[-1])
        _, forces, _ = truss.assemble_and_solve(A)
        W    = rho * float(np.dot(A, truss._L))
        r_arr = np.array([CAT_r[nearest_catalog(a)] for a in A])
        pen  = float(_penalty_nb(forces, A, truss._L, r_arr,
                                  truss.K_eff, 1e9))
        return W + pen

    for _ in range(restarts):
        x0  = CAT_A[np.random.randint(15, 35, nm)]
        res = minimize(obj, x0, method='Nelder-Mead',
                       options={'maxiter': 1000, 'xatol': 1e-4, 'fatol': 1e-4})
        if res.fun < best_W:
            best_W, best_x = res.fun, res.x
        hist.append(best_W)

    new_idx = np.array([nearest_catalog(a) for a in best_x], dtype=np.int64)
    W       = float(_weight_nb(CAT_A[new_idx], truss._L))
    ok, results = is800_compliant(truss, new_idx)
    _, f2, _ = truss.assemble_and_solve(CAT_A[new_idx])
    return OptResult("NLP", W, new_idx, ok, time.perf_counter() - t0,
                     hist, forces=f2,
                     dcr_max=max(r['dcr'] for r in results))


# ─── 6. MILP ──────────────────────────────────────────────────────
def opt_milp(truss: TrussModel) -> OptResult:
    t0  = time.perf_counter()
    nm  = truss.n_mem()
    idx = np.full(nm, 23, dtype=np.int64)
    for _ in range(8):
        _, forces, _ = truss.assemble_and_solve(CAT_A[idx])
        new_idx = np.array([
            smallest_feasible_catalog(forces[m], truss._L[m], truss.K_eff)
            for m in range(nm)
        ], dtype=np.int64)
        if np.array_equal(new_idx, idx):
            break
        idx = new_idx
    W = float(_weight_nb(CAT_A[idx], truss._L))
    _, f2, _ = truss.assemble_and_solve(CAT_A[idx])
    ok, results = is800_compliant(truss, idx)
    return OptResult("MILP", W, idx, ok, time.perf_counter() - t0,
                     [W], forces=f2,
                     dcr_max=max(r['dcr'] for r in results))


# ─── 7. MINLP (DE with integrality) ──────────────────────────────
def opt_minlp(truss: TrussModel,
              popsize: int = 8, maxiter: int = 80,
              seed: Optional[int] = None) -> OptResult:
    run_seed, _ = _prepare_seed(seed)
    t0   = time.perf_counter()
    nm   = truss.n_mem()
    hist = []

    def obj(x):
        idx   = np.clip(x.astype(int), 0, N_CAT - 1).astype(np.int64)
        A_arr = CAT_A[idx]; r_arr = CAT_r[idx]
        _, forces, _ = truss.assemble_and_solve(A_arr)
        W   = float(_weight_nb(A_arr, truss._L))
        pen = float(_penalty_nb(forces, A_arr, truss._L, r_arr,
                                 truss.K_eff, 1e9))
        return W + pen

    bounds = [(0, N_CAT - 1)] * nm
    res    = differential_evolution(
        obj, bounds, maxiter=maxiter, popsize=popsize,
        integrality=np.ones(nm, dtype=int),
        seed=run_seed, tol=1e-4, polish=False,
        callback=lambda xk, convergence: hist.append(obj(xk))
    )
    new_idx = np.clip(res.x.astype(int), 0, N_CAT - 1).astype(np.int64)
    W       = float(_weight_nb(CAT_A[new_idx], truss._L))
    ok, results = is800_compliant(truss, new_idx)
    _, f2, _ = truss.assemble_and_solve(CAT_A[new_idx])
    if not hist:
        hist = [W]
    return OptResult("MINLP", W, new_idx, ok, time.perf_counter() - t0,
                     hist, forces=f2,
                     dcr_max=max(r['dcr'] for r in results))


# ─── 8. Simulated Annealing ───────────────────────────────────────
def opt_sa(truss: TrussModel,
           T0: float = 500.0, Tend: float = 0.1,
           n_iter: int = 3000,
           seed: Optional[int] = None) -> OptResult:
    run_seed, _ = _prepare_seed(seed)
    t0  = time.perf_counter()
    nm  = truss.n_mem()
    idx0 = np.full(nm, 23, dtype=np.int64)

    # ── JIT SA loop ──
    best_idx, hist_w = _sa_inner_nb(
        idx0, truss.nodes, truss.conn, truss._bc_nb, truss.loads,
        truss._L, CAT_A, CAT_r, truss.K_eff, 1e9,
        float(T0), float(Tend), int(n_iter), run_seed
    )
    best_idx = np.asarray(best_idx, dtype=np.int64)
    hist     = list(hist_w)

    W  = float(_weight_nb(CAT_A[best_idx], truss._L))
    ok, res = is800_compliant(truss, best_idx)
    _, f2, _ = truss.assemble_and_solve(CAT_A[best_idx])
    return OptResult("SA", W, best_idx, ok, time.perf_counter() - t0,
                     hist, forces=f2,
                     dcr_max=max(r['dcr'] for r in res))


# ─── 9. Genetic Algorithm ─────────────────────────────────────────
def opt_ga(truss: TrussModel,
           npop: int = 40, ngen: int = 80,
           pc: float = 0.8, pm: float = 0.05,
           seed: Optional[int] = None) -> OptResult:
    run_seed, rng = _prepare_seed(seed)
    t0   = time.perf_counter()
    nm   = truss.n_mem()
    hist = []

    pop = np.random.randint(0, N_CAT, (npop, nm), dtype=np.int64)  # ← 2D array

    best_f = np.inf
    best_idx = pop[0].copy()

    for gen in range(ngen):
        # ── Parallel batch fitness ──
        fitness = _ga_fitness_nb(
            pop, truss.nodes, truss.conn, truss._bc_nb, truss.loads,
            truss._L, CAT_A, CAT_r, truss.K_eff, 1e9
        )
        fb = float(np.min(fitness))
        if fb < best_f:
            best_f   = fb
            best_idx = pop[int(np.argmin(fitness))].copy()

        hist.append(float(_weight_nb(CAT_A[best_idx], truss._L)))

        # Tournament selection
        ai = np.random.randint(0, npop, npop)
        bi = np.random.randint(0, npop, npop)
        sel = np.where(fitness[ai] < fitness[bi], ai, bi)
        new_pop = pop[sel].copy()

        # Crossover (two-point)
        for k in range(0, npop - 1, 2):
            if rng.random() < pc:
                pt = rng.randint(1, nm - 1)
                tmp = new_pop[k, pt:].copy()
                new_pop[k, pt:]     = new_pop[k + 1, pt:]
                new_pop[k + 1, pt:] = tmp

        # Mutation
        mut_mask = np.random.rand(npop, nm) < pm
        new_vals = np.random.randint(0, N_CAT, (npop, nm), dtype=np.int64)
        new_pop  = np.where(mut_mask, new_vals, new_pop).astype(np.int64)

        pop = new_pop

    W  = float(_weight_nb(CAT_A[best_idx], truss._L))
    ok, res = is800_compliant(truss, best_idx)
    _, f2, _ = truss.assemble_and_solve(CAT_A[best_idx])
    return OptResult("GA", W, best_idx, ok, time.perf_counter() - t0,
                     hist, forces=f2,
                     dcr_max=max(r['dcr'] for r in res))


# ─── 10. PSO ──────────────────────────────────────────────────────
def opt_pso(truss: TrussModel, np_: int = 30, n_iter: int = 100,
            seed: Optional[int] = None) -> OptResult:
    run_seed, _ = _prepare_seed(seed)
    t0   = time.perf_counter()
    nm   = truss.n_mem()
    hist = []
    w0, w1, c1, c2 = 0.9, 0.4, 2.0, 2.0

    pos     = np.random.randint(0, N_CAT, (np_, nm)).astype(np.float64)
    vel     = np.zeros((np_, nm))
    pbest   = pos.copy()
    pbest_f = _ga_fitness_nb(
        pos.astype(np.int64), truss.nodes, truss.conn,
        truss._bc_nb, truss.loads, truss._L, CAT_A, CAT_r,
        truss.K_eff, 1e9
    )
    gbest   = pbest[int(np.argmin(pbest_f))].copy()
    gbest_f = float(np.min(pbest_f))

    for it in range(n_iter):
        w   = w0 - (w0 - w1) * it / (n_iter - 1)
        r1  = np.random.rand(np_, nm)
        r2  = np.random.rand(np_, nm)
        vel = w * vel + c1 * r1 * (pbest - pos) + c2 * r2 * (gbest - pos)
        pos = np.clip(np.round(pos + vel), 0, N_CAT - 1)

        fs = _ga_fitness_nb(
            pos.astype(np.int64), truss.nodes, truss.conn,
            truss._bc_nb, truss.loads, truss._L, CAT_A, CAT_r,
            truss.K_eff, 1e9
        )
        improve = fs < pbest_f
        pbest_f = np.where(improve, fs, pbest_f)
        pbest   = np.where(improve[:, np.newaxis], pos, pbest)

        gmin = int(np.argmin(pbest_f))
        if pbest_f[gmin] < gbest_f:
            gbest_f = float(pbest_f[gmin])
            gbest   = pbest[gmin].copy()

        if it % 10 == 0:
            hist.append(float(_weight_nb(CAT_A[gbest.astype(int)], truss._L)))

    best_idx = gbest.astype(np.int64)
    W  = float(_weight_nb(CAT_A[best_idx], truss._L))
    ok, res = is800_compliant(truss, best_idx)
    _, f2, _ = truss.assemble_and_solve(CAT_A[best_idx])
    return OptResult("PSO", W, best_idx, ok, time.perf_counter() - t0,
                     hist, forces=f2,
                     dcr_max=max(r['dcr'] for r in res))


# ─── 11. ACO ──────────────────────────────────────────────────────
def opt_aco(truss: TrussModel,
            n_ants: int = 20, n_iter: int = 80,
            alpha: float = 1.0, beta: float = 2.0,
            rho_e: float = 0.4, Q: float = 100.0,
            seed: Optional[int] = None) -> OptResult:
    run_seed, _ = _prepare_seed(seed)
    t0   = time.perf_counter()
    nm   = truss.n_mem()
    hist = []

    tau      = np.ones((nm, N_CAT), dtype=np.float64)
    eta      = (1.0 / (CAT_w + 1e-6)).astype(np.float64)
    best_f   = np.inf
    best_idx = np.zeros(nm, dtype=np.int64)

    for it in range(n_iter):
        ant_idxs = []
        ant_fs   = []
        for _ in range(n_ants):
            idx   = _aco_prob_nb(tau, eta, alpha, beta, nm, N_CAT)
            f, _  = _eval_nb(idx, truss.nodes, truss.conn,
                              truss._bc_nb, truss.loads, truss._L,
                              CAT_A, CAT_r, truss.K_eff, 1e9)
            ant_idxs.append(idx)
            ant_fs.append(f)
            if f < best_f:
                best_f   = f
                best_idx = idx.copy()

        tau *= (1 - rho_e)
        for idx, f in zip(ant_idxs, ant_fs):
            if f < 2 * best_f:
                for m in range(nm):
                    tau[m, idx[m]] += Q / f

        if it % 8 == 0:
            hist.append(float(_weight_nb(CAT_A[best_idx], truss._L)))

    W  = float(_weight_nb(CAT_A[best_idx], truss._L))
    ok, res = is800_compliant(truss, best_idx)
    _, f2, _ = truss.assemble_and_solve(CAT_A[best_idx])
    return OptResult("ACO", W, best_idx, ok, time.perf_counter() - t0,
                     hist, forces=f2,
                     dcr_max=max(r['dcr'] for r in res))


# ─── 12. BB-BC ────────────────────────────────────────────────────
def opt_bbbc(truss: TrussModel,
             np_: int = 30, n_iter: int = 80,
             seed: Optional[int] = None) -> OptResult:
    run_seed, _ = _prepare_seed(seed)
    t0   = time.perf_counter()
    nm   = truss.n_mem()
    hist = []

    pop  = np.random.randint(0, N_CAT, (np_, nm), dtype=np.int64)
    fs   = _bbbc_fitness_nb(
        pop, truss.nodes, truss.conn, truss._bc_nb, truss.loads,
        truss._L, CAT_A, CAT_r, truss.K_eff, 1e9
    )
    best_f   = float(np.min(fs))
    best_idx = pop[int(np.argmin(fs))].copy()

    for it in range(1, n_iter + 1):
        w_inv = 1.0 / (fs.astype(np.float64) + 1e-9)
        xcm   = np.clip(
            np.round(np.dot(w_inv, pop.astype(np.float64)) / w_inv.sum()),
            0, N_CAT - 1
        ).astype(np.int64)

        alpha_r = N_CAT / it
        noise   = np.clip(
            np.round(xcm + alpha_r * np.random.randn(np_, nm)).astype(int),
            0, N_CAT - 1
        ).astype(np.int64)
        pop     = noise
        fs      = _bbbc_fitness_nb(
            pop, truss.nodes, truss.conn, truss._bc_nb, truss.loads,
            truss._L, CAT_A, CAT_r, truss.K_eff, 1e9
        )
        # Elitism
        pop[0]  = xcm
        f0, _   = _eval_nb(xcm, truss.nodes, truss.conn,
                            truss._bc_nb, truss.loads, truss._L,
                            CAT_A, CAT_r, truss.K_eff, 1e9)
        fs[0]   = f0

        gmin = int(np.argmin(fs))
        if float(fs[gmin]) < best_f:
            best_f   = float(fs[gmin])
            best_idx = pop[gmin].copy()

        if it % 8 == 0:
            hist.append(float(_weight_nb(CAT_A[best_idx], truss._L)))

    W  = float(_weight_nb(CAT_A[best_idx], truss._L))
    ok, res = is800_compliant(truss, best_idx)
    _, f2, _ = truss.assemble_and_solve(CAT_A[best_idx])
    return OptResult("BB-BC", W, best_idx, ok, time.perf_counter() - t0,
                     hist, forces=f2,
                     dcr_max=max(r['dcr'] for r in res))


# ─── 13. DE ───────────────────────────────────────────────────────
def opt_de(truss: TrussModel,
           popsize: int = 10, maxiter: int = 100,
           seed: Optional[int] = None) -> OptResult:
    run_seed, _ = _prepare_seed(seed)
    t0   = time.perf_counter()
    nm   = truss.n_mem()
    hist = []

    def obj(x):
        idx   = np.clip(x.astype(int), 0, N_CAT - 1).astype(np.int64)
        A_arr = CAT_A[idx]; r_arr = CAT_r[idx]
        _, forces, _ = truss.assemble_and_solve(A_arr)
        W   = float(_weight_nb(A_arr, truss._L))
        pen = float(_penalty_nb(forces, A_arr, truss._L, r_arr,
                                 truss.K_eff, 1e9))
        return W + pen

    bounds = [(0, N_CAT - 1)] * nm
    res    = differential_evolution(
        obj, bounds, maxiter=maxiter, popsize=popsize,
        integrality=np.ones(nm, dtype=int),
        seed=run_seed, tol=1e-4, polish=False,
        callback=lambda xk, convergence: hist.append(obj(xk))
    )
    best_idx = np.clip(res.x.astype(int), 0, N_CAT - 1).astype(np.int64)
    W        = float(_weight_nb(CAT_A[best_idx], truss._L))
    ok, res_c = is800_compliant(truss, best_idx)
    _, f2, _ = truss.assemble_and_solve(CAT_A[best_idx])
    if not hist:
        hist = [W]
    return OptResult("DE", W, best_idx, ok, time.perf_counter() - t0,
                     hist, forces=f2,
                     dcr_max=max(r['dcr'] for r in res_c))


# ─── 14. GA-MINLP (proposed hybrid) ──────────────────────────────
def opt_ga_minlp(truss: TrussModel,
                 npop: int = 50, ngen: int = 80,
                 top_k: int = 5, de_iter: int = 60,
                 seed: Optional[int] = None) -> OptResult:
    run_seed, rng = _prepare_seed(seed)
    t0   = time.perf_counter()
    nm   = truss.n_mem()
    hist = []

    # ── Phase 1: GA (parallel population fitness) ──────────────
    pop = np.random.randint(0, N_CAT, (npop, nm), dtype=np.int64)
    best_f = np.inf
    best_idx = pop[0].copy()
    ga_hist  = []

    for gen in range(ngen):
        fitness = _ga_fitness_nb(
            pop, truss.nodes, truss.conn, truss._bc_nb, truss.loads,
            truss._L, CAT_A, CAT_r, truss.K_eff, 1e9
        )
        fb = float(np.min(fitness))
        if fb < best_f:
            best_f   = fb
            best_idx = pop[int(np.argmin(fitness))].copy()
        ga_hist.append(float(_weight_nb(CAT_A[best_idx], truss._L)))

        ai = np.random.randint(0, npop, npop)
        bi = np.random.randint(0, npop, npop)
        sel     = np.where(fitness[ai] < fitness[bi], ai, bi)
        new_pop = pop[sel].copy()

        for k in range(0, npop - 1, 2):
            if rng.random() < 0.8:
                pt = rng.randint(1, nm - 1)
                tmp = new_pop[k, pt:].copy()
                new_pop[k, pt:]     = new_pop[k + 1, pt:]
                new_pop[k + 1, pt:] = tmp

        mut_mask = np.random.rand(npop, nm) < 0.04
        new_vals = np.random.randint(0, N_CAT, (npop, nm), dtype=np.int64)
        pop      = np.where(mut_mask, new_vals, new_pop).astype(np.int64)

    hist.extend(ga_hist)

    # Top-K elites
    final_fit = _ga_fitness_nb(
        pop, truss.nodes, truss.conn, truss._bc_nb, truss.loads,
        truss._L, CAT_A, CAT_r, truss.K_eff, 1e9
    )
    order  = np.argsort(final_fit)[:top_k]
    elites = [pop[i].copy() for i in order]

    # ── Phase 2: MINLP with exact IS 800 (no penalty approximation) ──
    def obj_minlp(x):
        idx   = np.clip(x.astype(int), 0, N_CAT - 1).astype(np.int64)
        A_arr = CAT_A[idx]; r_arr = CAT_r[idx]
        _, forces, _ = truss.assemble_and_solve(A_arr)
        W   = float(_weight_nb(A_arr, truss._L))
        # Exact Annex D — no penalty multiplier
        pen = 0.0
        fa_t = fy / gm0
        for m in range(nm):
            F = forces[m]; A = A_arr[m]; r = r_arr[m]; L = truss._L[m]
            KLr = truss.K_eff * L / r if r > 0 else 1e6
            fa  = float(_fcd_nb(KLr)) if F < 0 else fa_t
            sig = abs(F) / A if A > 0 else 1e12
            lim = 180.0 if F < 0 else 400.0
            if sig > fa:
                pen += 1e10 * (sig / fa - 1) ** 2
            if KLr > lim:
                pen += 1e10 * (KLr / lim - 1) ** 2
        return W + pen

    p2_best_f = np.inf
    p2_best   = best_idx.copy()
    bounds    = [(0, N_CAT - 1)] * nm

    for elite_rank, elite in enumerate(elites):
        init_pop = np.clip(
            elite + np.random.randint(-3, 4, (6 * nm, nm)),
            0, N_CAT - 1
        ).astype(float)
        res = differential_evolution(
            obj_minlp, bounds, maxiter=de_iter, popsize=6,
            integrality=np.ones(nm, dtype=int),
            init=init_pop[:6 * nm],
            seed=(run_seed + 1000 + elite_rank) % (2**31 - 1),
            tol=1e-5, polish=False
        )
        if res.fun < p2_best_f:
            p2_best_f = res.fun
            p2_best   = np.clip(res.x.astype(int), 0, N_CAT - 1).astype(np.int64)

    hist.append(p2_best_f)
    W  = float(_weight_nb(CAT_A[p2_best], truss._L))
    ok, res_c = is800_compliant(truss, p2_best)
    _, f2, _ = truss.assemble_and_solve(CAT_A[p2_best])
    return OptResult("GA-MINLP*", W, p2_best, ok, time.perf_counter() - t0,
                     hist, forces=f2,
                     dcr_max=max(r['dcr'] for r in res_c))


# ══════════════════════════════════════════════════════════════════
# Benchmark truss definitions
# ══════════════════════════════════════════════════════════════════

def make_tetrahedron() -> TrussModel:
    """6-bar, 4-node tetrahedral truss."""
    nodes = np.array([
        [0.0, 0.0, 0.0],
        [3.0, 0.0, 0.0],
        [1.5, 3.0, 0.0],
        [1.5, 1.5, 4.0],
    ], dtype=np.float64)
    conn  = np.array([[0,1],[1,2],[2,0],[0,3],[1,3],[2,3]], dtype=np.int64)
    bc    = [0, 1, 2, 4, 5, 8]
    loads = np.zeros(12, dtype=np.float64)
    loads[3*3+1] =  50e3
    loads[3*3+2] = -100e3
    return TrussModel(nodes, conn, bc, loads)


def make_25bar() -> TrussModel:
    """Classic ASCE 25-bar space truss."""
    nodes = np.array([
        [-0.9525,  0.9525, 3.048],
        [ 0.9525,  0.9525, 3.048],
        [ 0.9525, -0.9525, 3.048],
        [-0.9525, -0.9525, 3.048],
        [-1.905,   1.905,  1.524],
        [ 1.905,   1.905,  1.524],
        [ 1.905,  -1.905,  1.524],
        [-1.905,  -1.905,  1.524],
        [-1.905,   1.905,  0.0  ],
        [ 1.905,   1.905,  0.0  ],
        [ 1.905,  -1.905,  0.0  ],
        [-1.905,  -1.905,  0.0  ],
    ], dtype=np.float64)
    conn = np.array([
        [0,1],[1,2],[2,3],[3,0],
        [0,2],[1,3],
        [0,4],[1,5],[2,6],[3,7],
        [4,5],[5,6],[6,7],[7,4],
        [4,6],[5,7],
        [4,8],[5,9],[6,10],[7,11],
        [8,9],[9,10],[10,11],[11,8],
        [8,10],
    ], dtype=np.int64)
    bc = []
    for n in range(4, 12):
        bc += [3*n, 3*n+1, 3*n+2]
    loads = np.zeros(36, dtype=np.float64)
    loads[3*0+0] =  4.45e3;  loads[3*0+1] = -44.5e3; loads[3*0+2] = -22.25e3
    loads[3*1+1] = -44.5e3;  loads[3*1+2] = -22.25e3
    loads[3*2+0] =  2.225e3; loads[3*2+1] = -44.5e3; loads[3*2+2] = -22.25e3
    loads[3*3+1] = -44.5e3;  loads[3*3+2] = -22.25e3
    return TrussModel(nodes, conn, bc, loads)


def make_72bar() -> TrussModel:
    """Classic 72-bar cantilevered double-layer space truss."""
    h = [0.0, 1.2192, 2.4384, 3.6576]
    a = 0.9144

    def layer_nodes(z):
        return [[a,a,z],[-a,a,z],[-a,-a,z],[a,-a,z]]

    nl = []
    for z in h:
        nl.extend(layer_nodes(z))
    nl += [[0.0,0.0,h[-1]], [a,0.0,h[-1]], [0.0,a,h[-1]], [-a,0.0,h[-1]]]
    nodes   = np.array(nl, dtype=np.float64)
    n_nodes = len(nodes)

    conn_list = []
    for layer in range(4):
        base = layer * 4
        for i in range(4):
            conn_list.append([base+i, base+(i+1)%4])
        conn_list.append([base+0, base+2])
        conn_list.append([base+1, base+3])
    for layer in range(3):
        bot = layer*4; top = (layer+1)*4
        for i in range(4):
            conn_list.append([bot+i, top+i])
            conn_list.append([bot+i, top+(i+1)%4])
    top = 12
    for i in range(4):
        conn_list.append([top+i, 16])
        conn_list.append([top+i, 17])

    conn  = np.array(conn_list[:72], dtype=np.int64)
    bc    = list(range(12))
    loads = np.zeros(3*n_nodes, dtype=np.float64)
    for n in range(12, 20):
        loads[3*n+0] =  2.225e3
        loads[3*n+2] = -4.45e3
    return TrussModel(nodes, conn, bc, loads)


TRUSSES = {
    "6-Bar Tetrahedron":  make_tetrahedron,
    "25-Bar Space Truss": make_25bar,
    "72-Bar Space Truss": make_72bar,
}

ALL_METHODS = [
    ("FSD",       opt_fsd),
    ("LP",        opt_lp),
    ("SLP",       opt_slp),
    ("SQP",       opt_sqp),
    ("NLP",       opt_nlp),
    ("MILP",      opt_milp),
    ("MINLP",     opt_minlp),
    ("SA",        opt_sa),
    ("GA",        opt_ga),
    ("PSO",       opt_pso),
    ("ACO",       opt_aco),
    ("BB-BC",     opt_bbbc),
    ("DE",        opt_de),
    ("GA-MINLP*", opt_ga_minlp),
]


# ══════════════════════════════════════════════════════════════════
# JIT warm-up  (call once at import time so first real run is fast)
# ══════════════════════════════════════════════════════════════════
def warmup_jit():
    """
    Trigger AOT compilation of all JIT kernels on a tiny 2-member truss.
    Call once per session (or rely on cache for subsequent runs).
    """
    _nodes = np.array([[0.,0.,0.],[1.,0.,0.],[0.,1.,0.]], dtype=np.float64)
    _conn  = np.array([[0,1],[1,2]], dtype=np.int64)
    _bc    = np.array([0,1,2], dtype=np.int64)
    _loads = np.zeros(9, dtype=np.float64)
    _loads[6] = 1000.0
    _A     = np.array([CAT_A[10], CAT_A[10]], dtype=np.float64)
    _idx   = np.array([10, 10], dtype=np.int64)
    _L     = _member_lengths_nb(_nodes, _conn)
    _dsm_solve_nb(_nodes, _conn, _bc, _loads, _A)
    _eval_nb(_idx, _nodes, _conn, _bc, _loads, _L, CAT_A, CAT_r, 1.0, 1e9)
    _fcd_nb(100.0)
    _fcd_vec_nb(np.array([80., 120., 160.], dtype=np.float64))
    _fsd_inner_nb(2, _idx, np.array([5000., -3000.], np.float64),
                  _A, CAT_r[_idx], _L, 1.0, CAT_A, CAT_r, N_CAT)
    _pop2 = np.array([[10,15],[20,25]], dtype=np.int64)
    _ga_fitness_nb(_pop2, _nodes, _conn, _bc, _loads, _L, CAT_A, CAT_r, 1.0, 1e9)
    print("  [truss_engine] JIT warm-up complete.")
