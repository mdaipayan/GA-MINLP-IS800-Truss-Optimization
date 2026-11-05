"""
engine2.py  --  Benchmark definitions, linear-elastic analysis and IS 800:2007 checks
=====================================================================================

Version 2 of the analysis engine for Paper 1 (Engineering with Computers).

Main changes from truss_engine.py (v1):
  * 25-bar and 72-bar trusses rebuilt to the literature definitions
    (10-node ASCE 25-bar with 8 member groups; 4-storey 72-bar with 16 groups
    and two load cases).
  * Displacement limits and multiple load cases are supported.
  * Every model is checked for kinematic stability when it is built.
    A singular or near-singular stiffness matrix raises an error instead of
    silently returning zero forces (the v1 bug).
  * The SP 6(1) catalogue is sorted by area, so "one step up" always means
    "the next heavier section".
  * Every call to evaluate() counts as one structural analysis (NFE), so all
    optimizers can be given the same analysis budget.

Units: SI (N, m, Pa, kg).
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np
from scipy.linalg import cho_factor, cho_solve

# ----------------------------------------------------------------------------
# SP 6(1):1964 equal-angle catalogue  (label, A cm^2, r_min cm)
# ----------------------------------------------------------------------------
_SP6 = [
    ("ISA 20x20x3", 1.11, 0.609), ("ISA 25x25x3", 1.42, 0.762),
    ("ISA 25x25x4", 1.85, 0.748), ("ISA 30x30x3", 1.73, 0.922),
    ("ISA 30x30x4", 2.27, 0.906), ("ISA 35x35x4", 2.67, 1.065),
    ("ISA 35x35x5", 3.25, 1.048), ("ISA 40x40x4", 3.08, 1.225),
    ("ISA 40x40x5", 3.79, 1.208), ("ISA 45x45x4", 3.49, 1.384),
    ("ISA 45x45x5", 4.30, 1.366), ("ISA 45x45x6", 5.07, 1.349),
    ("ISA 50x50x5", 4.80, 1.526), ("ISA 50x50x6", 5.69, 1.508),
    ("ISA 55x55x5", 5.30, 1.686), ("ISA 55x55x6", 6.30, 1.667),
    ("ISA 60x60x5", 5.82, 1.846), ("ISA 60x60x6", 6.91, 1.826),
    ("ISA 60x60x8", 9.03, 1.789), ("ISA 65x65x6", 7.52, 1.987),
    ("ISA 65x65x8", 9.86, 1.950), ("ISA 70x70x6", 8.13, 2.147),
    ("ISA 70x70x7", 9.40, 2.129), ("ISA 75x75x6", 8.74, 2.306),
    ("ISA 75x75x8", 11.40, 2.270), ("ISA 75x75x10", 14.00, 2.234),
    ("ISA 80x80x6", 9.35, 2.466), ("ISA 80x80x8", 12.30, 2.428),
    ("ISA 80x80x10", 15.20, 2.391), ("ISA 90x90x6", 10.60, 2.785),
    ("ISA 90x90x8", 13.90, 2.747), ("ISA 90x90x10", 17.10, 2.709),
    ("ISA 100x100x6", 11.80, 3.100), ("ISA 100x100x8", 15.60, 3.060),
    ("ISA 100x100x10", 19.20, 3.021), ("ISA 100x100x12", 22.80, 2.983),
    ("ISA 110x110x8", 17.20, 3.380), ("ISA 110x110x10", 21.20, 3.340),
    ("ISA 110x110x12", 25.10, 3.300), ("ISA 120x120x8", 18.80, 3.699),
    ("ISA 120x120x10", 23.20, 3.659), ("ISA 120x120x12", 27.50, 3.618),
    ("ISA 130x130x10", 25.20, 3.977), ("ISA 130x130x12", 29.90, 3.937),
    ("ISA 150x150x10", 29.20, 4.617), ("ISA 150x150x12", 34.70, 4.575),
    ("ISA 150x150x15", 42.90, 4.512), ("ISA 200x200x16", 62.60, 6.153),
    ("ISA 200x200x20", 77.60, 6.091), ("ISA 200x200x25", 95.70, 6.010),
    ("ISA 200x200x32", 121.00, 5.910),
]
_order = sorted(range(len(_SP6)), key=lambda i: (_SP6[i][1], -_SP6[i][2]))
CAT_LABEL = [_SP6[i][0] for i in _order]
CAT_A = np.array([_SP6[i][1] for i in _order]) * 1e-4      # m^2
CAT_R = np.array([_SP6[i][2] for i in _order]) * 1e-2      # m
N_CAT = len(CAT_A)

# Material and IS 800:2007 constants
E = 200e9
RHO = 7850.0
FY = 250e6
GAMMA_M0 = 1.10
ALPHA_C = 0.49           # buckling class c (angles), IS 800 Table 7/10
SLEND_COMP = 180.0       # IS 800 Table 3
SLEND_TENS = 400.0

IN = 0.0254
KIP = 4448.2216


def fcd(kl_r):
    """IS 800:2007 Cl. 7.1.2.1 design compressive stress (Pa). Vectorised."""
    kl_r = np.asarray(kl_r, dtype=float)
    fcc = math.pi ** 2 * E / np.maximum(kl_r, 1e-9) ** 2
    lam = np.sqrt(FY / fcc)
    phi = 0.5 * (1.0 + ALPHA_C * (lam - 0.2) + lam ** 2)
    chi = np.minimum(1.0, 1.0 / (phi + np.sqrt(np.maximum(phi ** 2 - lam ** 2, 0.0))))
    return chi * FY / GAMMA_M0


class BudgetExhausted(Exception):
    """Raised when an optimizer exceeds its structural-analysis budget."""


@dataclass
class Result:
    method: str
    idx: np.ndarray              # catalogue index per group
    weight: float
    feasible: bool
    nfe: int                     # analyses used
    nfe_best: int                # analysis count when the final best was found
    history: List[tuple] = field(default_factory=list)   # (nfe, best feasible W)
    runtime: float = 0.0
    note: str = ""


class Problem:
    """Discrete sizing problem: one SP 6(1) section per member group."""

    def __init__(self, name, nodes, conn, bc_dof, load_cases, groups,
                 disp_limit=None, disp_dofs=None, K_eff=1.0):
        self.name = name
        self.nodes = np.asarray(nodes, float)
        self.conn = np.asarray(conn, int)
        self.n_mem = len(self.conn)
        self.groups = np.asarray(groups, int)          # member -> group
        self.n_grp = int(self.groups.max()) + 1
        self.K_eff = K_eff
        ndof = 3 * len(self.nodes)
        self.free = np.array([d for d in range(ndof) if d not in set(bc_dof)])
        self.P = np.column_stack([np.asarray(f, float)[self.free] for f in load_cases])
        self.n_lc = self.P.shape[1]

        d = self.nodes[self.conn[:, 1]] - self.nodes[self.conn[:, 0]]
        self.L = np.linalg.norm(d, axis=1)
        cosines = d / self.L[:, None]
        # Compatibility matrix B: elongation e = B u  (rows = members, cols = free DOF)
        Bfull = np.zeros((self.n_mem, ndof))
        for m, (i, j) in enumerate(self.conn):
            Bfull[m, 3 * i:3 * i + 3] = -cosines[m]
            Bfull[m, 3 * j:3 * j + 3] = cosines[m]
        self.B = Bfull[:, self.free]

        # Displacement-limited DOFs (positions inside the free-DOF vector)
        self.disp_limit = disp_limit
        if disp_limit is None:
            self.disp_pos = np.array([], int)
        else:
            dl = self.free if disp_dofs is None else np.asarray(disp_dofs)
            pos = {d: k for k, d in enumerate(self.free)}
            self.disp_pos = np.array([pos[d] for d in dl if d in pos], int)

        self.L_grp = np.bincount(self.groups, weights=self.L, minlength=self.n_grp)
        self.w_max = self.weight(np.full(self.n_grp, N_CAT - 1))
        self.nfe = 0
        self.budget = None
        self._check_stability()

    # ------------------------------------------------------------------
    def _check_stability(self):
        k = E * CAT_A[20] / self.L
        K = self.B.T @ (k[:, None] * self.B)
        ev = np.linalg.eigvalsh(K)
        ratio = ev[0] / ev[-1]
        if ratio < 1e-10:
            raise ValueError(f"{self.name}: stiffness matrix is singular "
                             f"(lambda_min/lambda_max = {ratio:.2e}); "
                             f"the model is a mechanism or has unconnected nodes.")
        self.stability_ratio = ratio

    def weight(self, idx):
        return float(RHO * np.dot(CAT_A[np.asarray(idx, int)], self.L_grp))

    def analyse(self, idx):
        """Linear-elastic analysis for all load cases. Not counted as NFE."""
        idx = np.asarray(idx, int)
        A = CAT_A[idx][self.groups]
        k = E * A / self.L
        K = self.B.T @ (k[:, None] * self.B)
        try:
            cf = cho_factor(K, check_finite=False)
        except np.linalg.LinAlgError as exc:
            raise ValueError(f"{self.name}: stiffness not positive definite") from exc
        U = cho_solve(cf, self.P, check_finite=False)       # (n_free, n_lc)
        F = k[:, None] * (self.B @ U)                        # (n_mem, n_lc)
        return U, F

    def constraints(self, idx):
        """Return dict of normalised constraint ratios (feasible if all <= 1)."""
        idx = np.asarray(idx, int)
        U, F = self.analyse(idx)
        A = CAT_A[idx][self.groups]
        r = CAT_R[idx][self.groups]
        klr = self.K_eff * self.L / r
        f_t = A * FY / GAMMA_M0
        f_c = A * fcd(klr)
        ten = np.max(np.maximum(F, 0.0), axis=1) / f_t
        com = np.max(np.maximum(-F, 0.0), axis=1) / f_c
        in_comp = np.any(F < 0.0, axis=1)
        sl = klr / np.where(in_comp, SLEND_COMP, SLEND_TENS)
        if len(self.disp_pos):
            dr = np.max(np.abs(U[self.disp_pos, :])) / self.disp_limit
        else:
            dr = 0.0
        return dict(stress=np.maximum(ten, com), slender=sl, disp=dr, F=F, U=U,
                    klr=klr)

    def evaluate(self, idx):
        """One structural analysis. Returns (weight, violation, feasible)."""
        if self.budget is not None and self.nfe >= self.budget:
            raise BudgetExhausted
        self.nfe += 1
        c = self.constraints(idx)
        viol = (np.sum(np.maximum(c["stress"] - 1.0, 0.0))
                + np.sum(np.maximum(c["slender"] - 1.0, 0.0))
                + max(c["disp"] - 1.0, 0.0))
        return self.weight(idx), float(viol), viol <= 1e-9

    def fitness(self, idx):
        """Deb's feasibility rule as one scalar: feasible designs always rank
        above infeasible ones; infeasible designs are ranked by violation.
        No penalty coefficient to tune, identical for every method."""
        W, v, ok = self.evaluate(idx)
        return (W if ok else self.w_max + 1e3 * v), W, ok

    def reset(self, budget=None):
        self.nfe = 0
        self.budget = budget

    def member_table(self, idx):
        """Per-member IS 800 check for reporting (worst load case)."""
        c = self.constraints(idx)
        rows = []
        for m in range(self.n_mem):
            g = self.groups[m]
            F = c["F"][m]
            Fg = F[np.argmax(np.abs(F))]
            rows.append(dict(member=m + 1, group=g + 1, section=CAT_LABEL[idx[g]],
                             A_cm2=CAT_A[idx[g]] * 1e4, r_mm=CAT_R[idx[g]] * 1e3,
                             F_kN=Fg / 1e3, klr=c["klr"][m],
                             DCR=c["stress"][m], slender_ratio=c["slender"][m]))
        return rows, c["disp"]


# ----------------------------------------------------------------------------
# Benchmarks
# ----------------------------------------------------------------------------
def make_6bar():
    """6-bar tetrahedron (unchanged from v1, used as regression check)."""
    nodes = [[0, 0, 0], [3, 0, 0], [1.5, 3, 0], [1.5, 1.5, 4]]
    conn = [[0, 1], [1, 2], [2, 0], [0, 3], [1, 3], [2, 3]]
    P = np.zeros(12); P[10] = 50e3; P[11] = -100e3
    return Problem("6-bar", nodes, conn, [0, 1, 2, 4, 5, 8], [P], range(6))


def make_25bar():
    """ASCE 25-bar transmission tower: 10 nodes, 25 members, 8 groups.
    Geometry and load case as used in the discrete benchmark of
    Camp & Bichon (2004)."""
    nodes = np.array([[-37.5, 0, 200], [37.5, 0, 200],
                      [-37.5, 37.5, 100], [37.5, 37.5, 100],
                      [37.5, -37.5, 100], [-37.5, -37.5, 100],
                      [-100, 100, 0], [100, 100, 0], [100, -100, 0],
                      [-100, -100, 0]]) * IN
    # (node numbers are the usual 1..10 minus one)
    conn = [[0, 1],
            [0, 3], [1, 2], [0, 4], [1, 5],
            [1, 3], [1, 4], [0, 2], [0, 5],
            [2, 5], [3, 4],
            [2, 3], [4, 5],
            [2, 9], [5, 6], [3, 8], [4, 7],
            [2, 7], [3, 6], [5, 8], [4, 9],
            [2, 6], [3, 7], [4, 8], [5, 9]]
    groups = [0] + [1] * 4 + [2] * 4 + [3] * 2 + [4] * 2 + [5] * 4 + [6] * 4 + [7] * 4
    bc = [d for n in range(6, 10) for d in (3 * n, 3 * n + 1, 3 * n + 2)]
    P = np.zeros(30)
    P[0], P[1], P[2] = 1.0 * KIP, -10.0 * KIP, -10.0 * KIP     # node 1
    P[4], P[5] = -10.0 * KIP, -10.0 * KIP                      # node 2
    P[6] = 0.5 * KIP                                           # node 3
    P[15] = 0.5 * KIP                                          # node 6
    return Problem("25-bar", nodes, conn, bc, [P], groups,
                   disp_limit=0.35 * IN)


def make_72bar():
    """Four-storey 72-bar space truss: 20 nodes, 72 members, 16 groups,
    two load cases, top-node x/y displacement limit 0.25 in."""
    s, hz = 120 * IN, 60 * IN
    nodes = [[x, y, lvl * hz] for lvl in range(5)
             for (x, y) in ((0, 0), (s, 0), (s, s), (0, s))]
    conn, groups = [], []
    for st in range(4):                      # storey 0 = bottom
        b, t = 4 * st, 4 * (st + 1)
        g0 = 4 * (3 - st)                    # groups numbered from the top storey
        for i in range(4):
            conn.append([b + i, t + i]); groups.append(g0 + 0)          # verticals
        for i in range(4):
            j = (i + 1) % 4
            conn += [[b + i, t + j], [b + j, t + i]]; groups += [g0 + 1] * 2   # face diagonals
        for i in range(4):
            conn.append([t + i, t + (i + 1) % 4]); groups.append(g0 + 2)  # horizontals
        conn += [[t + 0, t + 2], [t + 1, t + 3]]; groups += [g0 + 3] * 2  # plan diagonals
    bc = list(range(12))
    top = [16, 17, 18, 19]
    P1 = np.zeros(60); P1[3 * 16:3 * 16 + 3] = [5 * KIP, 5 * KIP, -5 * KIP]
    P2 = np.zeros(60)
    for n in top:
        P2[3 * n + 2] = -5 * KIP
    disp_dofs = [3 * n + a for n in top for a in (0, 1)]
    return Problem("72-bar", nodes, conn, bc, [P1, P2], groups,
                   disp_limit=0.25 * IN, disp_dofs=disp_dofs)


BENCHMARKS = {"6-bar": make_6bar, "25-bar": make_25bar, "72-bar": make_72bar}
