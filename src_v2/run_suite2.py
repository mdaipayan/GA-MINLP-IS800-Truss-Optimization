"""
run_suite2.py -- N = 20 seed-controlled runs, equal analysis budget per truss.
Writes ../results_v2/stats_v2.json
"""
import json
import sys
import time

import numpy as np

import engine2 as e
import optimizers2 as o

BUDGET = {"6-bar": 3000, "25-bar": 10000, "72-bar": 20000}
N_RUNS = 20


def main(out="../results_v2/stats_v2.json"):
    res = {"budget": BUDGET, "n_runs": N_RUNS, "trusses": {}}
    for tname, make in e.BENCHMARKS.items():
        p = make()
        T = res["trusses"][tname] = {"n_mem": p.n_mem, "n_grp": p.n_grp,
                                     "n_lc": p.n_lc, "methods": {}}
        for mname, f in o.STOCHASTIC.items():
            runs = []
            t0 = time.time()
            for s in range(N_RUNS):
                r = f(p, BUDGET[tname], s)
                runs.append(dict(seed=s, weight=r.weight, feasible=bool(r.feasible),
                                 nfe=r.nfe, nfe_best=r.nfe_best, runtime=r.runtime,
                                 idx=[int(i) for i in r.idx], history=r.history,
                                 note=r.note))
            T["methods"][mname] = {"type": "stochastic", "runs": runs}
            w = [x["weight"] for x in runs if x["feasible"]]
            print(f"{tname:7s} {mname:7s} best={min(w):8.2f} mean={np.mean(w):8.2f} "
                  f"SR={len(w)/N_RUNS:.0%} ({time.time()-t0:.0f}s)", flush=True)
        for mname, f in o.DETERMINISTIC.items():
            r = f(p, BUDGET[tname], None)
            T["methods"][mname] = {"type": "deterministic", "runs": [dict(
                seed=None, weight=r.weight, feasible=bool(r.feasible), nfe=r.nfe,
                nfe_best=r.nfe_best, runtime=r.runtime,
                idx=[int(i) for i in r.idx], history=r.history, note=r.note)]}
            print(f"{tname:7s} {mname:7s} W={r.weight:8.2f} feas={r.feasible}", flush=True)
        json.dump(res, open(out, "w"), default=float)
    return res


if __name__ == "__main__":
    main(*sys.argv[1:])
