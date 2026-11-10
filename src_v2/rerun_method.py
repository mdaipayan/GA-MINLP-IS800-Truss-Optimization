"""Re-run one stochastic method (same seeds, same budgets) and update stats_v2.json.
Also stores the no-GA ablation (DNS + ILS from the random initial population)."""
import json, sys, numpy as np, engine2 as e, optimizers2 as o
F = "../results_v2/stats_v2.json"; S = json.load(open(F))
def pack(r, s): return dict(seed=s, weight=r.weight, feasible=bool(r.feasible), nfe=r.nfe, nfe_best=r.nfe_best,
                           runtime=r.runtime, idx=[int(i) for i in r.idx], history=r.history, note=r.note)
abl = {}
for t, make in e.BENCHMARKS.items():
    p = make(); B = S["budget"][t]
    runs = [pack(o.ga_dns(p, B, s), s) for s in range(S["n_runs"])]
    S["trusses"][t]["methods"]["GA-DNS"] = {"type": "stochastic", "runs": runs}
    json.dump(S, open(F, "w"), default=float)
    w = [r["weight"] for r in runs]
    print(t, "GA-DNS best %.2f median %.2f mean %.2f worst %.2f certified %d/20" % (min(w), np.median(w), np.mean(w), max(w), sum(r["note"] == "certified local minimum" for r in runs)), flush=True)
    abl[t] = [pack(o.ga_dns(p, B, s, ga_share=0.0), s) for s in range(S["n_runs"])]
    w = [r["weight"] for r in abl[t]]
    print(t, "no-GA   best %.2f median %.2f mean %.2f worst %.2f" % (min(w), np.median(w), np.mean(w), max(w)), flush=True)
    json.dump(abl, open("../results_v2/ablation_noGA.json", "w"), default=float)
