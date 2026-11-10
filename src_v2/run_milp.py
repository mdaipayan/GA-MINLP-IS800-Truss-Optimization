"""Exact MILP for one truss.  usage: python run_milp.py <truss> <time_limit_s>
Warm start = best feasible design in results_v2/stats_v2.json."""
import json, sys, engine2 as e, milp_exact as mx
t, tl = sys.argv[1], float(sys.argv[2])
S = json.load(open("../results_v2/stats_v2.json"))
best = min((r for m in S["trusses"][t]["methods"].values() for r in m["runs"] if r["feasible"]),
           key=lambda r: r["weight"])
o = mx.solve_exact(e.BENCHMARKS[t](), time_limit=tl, warm_idx=best["idx"], threads=1)
o["idx"] = [int(i) for i in o["idx"]]; o["sections"] = [e.CAT_LABEL[i] for i in o["idx"]]
o["time_limit"] = tl
json.dump(o, open(f"../results_v2/milp_{t.replace('-', '')}.json", "w"), indent=1, default=float)
print(t, o["status"], "incumbent", round(o["weight"], 2), "bound", round(o["bound"], 2), "nodes", o["nodes"], "time", round(o["runtime"], 1))
