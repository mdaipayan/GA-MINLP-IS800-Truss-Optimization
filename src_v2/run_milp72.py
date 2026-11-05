import json, numpy as np, engine2 as e, optimizers2 as o, milp_exact as mx
p = e.make_72bar()
best = None
for f in (o.ga, o.ga_dns, o.de, o.sa):
    for s in range(3):
        r = f(p, 20000, s)
        if r.feasible and (best is None or r.weight < best.weight):
            best = r
print("warm start", best.method, best.weight, flush=True)
out = mx.solve_exact(p, time_limit=5400, warm_idx=best.idx, verbose=True)
out["idx"] = [int(i) for i in out["idx"]]
out["sections"] = [e.CAT_LABEL[i] for i in out["idx"]]
json.dump(out, open("../results_v2/milp_72bar.json", "w"), indent=1, default=float)
print({k: v for k, v in out.items()}, flush=True)
