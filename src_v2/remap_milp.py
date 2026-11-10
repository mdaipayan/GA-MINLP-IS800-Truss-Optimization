"""Re-index a MILP result by section labels (robust to catalogue ordering)
and re-verify it with the analysis engine."""
import json, sys, engine2 as e
for t in sys.argv[1:]:
    f = f"../results_v2/milp_{t.replace('-', '')}.json"; o = json.load(open(f))
    o["idx"] = [e.CAT_LABEL.index(s) for s in o["sections"]]
    p = e.BENCHMARKS[t](); W, v, ok = p.evaluate(o["idx"])
    assert ok and abs(W - o["weight"]) < 1e-6, (t, W, o["weight"], v)
    json.dump(o, open(f, "w"), indent=1); print(t, "re-indexed and verified:", round(W, 2), o["sections"])
