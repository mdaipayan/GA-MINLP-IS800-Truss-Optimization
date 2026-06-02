"""
run_fresh.py
============
Complete fresh rerun for Paper 1.
- Works on Windows (no /mnt/ paths)
- Deletes old cache automatically
- Fixes Figure 6 crash
- Generates all stats, figures, tables in current folder

Usage:
    cd G:\PhD\paper1
    python run_fresh.py

Output: all files saved in G:\PhD\paper1\  (same folder)
Time:   25–40 minutes
"""

import os, sys, json, time, random, math, warnings
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.ticker import MaxNLocator
warnings.filterwarnings("ignore")

# ── Work in the same folder as this script ────────────────────────
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
JSON_PATH = os.path.join(HERE, "stats_results.json")

# ── Delete old cache if it exists ────────────────────────────────
for old in ["stats_results.json", "stats_results_fixed_6bar.json"]:
    p = os.path.join(HERE, old)
    if os.path.exists(p):
        os.remove(p)
        print(f"Deleted old cache: {old}")

print("\n" + "="*60)
print("  PAPER 1 — COMPLETE FRESH RERUN")
print("  N=20 statistical runs × 14 methods × 3 trusses")
print("="*60)

# ── Import engine ─────────────────────────────────────────────────
from truss_engine import (
    TRUSSES, ALL_METHODS, SP6_CATALOG, CAT_A, CAT_r,
    is800_compliant, weight_only, fallow_member,
    KL_lim_comp, KL_lim_tens, rho, fcd_is800, fy, gm0,
    opt_fsd, opt_lp, opt_slp, opt_sqp, opt_nlp, opt_milp, opt_minlp,
    opt_sa, opt_ga, opt_pso, opt_aco, opt_bbbc, opt_de, opt_ga_minlp,
    make_tetrahedron, make_25bar, make_72bar, N_CAT
)

STOCHASTIC = [
    ("SA",        opt_sa),
    ("GA",        opt_ga),
    ("PSO",       opt_pso),
    ("ACO",       opt_aco),
    ("BB-BC",     opt_bbbc),
    ("DE",        opt_de),
    ("GA-MINLP*", opt_ga_minlp),
]
DETERMINISTIC = [
    ("FSD",   opt_fsd),
    ("LP",    opt_lp),
    ("SLP",   opt_slp),
    ("SQP",   opt_sqp),
    ("NLP",   opt_nlp),
    ("MILP",  opt_milp),
    ("MINLP", opt_minlp),
]
TRUSSES_LIST = [
    ("6-Bar Tetrahedron",  make_tetrahedron),
    ("25-Bar Space Truss", make_25bar),
    ("72-Bar Space Truss", make_72bar),
]

# ══════════════════════════════════════════════════════════════════
# STEP 1 — STATISTICAL RUNS
# ══════════════════════════════════════════════════════════════════
print("\n" + "-"*60)
print("  STEP 1 — Statistical runs (N=20 per stochastic method)")
print("-"*60)

N      = 20
results = {}
t_start = time.perf_counter()

total_calls = len(STOCHASTIC) * N * len(TRUSSES_LIST)
done = 0

for tname, tfac in TRUSSES_LIST:
    truss = tfac()
    nm    = truss.n_mem()
    results[tname] = {}
    print(f"\n  [{tname}]  {nm} members")

    # Deterministic (single run each)
    for mname, mfn in DETERMINISTIC:
        random.seed(0); np.random.seed(0)
        try:
            r = mfn(truss)
            results[tname][mname] = {
                "type":     "deterministic",
                "weight":   r.weight,
                "is800":    r.is800_ok,
                "dcr_max":  r.dcr_max,
                "runtime":  r.runtime,
                "history":  r.history,
                "cat_idx":  r.cat_idx.tolist(),
            }
            print(f"    DET {mname:<8} W={r.weight:9.2f}  "
                  f"IS800={str(r.is800_ok):<5}  DCR={r.dcr_max:.3f}")
        except Exception as e:
            print(f"    DET {mname:<8} ERROR: {e}")
            results[tname][mname] = {
                "type": "deterministic", "weight": 9999, "is800": False,
                "dcr_max": 99, "runtime": 0, "history": [], "cat_idx": []
            }

    # Stochastic (N=20 runs each)
    for mname, mfn in STOCHASTIC:
        ws, ok_list, hists, ts = [], [], [], []
        for seed in range(N):
            random.seed(seed)
            np.random.seed(seed)
            try:
                r = mfn(truss, seed=seed)
                ws.append(r.weight)
                ok_list.append(int(r.is800_ok))
                # Store history ONLY if weight-scale (< 5000 kg)
                h = r.history if r.history else []
                h_clean = [v for v in h if isinstance(v, float) and v < 5000 and v > 0]
                hists.append(h_clean)
                ts.append(r.runtime)
            except Exception as e:
                ws.append(float("nan"))
                ok_list.append(0)
                hists.append([])
                ts.append(0)

            done += 1
            elapsed = time.perf_counter() - t_start
            pct = 100 * done / total_calls
            eta = (elapsed / done) * (total_calls - done) if done > 0 else 0
            print(f"    {mname:<12} seed={seed:2d}  "
                  f"W={ws[-1]:9.2f}  IS800={'✓' if ok_list[-1] else '✗'}  "
                  f"[{pct:4.0f}%  ETA {eta/60:.1f}min]")

        valid = [w for w, f in zip(ws, ok_list)
                 if f and not math.isnan(w) and w < 9000]
        results[tname][mname] = {
            "type":       "stochastic",
            "weights":    [float(w) for w in ws],
            "feasibles":  ok_list,
            "histories":  hists,
            "mean":       float(np.nanmean(ws)),
            "std":        float(np.nanstd(ws)),
            "best":       float(np.nanmin(ws)),
            "mean_feas":  float(np.mean(valid))   if valid else float("nan"),
            "std_feas":   float(np.std(valid))    if len(valid) > 1 else 0.0,
            "best_feas":  float(np.min(valid))    if valid else float("nan"),
            "worst":      float(np.nanmax(ws)),
            "sr":         sum(ok_list) / N * 100,
            "mean_time":  float(np.nanmean(ts)),
            "cat_idx":    [],
        }
        d = results[tname][mname]
        mf = d["mean_feas"]; sf = d["std_feas"]; bf = d["best_feas"]
        print(f"    → {mname:<12} "
              f"mean={mf:.2f}  std={sf:.2f}  best={bf:.2f}  SR={d['sr']:.0f}%")

    # Save after each truss (resume-safe)
    with open(JSON_PATH, "w") as f:
        json.dump(results, f, indent=2)
    print(f"  Saved {JSON_PATH}")

total_time = time.perf_counter() - t_start
print(f"\n  STEP 1 COMPLETE: {total_time/60:.1f} min")
print(f"  JSON: {os.path.getsize(JSON_PATH)//1024} KB")

# ══════════════════════════════════════════════════════════════════
# STEP 2 — FIGURES
# ══════════════════════════════════════════════════════════════════
print("\n" + "-"*60)
print("  STEP 2 — Generating figures")
print("-"*60)

plt.rcParams.update({
    "font.family":       "serif",
    "font.size":         10,
    "axes.titlesize":    11,
    "axes.labelsize":    10,
    "legend.fontsize":   8.5,
    "xtick.labelsize":   9,
    "ytick.labelsize":   9,
    "lines.linewidth":   1.6,
    "figure.dpi":        300,
    "savefig.dpi":       300,
    "savefig.bbox":      "tight",
    "savefig.pad_inches": 0.05,
})

COLORS = {
    "SA": "#E67E22", "GA": "#E74C3C", "PSO": "#8E44AD",
    "ACO": "#27AE60", "BB-BC": "#F39C12", "DE": "#16A085",
    "GA-MINLP*": "#C0392B", "FSD": "#2C3E50",
}
LSMAP = {
    "SA": "-", "GA": "--", "PSO": "-.", "ACO": ":",
    "BB-BC": (0,(5,1)), "DE": (0,(3,1,1,1)), "GA-MINLP*": "-",
}
STO_ORDER  = ["SA","GA","PSO","ACO","BB-BC","DE","GA-MINLP*"]
TNAMES     = ["6-Bar Tetrahedron","25-Bar Space Truss","72-Bar Space Truss"]
TSHORT     = ["6bar","25bar","72bar"]


def mean_and_std_hist(hists):
    """Compute mean ± std convergence curve from list of history lists."""
    clean = [h for h in hists if h and len(h) > 1]
    if not clean:
        return None, None
    max_len = max(len(h) for h in clean)
    padded  = []
    for h in clean:
        a = np.minimum.accumulate(np.array(h, dtype=float))
        if len(a) < max_len:
            a = np.concatenate([a, np.full(max_len - len(a), a[-1])])
        padded.append(a[:max_len])
    arr = np.array(padded)
    return arr.mean(axis=0), arr.std(axis=0)


# ── Figure 1: Geometry ────────────────────────────────────────────
print("  Figure 1: Truss geometry...")
try:
    from mpl_toolkits.mplot3d import Axes3D
    factories = [make_tetrahedron, make_25bar, make_72bar]
    fig = plt.figure(figsize=(13, 4.2))
    view_angles = [(25,-60),(20,-55),(22,-50)]
    for i, (tname, tfac, va) in enumerate(zip(TNAMES, factories, view_angles)):
        ax = fig.add_subplot(1, 3, i+1, projection="3d")
        truss = tfac(); nodes = truss.nodes; conn = truss.conn
        for mi, mj in conn:
            ax.plot([nodes[mi,0],nodes[mj,0]],[nodes[mi,1],nodes[mj,1]],
                    [nodes[mi,2],nodes[mj,2]],"b-",lw=0.9,alpha=0.65)
        fixed = {d//3 for d in truss.bc_dof}
        free  = set(range(truss.n_nodes())) - fixed
        for n in fixed: ax.scatter(*nodes[n], color="#E74C3C", s=35, zorder=5)
        for n in free:  ax.scatter(*nodes[n], color="#2E86C1", s=20, zorder=5)
        lv = truss.loads
        for n in free:
            fx,fy_,fz = lv[3*n], lv[3*n+1], lv[3*n+2]
            mag = math.sqrt(fx**2 + fy_**2 + fz**2)
            if mag > 0:
                sc = 0.4*max(np.ptp(nodes[:,0]),np.ptp(nodes[:,1]),
                             np.ptp(nodes[:,2]))/mag
                ax.quiver(nodes[n,0],nodes[n,1],nodes[n,2],
                          fx*sc,fy_*sc,fz*sc,
                          color="#E67E22",lw=1.5,arrow_length_ratio=0.3)
        ax.set_title(f"({chr(97+i)}) {tname}", fontsize=9, pad=4)
        ax.view_init(*va); ax.tick_params(labelsize=6)
        ax.set_xlabel("X (m)",fontsize=7); ax.set_ylabel("Y (m)",fontsize=7)
        ax.set_zlabel("Z (m)",fontsize=7)
        if i == 0:
            ax.legend(handles=[
                mpatches.Patch(color="#E74C3C",label="Support"),
                mpatches.Patch(color="#2E86C1",label="Free")],fontsize=6)
    fig.suptitle("Benchmark truss configurations",
                 fontsize=10, fontweight="bold", y=1.0)
    fig.tight_layout()
    out = os.path.join(HERE, "fig1_geometry.pdf")
    fig.savefig(out); plt.close()
    print(f"  Saved: {out}")
except Exception as e:
    print(f"  Fig 1 error (non-critical): {e}")


# ── Figures 2–4: Convergence curves ──────────────────────────────
for fi, (tname, tshort) in enumerate(zip(TNAMES, TSHORT)):
    print(f"  Figure {fi+2}: Convergence {tname}...")
    fig, ax = plt.subplots(figsize=(7.2, 4.5))
    ax.set_facecolor("#FAFAFA")
    plotted = 0

    for mname in STO_ORDER:
        d = results[tname].get(mname, {})
        if d.get("type") != "stochastic":
            continue
        hists = d.get("histories", [])
        mh, sh = mean_and_std_hist(hists)
        if mh is None or len(mh) < 2:
            continue

        x = np.arange(len(mh))
        color = COLORS.get(mname, "#333")
        ls    = LSMAP.get(mname, "-")
        lw    = 2.1 if mname == "GA-MINLP*" else 1.4

        ax.fill_between(x, np.maximum(mh - sh, 0), mh + sh,
                        color=color, alpha=0.12)
        ax.plot(x, mh, color=color, ls=ls, lw=lw, label=mname)
        plotted += 1

    if plotted == 0:
        ax.text(0.5, 0.5, "No convergence data\n(run locally with full parameters)",
                ha="center", va="center", transform=ax.transAxes,
                fontsize=11, color="#888")

    fsd_w = results[tname].get("FSD", {}).get("weight", None)
    if fsd_w and fsd_w < 9000:
        ax.axhline(fsd_w, color=COLORS["FSD"], ls=":", lw=1.2,
                   label=f"FSD ({fsd_w:.1f} kg)")

    ax.set_xlabel("Iteration / Generation", labelpad=5)
    ax.set_ylabel("Best Feasible Weight (kg)", labelpad=5)
    ax.set_title(f"Convergence \u2014 {tname}", fontweight="bold", pad=7)
    ax.xaxis.set_major_locator(MaxNLocator(integer=True, nbins=6))
    ax.grid(True, ls="--", lw=0.4, alpha=0.55)
    ax.spines[["top","right"]].set_visible(False)
    if plotted > 0:
        ax.legend(loc="upper right", framealpha=0.9, ncol=2, fontsize=8)

    fig.tight_layout()
    out = os.path.join(HERE, f"fig{fi+2}_convergence_{tshort}.pdf")
    fig.savefig(out); plt.close()
    print(f"  Saved: {out}  ({plotted} method curves)")


# ── Figure 5: Weight comparison bar chart ────────────────────────
print("  Figure 5: Weight comparison...")
METHOD_ORDER = ["FSD","LP","SLP","SQP","NLP","MILP","MINLP",
                "SA","GA","PSO","ACO","BB-BC","DE","GA-MINLP*"]
tcolors = ["#2E86C1","#E67E22","#27AE60"]
talpha  = [0.85, 0.75, 0.70]
hlist   = ["", "///", "..."]

fig, ax = plt.subplots(figsize=(13, 5))
ax.set_facecolor("#FAFAFA")
x = np.arange(len(METHOD_ORDER))
w = 0.25

for ti, (tname, tc, ta, hatch) in enumerate(zip(TNAMES, tcolors, talpha, hlist)):
    off  = (ti - 1) * w
    vals = []
    for mname in METHOD_ORDER:
        d = results[tname].get(mname, {})
        if d.get("type") == "stochastic":
            v  = d.get("best_feas", float("nan"))
            ok = d.get("sr", 0) > 0
        else:
            v  = d.get("weight", float("nan"))
            ok = d.get("is800", False)
        vals.append(v if ok and not math.isnan(v) and v < 9000 else float("nan"))

    heights = [0 if math.isnan(v) else v for v in vals]
    bars = ax.bar(x + off, heights, width=w, label=tname,
                  color=tc, alpha=ta, hatch=hatch,
                  edgecolor="white", lw=0.5)

    mi = METHOD_ORDER.index("GA-MINLP*")
    if not math.isnan(vals[mi]) and vals[mi] > 0:
        ax.annotate("★", xy=(mi+off, vals[mi]), xytext=(0, 4),
                    textcoords="offset points", ha="center",
                    fontsize=10, color="#922B21")

ax.set_xticks(x)
ax.set_xticklabels(METHOD_ORDER, rotation=35, ha="right", fontsize=9)
ax.set_ylabel("Best IS 800-Compliant Weight (kg)", labelpad=6)
ax.set_title("All-Method Weight Comparison \u2014 Three Benchmark Trusses",
             fontweight="bold", pad=8)
ax.legend(framealpha=0.9)
ax.grid(True, axis="y", ls="--", lw=0.4, alpha=0.55)
ax.spines[["top","right"]].set_visible(False)
fig.tight_layout()
out = os.path.join(HERE, "fig5_weight_comparison.pdf")
fig.savefig(out); plt.close()
print(f"  Saved: {out}")


# ── Figure 6: Phase improvement ───────────────────────────────────
print("  Figure 6: GA-MINLP phase improvement...")
try:
    phase1_w, phase2_w, labels = [], [], []
    for tname, tfac in TRUSSES_LIST:
        truss = tfac()
        random.seed(0); np.random.seed(0)
        r1 = opt_ga(truss)
        random.seed(0); np.random.seed(0)
        r2 = opt_ga_minlp(truss)
        phase1_w.append(r1.weight)
        phase2_w.append(r2.weight)
        labels.append(tname.replace(" Space Truss","").replace(" Tetrahedron",""))

    x2  = np.arange(len(TRUSSES_LIST))
    w2  = 0.35
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.set_facecolor("#FAFAFA")
    ax.bar(x2 - w2/2, phase1_w, w2,
           label="Phase 1 \u2014 GA (sizing only)",
           color="#3498DB", alpha=0.85, edgecolor="white")
    ax.bar(x2 + w2/2, phase2_w, w2,
           label="Phase 2 \u2014 MINLP (exact IS 800)",
           color="#C0392B", alpha=0.85, edgecolor="white")

    for i, (p1, p2) in enumerate(zip(phase1_w, phase2_w)):
        if p1 > 0:
            saving = 100 * (p1 - p2) / p1
            ax.annotate(f"{saving:+.1f}%",
                        xy=(x2[i] + w2/2, p2), xytext=(0, 5),
                        textcoords="offset points", ha="center",
                        fontsize=9, color="#922B21", fontweight="bold")

    ax.set_xticks(x2)
    ax.set_xticklabels(labels, fontsize=10)
    ax.set_ylabel("Best Weight (kg)", labelpad=6)
    ax.set_title("GA\u2013MINLP: Phase 1 (GA) vs Phase 2 (MINLP) Weight Improvement",
                 fontweight="bold", pad=8)
    ax.legend(framealpha=0.9)
    ax.grid(True, axis="y", ls="--", lw=0.4, alpha=0.55)
    ax.spines[["top","right"]].set_visible(False)
    fig.tight_layout()
    out = os.path.join(HERE, "fig6_phase_improvement.pdf")
    fig.savefig(out); plt.close()
    print(f"  Saved: {out}")
except Exception as e:
    print(f"  Fig 6 error: {e}")
    import traceback; traceback.print_exc()


# ══════════════════════════════════════════════════════════════════
# STEP 3 — LaTeX TABLES
# ══════════════════════════════════════════════════════════════════
print("\n" + "-"*60)
print("  STEP 3 — Generating LaTeX tables")
print("-"*60)

# Re-import generate_tables with local paths
import importlib.util, types

# Run generate_tables.py with correct OUT path
gt_path = os.path.join(HERE, "generate_tables.py")
if os.path.exists(gt_path):
    # Patch DATA_DIR in generate_tables
    with open(gt_path) as f:
        gt_src = f.read()
    # Replace OUT path in source
    gt_src = gt_src.replace(
        'OUT = "/mnt/user-data/outputs/paper1"',
        f'OUT = r"{HERE}"'
    ).replace(
        "OUT = '/mnt/user-data/outputs/paper1'",
        f"OUT = r'{HERE}'"
    )
    # Also fix JSON path
    gt_src = gt_src.replace(
        'with open(f"{OUT}/stats_results.json")',
        f'with open(r"{JSON_PATH}")'
    )
    exec(compile(gt_src, gt_path, "exec"), {"__builtins__": __builtins__,
                                             "__file__": gt_path})
else:
    print("  generate_tables.py not found — skipping tables")


# ══════════════════════════════════════════════════════════════════
# FINAL SUMMARY
# ══════════════════════════════════════════════════════════════════
print("\n" + "="*60)
print("  COMPLETE")
print(f"  Total time: {(time.perf_counter()-t_start)/60:.1f} min")
print()
print("  Files generated in:", HERE)
for fname in sorted(os.listdir(HERE)):
    if fname.endswith((".pdf",".tex",".json")) and not fname.startswith("paper1_manuscript"):
        sz = os.path.getsize(os.path.join(HERE, fname))
        print(f"    {fname:<45} {sz//1024:>5} KB")

print()
print("  Next step:")
print("    pdflatex paper1_manuscript.tex")
print("    pdflatex paper1_manuscript.tex")
print("="*60)
