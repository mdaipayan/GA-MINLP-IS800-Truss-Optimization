"""generate_tables.py — run once to produce all LaTeX tables from stats_results.json"""
import sys, os, json, math
import numpy as np
sys.path.insert(0, os.path.dirname(__file__))
from truss_engine import SP6_CATALOG, CAT_A, CAT_r, TRUSSES, fcd_is800, fy, gm0

OUT = "/mnt/user-data/outputs/paper1"
with open(f"{OUT}/stats_results.json") as f:
    results = json.load(f)

TNAMES = ["6-Bar Tetrahedron", "25-Bar Space Truss", "72-Bar Space Truss"]
STO    = ["SA", "GA", "PSO", "ACO", "BB-BC", "DE", "GA-MINLP*"]
ALL_M  = ["FSD","LP","SLP","SQP","NLP","MILP","MINLP"] + STO

def isnan(v):
    return isinstance(v, float) and math.isnan(v)

def getw(d):
    if d.get("type") == "stochastic":
        v = d.get("best_feas", float("nan"))
    else:
        v = d.get("weight", float("nan")) if d.get("is800", False) else float("nan")
    return v if (not isnan(v)) and v < 9000 else float("nan")


# ══ TABLE 2: Statistical results ═════════════════════════════════
t2 = open(f"{OUT}/table2_statistical.tex", "w")

t2.write("% Table 2: Statistical results (N=20)\n")
t2.write("\\begin{table*}[ht]\\centering\n")
t2.write("\\caption{Statistical optimization results over $N=20$ independent "
         "runs per method per truss. SR\\,=\\,Success Rate (\\% of runs yielding "
         "IS~800:2007-compliant designs). Mean$_F$ and Best$_F$ are computed over "
         "feasible runs only. Bold = lowest best weight for that truss.}\n")
t2.write("\\label{tab:statistical}\n")
t2.write("\\begin{tabular}{@{}llccccc@{}}\n")
t2.write("\\toprule\n")
t2.write("\\textbf{Truss} & \\textbf{Method} & "
         "\\textbf{Mean}$_F$ (kg) & \\textbf{Std}$_F$ (kg) & "
         "\\textbf{Best}$_F$ (kg) & \\textbf{SR} (\\%) & $\\bar{t}$ (s) \\\\\n")
t2.write("\\midrule\n")

for ti, tname in enumerate(TNAMES):
    short = ["6-Bar", "25-Bar", "72-Bar"][ti]
    if ti > 0:
        t2.write("\\midrule\n")
    vb = [results[tname].get(mn, {}).get("best_feas", float("nan"))
          for mn in STO
          if not isnan(results[tname].get(mn, {}).get("best_feas", float("nan")))
          and results[tname].get(mn, {}).get("best_feas", 9999) < 9000]
    ob = min(vb) if vb else float("nan")
    first = True
    for mn in STO:
        d   = results[tname].get(mn, {})
        mf  = d.get("mean_feas", float("nan"))
        sf  = d.get("std_feas",  0.0)
        bf  = d.get("best_feas", float("nan"))
        sr  = d.get("sr", 0)
        tt  = d.get("mean_time", 0)
        mf_s = f"{mf:.2f}" if not isnan(mf) and mf < 9000 else "---"
        sf_s = f"{sf:.2f}" if not isnan(sf) else "0.00"
        bf_s = f"{bf:.2f}" if not isnan(bf) and bf < 9000 else "---"
        is_b = (not isnan(bf)) and (not isnan(ob)) and abs(bf - ob) < 1.0
        if is_b:
            mf_s = "\\textbf{" + mf_s + "}"
            bf_s = "\\textbf{" + bf_s + "}"
        tc = ("\\multirow{7}{*}{" + short + "}") if first else ""
        t2.write(f"{tc} & {mn} & {mf_s} & {sf_s} & {bf_s} & {sr:.0f} & {tt:.1f} \\\\\n")
        first = False

t2.write("\\bottomrule\n\\end{tabular}\n\\end{table*}\n")
t2.close()
print("table2_statistical.tex done")


# ══ TABLE 3: Benchmark comparison ════════════════════════════════
t3 = open(f"{OUT}/table3_benchmark.tex", "w")
t3.write("\\begin{table}[ht]\\centering\n")
t3.write("\\caption{Best IS~800:2007-compliant weights (kg) compared with "
         "published AISC-based results. $^\\dagger$AISC sections; "
         "$^\\ddagger$This work, IS~800:2007 / SP~6(1):1964.}\n")
t3.write("\\label{tab:benchmark}\n")
t3.write("\\begin{tabular}{@{}lcc@{}}\\toprule\n")
t3.write("\\textbf{Method / Reference} & \\textbf{25-Bar (kg)} & "
         "\\textbf{72-Bar (kg)} \\\\\n")
t3.write("\\midrule\n")
t3.write("\\textit{Published (AISC$^\\dagger$)} & & \\\\\n")
t3.write("\\quad Camp \\& Bichon (2004) ACO & 216.9 & --- \\\\\n")
t3.write("\\quad Perez \\& Behdinan (2007) PSO & 213.7 & 162.9 \\\\\n")
t3.write("\\quad Camp (2007) BB-BC & --- & 167.3 \\\\\n")
t3.write("\\midrule\n")
t3.write("\\textit{This work (IS~800 / SP~6(1)$^\\ddagger$)} & & \\\\\n")
for mn in ALL_M:
    w25 = getw(results.get("25-Bar Space Truss", {}).get(mn, {}))
    w72 = getw(results.get("72-Bar Space Truss", {}).get(mn, {}))
    s25 = f"{w25:.2f}" if not isnan(w25) else "n/a"
    s72 = f"{w72:.2f}" if not isnan(w72) else "n/a"
    if mn == "GA-MINLP*":
        t3.write("\\quad\\textbf{GA--MINLP}$^\\ddagger$ & "
                 "\\textbf{" + s25 + "} & \\textbf{" + s72 + "} \\\\\n")
    else:
        t3.write(f"\\quad {mn} & {s25} & {s72} \\\\\n")
t3.write("\\bottomrule\n\\end{tabular}\n\\end{table}\n")
t3.close()
print("table3_benchmark.tex done")


# ══ TABLES 4–6: IS 800 compliance ════════════════════════════════
TRUSS_SHORTS = {"6-Bar Tetrahedron": "6bar",
                "25-Bar Space Truss": "25bar",
                "72-Bar Space Truss": "72bar"}
TRUSS_CAPS   = {"6-Bar Tetrahedron": "6-Bar",
                "25-Bar Space Truss": "25-Bar",
                "72-Bar Space Truss": "72-Bar"}

for ti_tab, tname in enumerate(TNAMES):
    table_num = ti_tab + 4
    truss     = TRUSSES[tname]()
    short     = TRUSS_SHORTS[tname]
    cap       = TRUSS_CAPS[tname]

    # Find best IS800-compliant result
    best_w, best_mn, best_cat = 9999.0, None, []
    for mn in STO + ["FSD", "LP", "SLP", "MILP", "MINLP", "SQP"]:
        d  = results[tname].get(mn, {})
        w  = getw(d)
        ok = (d.get("sr", 0) > 0 if d.get("type") == "stochastic"
              else d.get("is800", False))
        cat = d.get("cat_idx", [])
        if ok and not isnan(w) and w < best_w and cat:
            best_w = w; best_mn = mn; best_cat = cat

    if not best_cat:
        print(f"  No feasible result for {tname} — skipping IS800 table")
        continue

    cat_idx = np.array(best_cat, dtype=int)
    A_arr   = CAT_A[cat_idx]
    r_arr   = CAT_r[cat_idx]
    L_arr   = truss.member_lengths()
    _, forces, _ = truss.assemble_and_solve(A_arr)

    fname = f"{OUT}/table{table_num}_is800_{short}.tex"
    with open(fname, "w") as tf:
        tf.write("\\begin{table}[ht]\\centering\\small\n")
        tf.write("\\caption{IS~800:2007 Annex~D member compliance for "
                 "\\textbf{" + best_mn + "} solution on the "
                 + cap + " truss ($W=" + f"{best_w:.2f}" + "$\\,kg). "
                 "All DCR\\,$\\leq$\\,1.0.}\n")
        tf.write("\\label{tab:is800_" + short + "}\n")
        tf.write("\\setlength{\\tabcolsep}{4pt}\n")
        tf.write("\\begin{tabular}{@{}clcccccc@{}}\n")
        tf.write("\\toprule\n")
        tf.write("Mem & Section & $A$ (cm$^2$) & $r$ (mm) & "
                 "$F$ (kN) & $\\sigma$ (MPa) & $KL/r$ & DCR \\\\\n")
        tf.write("\\midrule\n")
        ncap = min(len(cat_idx), 20)
        for m in range(ncap):
            sec   = SP6_CATALOG[cat_idx[m]]
            F     = forces[m]; A = A_arr[m]; r = r_arr[m]; L = L_arr[m]
            sig   = abs(F) / A if A > 0 else 0.0
            KLr   = truss.K_eff * L / r if r > 0 else 0.0
            nat   = "C" if F < 0 else "T"
            fa    = fcd_is800(KLr) if F < 0 else fy / gm0
            dcr   = sig / fa if fa > 0 else 0.0
            d_str = ("{\\color{failred}" + f"{dcr:.3f}" + "}"
                     if dcr > 1.0 else f"{dcr:.3f}")
            tf.write(f"M{m+1} & {sec[0]} & {sec[1]:.2f} & "
                     f"{sec[2]*10:.1f} & {F/1e3:.1f}\\,({nat}) & "
                     f"{sig/1e6:.1f} & {KLr:.0f} & {d_str} \\\\\n")
        if len(cat_idx) > 20:
            extra = len(cat_idx) - 20
            tf.write("\\multicolumn{8}{c}{$\\cdots$ (" + str(extra)
                     + " more members)} \\\\\n")
        tf.write("\\bottomrule\n\\end{tabular}\n\\end{table}\n")
    print(f"table{table_num}_is800_{short}.tex done")

print("\nAll tables generated successfully.")
