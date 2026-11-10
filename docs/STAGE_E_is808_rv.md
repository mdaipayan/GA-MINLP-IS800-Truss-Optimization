# Stage E: section data corrected to IS 808:2021 (r_v) and methods adapted

## What was wrong
- The v1/v2 catalogue (51 "SP 6(1)" angles) listed the radius of gyration about the
  axis parallel to a leg (r_xx), not the minimum value r_v. Slenderness and
  buckling capacity were therefore unconservative by a factor of about 1.5.
- SP 6(1):1964 is shown as withdrawn by BIS. The current source is IS 808:2021.

## What changed
- `src_v2/is808_2021_equal_angles.csv`: the 72 equal leg angles of IS 808:2021
  Table 9 (area, I_vv, r_v). Every row is cross-checked: r_v = sqrt(I_vv/A),
  mass = 7850 A, area against leg/thickness geometry. For legs >= 100 mm the
  standard prints r_v without the decimal point; the corrected value is used.
- `engine2.py`: catalogue read from the CSV, kept in the order of the standard
  (leg, then thickness); grid neighbours of every section (`GRID_NB`).
- `optimizers2.py`: DNS neighbourhood = catalogue moves on the leg-thickness grid
  in one or two groups; FSD and SQP-R pick the lightest adequate section.
- `analysis2.py`: two-sided tests; DNS-ILS (no-GA ablation) added as a method.
- All results in `results_v2/` were regenerated. The earlier r_xx-based results
  (101.17 / 580.06 / 1032.47 kg) are superseded and remain only in git history.

## Why the catalogue order changed
With the catalogue sorted by area, adjacent positions often belong to different
leg sizes, so r_v jumps up and down from one position to the next. In a first
run with this order, index-based searches were trapped (SA often ended without a
feasible design). The order of the standard avoids this for all methods.
That first run was overwritten and is not part of the archived results.

## Key results (N = 20, equal analysis budgets)
| truss | W* (kg) | status | GA-DNS hit | DNS-ILS hit | FSD |
|---|---|---|---|---|---|
| 6-bar | 156.82 | certified global optimum | 100 % | 100 % | = W* |
| 25-bar | 981.22 | certified global optimum | 100 % | 100 % | = W* |
| 72-bar | 1651.48 | best known; MILP lower bound 812.39 kg, gap 51 % (90 min) | 5 % (16/20 within 1 %) | 0 % (14/20 within 1 %) | +5.7 % |

## Still to verify by the author
- 25-bar load at node 6 (0.5 or 0.6 kip) against the Camp & Bichon (2004) paper.
- 72-bar load case 1 and displacement-limit convention against Camp (2007).
