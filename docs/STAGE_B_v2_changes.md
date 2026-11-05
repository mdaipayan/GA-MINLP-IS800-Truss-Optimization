# Stage B (v2): corrected benchmarks and fair re-run

## Bugs found in v1 (truss_engine.py)
1. 72-bar: `conn_list[:72]` had only 56 members; nodes 18-19 unconnected -> singular
   stiffness -> `except` branch returned zero displacements -> all member forces zero.
   All v1 72-bar results (241.67 kg etc.) are invalid.
2. 25-bar: mid-level nodes fixed; top square hung on 4 verticals -> mechanism
   (3 zero-energy modes, displacements ~1e13 m). All v1 25-bar results are invalid.
3. Neither model matched the literature benchmark used in Table 3.
4. "LP" and "MILP" were the same resizing loop; "MINLP" and "DE" were both SciPy DE.
5. GA had no topology variables, although the manuscript says it does.
6. Unequal analysis budgets (hybrid had 15-30x more analyses than GA).

## v2 (src_v2/)
- engine2.py: literature 25-bar (10 nodes, 8 groups, d_lim 0.35 in) and 72-bar
  (4 storeys, 72 members, 16 groups, 2 load cases, top x/y d_lim 0.25 in);
  stability check at build time; catalogue sorted by area; NFE counter.
- milp_exact.py: exact disaggregated MILP (HiGHS) -> certified global optimum.
- optimizers2.py: SA, GA, PSO, ACO, BB-BC, DE, FSD, SQP-R, GA-DNS; same NFE
  budget (3k / 10k / 20k) and same Deb feasibility rule for all.
- GA-DNS: GA (50 % budget) + deterministic discrete neighbourhood search with a
  local-optimality certificate + iterated local search on the remaining budget.

## Key results (N = 20)
| truss | W* (kg) | certificate | GA-DNS hit W* |
|---|---|---|---|
| 6-bar | 101.17 | MILP global optimum (0.02 s) | 95 % |
| 25-bar | 580.06 | MILP global optimum (127 s) | 90 % |
| 72-bar | 1032.47 | MILP gap 48 % after 90 min (lower bound 535.2) | 10 % |

## To verify by the author
- 25-bar load values (node 3 and node 6 Px = 0.5 kip) against the Camp & Bichon (2004) PDF.
- 72-bar load case 1 node and displacement-limit convention against Camp (2007).
