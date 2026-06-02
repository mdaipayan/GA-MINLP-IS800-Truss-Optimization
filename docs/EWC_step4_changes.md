# EWC Step 4 changes

## Main purpose
Prepared the repository structure for GitHub/Zenodo archival and fixed seed-control problems before final rerun.

## Code changes made

1. Added `_prepare_seed(seed)` helper in `src/truss_engine.py`.
2. Added optional `seed` argument to stochastic and stochastic-like optimization routines:
   - `opt_minlp`
   - `opt_sa`
   - `opt_ga`
   - `opt_pso`
   - `opt_aco`
   - `opt_bbbc`
   - `opt_de`
   - `opt_ga_minlp`
3. Replaced hard-coded `seed=42` in SciPy differential-evolution calls with explicit run seeds.
4. Updated `src/paper_suite.py` and `src/run_fresh.py` so each stochastic method is called as `mfn(truss, seed=seed)`.
5. Preserved the old numerical results as `PRE_SEED_FIX` files for audit trail only.

## Required before submission

Rerun the full statistical suite after this seed fix. Do not submit the pre-seed-fix results as final Engineering with Computers results unless the manuscript clearly states that DE/MINLP were deterministic by design. The better SCI-safe option is to rerun.
