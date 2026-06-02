import sys, platform
import numpy, scipy, matplotlib
try:
    import numba
    numba_v = numba.__version__
except Exception:
    numba_v = "not installed"
print("Python", sys.version)
print("Platform", platform.platform())
print("NumPy", numpy.__version__)
print("SciPy", scipy.__version__)
print("Matplotlib", matplotlib.__version__)
print("Numba", numba_v)
from truss_engine import make_tetrahedron, opt_de, opt_ga_minlp
tr = make_tetrahedron()
r1 = opt_de(tr, maxiter=2, popsize=3, seed=0)
r2 = opt_de(tr, maxiter=2, popsize=3, seed=1)
print("Smoke test DE seed0", r1.weight, r1.is800_ok)
print("Smoke test DE seed1", r2.weight, r2.is800_ok)
print("OK: seed-controlled functions import and run.")
