"""
run_locally.py
==============
Run this on YOUR machine to complete the N=20 statistical suite
(takes ~25 min), then regenerate all tables and figures.

Usage:
    pip install numpy scipy matplotlib
    python3 run_locally.py
"""
import subprocess, sys, os

steps = [
    ("Full N=20 statistical run", "paper_suite.py"),
    ("Regenerate all tables",     "generate_tables.py"),
]

for desc, script in steps:
    print(f"\n{'='*60}\n  {desc}\n{'='*60}")
    result = subprocess.run([sys.executable, script], check=False)
    if result.returncode != 0:
        print(f"  WARNING: {script} returned non-zero exit code")

print("\nDone. All outputs in the paper1/ folder.")
print("Compile the manuscript:")
print("  pdflatex paper1_manuscript.tex")
print("  pdflatex paper1_manuscript.tex  (second pass for ToC)")
