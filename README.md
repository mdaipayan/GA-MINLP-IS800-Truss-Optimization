# GA-MINLP-IS800-Truss-Optimization

Reproducibility repository for the manuscript:

**A GA-Warm-Started Mixed-Integer Nonlinear Framework for Discrete IS 800-Compliant Space-Truss Optimization**

Target journal: **Engineering with Computers**.

This repository contains Python source code, seed-fixed statistical results, figure files, LaTeX tables, and Springer manuscript files for a computational framework that combines genetic-algorithm warm-starting with integer nonlinear refinement for catalogue-based IS 800:2007-compliant steel space-truss optimization.

## Repository structure

```text
src/          Python source code for truss analysis, IS 800 checks, optimizers, and table/figure generation
results/      Seed-fixed statistical JSON and generated LaTeX result tables
figures/      Final manuscript figure PDFs
manuscript/   Springer manuscript source/PDF, title page, and article highlights
notebooks/    Google Colab notebook used for execution support
docs/         Revision notes and compile-check report
```

## Quick start

```bash
python -m venv .venv
source .venv/bin/activate      # Linux/macOS
# .venv\Scripts\activate      # Windows
pip install -r requirements.txt
cd src
python verify_environment.py
python run_fresh.py
python generate_tables.py
```

## Reproducibility protocol

The final reported results use `N = 20` independent stochastic runs with external seeds `0--19`. The seed-fixed workflow passes the run seed to Python, NumPy, and SciPy differential-evolution based routines so that stochastic variability is visible in the reported statistics.

The key output file is:

```text
results/stats_results_seedfixed.json
```

## Manuscript files

The Springer manuscript with separate title page is stored in:

```text
manuscript/paper1_manuscript_EWC_springer_with_titlepage.tex
manuscript/paper1_manuscript_EWC_springer_with_titlepage.pdf
```

Before journal submission, replace any author and repository placeholders in the manuscript with final author details and the archived DOI/URL.

## Citation

If using this repository, please cite the associated manuscript and archived release DOI.
