# Step 4 action checklist for the author

## On your Windows machine

1. Unzip `EWC_step4_reproducibility_repo.zip`.
2. Open a terminal inside the unzipped folder.
3. Run:

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
cd src
python run_fresh.py
python generate_tables.py
```

4. Save the new outputs.
5. Compare the new best/mean/std/SR values with the old `PRE_SEED_FIX` outputs.
6. Send the new `stats_results.json` back for Step 5 manuscript-number update.

## GitHub setup

```bash
git init
git add .
git commit -m "Initial reproducibility package for EWC submission"
git branch -M main
git remote add origin https://github.com/<your-user>/<repo-name>.git
git push -u origin main
```

## Zenodo setup

After pushing to GitHub, connect the repository to Zenodo, create a release, and copy the Zenodo DOI into the manuscript Code availability statement.
