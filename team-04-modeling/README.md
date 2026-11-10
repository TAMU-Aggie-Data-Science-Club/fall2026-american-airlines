# team-04-modeling

Team 04's Deliverable 3 work. Full brief:
[`DELIVERABLES.md`](../DELIVERABLES.md#deliverable-3--real-data-ingestion--eda).

```
team-04-modeling/
├── scripts/     # config.py (paths, settings), features.py (candidates, self-check), make_fake_data.py
├── notebooks/   # 01_eda, 02_feature_set
├── models/      # saved models and predictions (git-ignored)
└── results/     # plots and summary tables (git-ignored, they carry real numbers)
```

The code uses generic column names. Until `data/real/columns.json` exists, it runs on made-up data from
`make_fake_data.py`. The first output line of each notebook says `USING FAKE DATA` or `USING REAL DATA`.

## Setup

From the repo root:

```sh
python3 -m venv .venv
source .venv/bin/activate
pip install pandas pyarrow scikit-learn matplotlib notebook
python team-04-modeling/scripts/features.py
python team-04-modeling/scripts/make_fake_data.py
jupyter notebook team-04-modeling/notebooks
```

Run `01_eda`, then `02_feature_set`. `02` writes the feature set to `data/processed/feature_set.parquet`.

## Switching to the real data

Everything here stays in the git-ignored `data/real/` folder. Dhruv DMs you the files and `columns.json`.

1. Save both data files and `columns.json` in `data/real/`.
2. Run both notebooks. Executed copies, with real outputs, land in the git-ignored `results/`, so the
   tracked notebooks stay clear:
   ```sh
   jupyter nbconvert --to notebook --execute --output-dir team-04-modeling/results team-04-modeling/notebooks/01_eda.ipynb team-04-modeling/notebooks/02_feature_set.ipynb
   ```
3. Open `results/01_eda.ipynb` and `results/02_feature_set.ipynb`, and check the first line says `USING REAL DATA`.
   `02` also rewrites `FEATURES.md`, the documented feature set, with no numbers in it.
4. Results on fake data mean nothing: use only what the real run shows.

## Open questions for a PM

- **Horizon:** how far ahead are we forecasting? `HORIZON` in `config.py` decides which lags are legal.
- **Target:** what exactly does the passenger column measure?
- **Absent months:** does a route with no row in a month mean zero, or below a reporting threshold?
- **Latest months:** are they final?

## Rules

- **NDA:** never commit the data, a sample of it, or a screenshot of it, and don't paste rows into Discord
  or a PR. Plots, tables and written findings with real numbers count too, so keep them in `results/`.
- **Real column names** stay in `data/real/columns.json`, not in tracked files, until a PM says otherwise.
- **No AI tools or uploads near the real data** until a PM says otherwise: nothing that reads `data/real/`
  or notebook outputs.
- **Clear outputs before committing:**
  `jupyter nbconvert --clear-output --inplace team-04-modeling/notebooks/*.ipynb`
- Run `git status` before every commit. No `.csv` or `columns.json` should ever appear.
- PR your branch into `team-04/main`, not `main`.
