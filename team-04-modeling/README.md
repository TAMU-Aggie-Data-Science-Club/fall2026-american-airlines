# team-04-modeling

Team 04's final model: a two-year-ahead passenger forecast per route (XGBoost + Random Forest ensemble). Full brief:
[`DELIVERABLES.md`](../DELIVERABLES.md#deliverable-3--real-data-ingestion--eda).

```
team-04-modeling/
├── scripts/     # config.py (paths, settings), features.py (candidates, self-check), make_fake_data.py
├── notebooks/   # 01–03 EDA (D3), 04–09 final model pipeline
├── models/      # trained models (git-ignored)
└── results/     # metrics and slide charts (git-ignored, they carry real numbers)
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

## Final model (notebooks 04–09)

Built by Daniel Ko. Run in order from `team-04-modeling/notebooks/`; every path resolves to the repo-root `data/` folder.

| Notebook | What it does |
|---|---|
| `04_collect_external_data` | World Bank GDP and population, FRED exchange rates, a monthly calendar table |
| `05_build_dataset_part1` | Route-month spine, airports, calendar, BTS T-100 market data, FAA cargo |
| `06_build_dataset_part2` | FAA enplanements, BEA county GDP and income, Census metro population and migration, country GDP |
| `07_feature_engineering` | Distance, international flags, economic-gravity and purchasing-power scores, seasonality |
| `08_model_exploration` | LightGBM, XGBoost, HistGradientBoosting and Random Forest on the same time-based split |
| `09_final_model` | Tunes XGBoost + Random Forest, blends them, compares with a naive forecast, forecasts 24 months, saves slide charts |

**Leakage rules.** Every outside feature is lagged two years (`DEP_YEAR - 2`), and route history enters only as
24-month lags, so each prediction uses only what is known two years earlier. `*_L12` features are excluded: the
two-year forecast never has them. Validation is the last 12 months of actuals, trained only on earlier months;
COVID years (2020–2021) are left out of training targets.

**Outputs (all git-ignored):** `results/pax_validation_vs_naive.csv`, `results/figures/*.png` (network monthly
totals and feature importance only), `models/*`.

Extra packages: `pip install xgboost lightgbm joblib wbgapi holidays geopandas shapely seaborn`

## Open questions for a PM

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
