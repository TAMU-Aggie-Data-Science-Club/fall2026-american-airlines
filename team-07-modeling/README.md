# team-07-modeling

Team 07's work on the real American Airlines data (Deliverable 3 onward). Full brief:
[`DELIVERABLES.md`](../DELIVERABLES.md#deliverable-3--real-data-ingestion--eda).

```
team-07-modeling/
├── xgboost_model.py  # final model (Deliverable 4): two-year OD_PAX forecast
├── notebooks/        # EDA + feature set (run 00 first; every other notebook %runs it)
├── models/           # forecast outputs written by xgboost_model.py (git-ignored)
└── results/          # model comparison and baseline decision write-ups
```

## Final model (Deliverable 4)

[`xgboost_model.py`](xgboost_model.py) is the team's final model. It forecasts passengers (`OD_PAX`) for every
route, for each of the 24 months after a forecast date, using only data up to that date. Results and
reasoning: [`results/model-comparison.md`](results/model-comparison.md) and
[`results/baseline-decision.md`](results/baseline-decision.md).

```
python team-07-modeling/xgboost_model.py             # test results + the 2026-08 to 2028-07 forecast (~3 min)
python team-07-modeling/xgboost_model.py --backtest  # also re-tests from 5 forecast dates (~4 min)
```

## Setup

1. Get the two files from the team Drive link (ask Patrick — it is **not** posted in the repo):
   `PAX_FORECASTING` and `AIRPORT_INFO`.
2. Put them in the repo's git-ignored `data/raw/` folder (repo root, not this folder):
   ```
   data/raw/PAX_FORECASTING.csv
   data/raw/AIRPORT_INFO.csv
   ```
   Any filename works as long as it contains `PAX_FORECASTING` / `AIRPORT_INFO` and there's one of each.
3. Open `notebooks/00_load_and_preprocess.ipynb` and Run All. You should get a cleaning report and no errors.

## Who does what (Deliverable 3)

| Notebook | Owner | Branch |
|---|---|---|
| `00_load_and_preprocess` | Patrick | `team-07-main` |
| `01_schema_and_missingness` | Kingston | `team-07-kingston` |
| `02_distributions_and_outliers` | Lasya | `team-07-srilasyakamma` |
| `03_time_and_relationships` | Ranbin | `team-07-ranbin` |
| `04_leakage_cardinality_airports` | Xiaojie | `team-07-xiaojie-chen` |
| `05_feature_set` | Kingston (planned for Patrick) | `team-07-kingston` |
| `xgboost_model.py` (Deliverable 4) | Kingston | `team-07-kingston` |

Each EDA notebook ends with an *observation → modelling consequence* table. `05` turns those tables
into the team's model family and feature set.

## Rules

- **NDA:** never commit the data, a sample of it, or a screenshot of it. Don't paste rows into Discord
  or a PR. `.gitignore` in this folder blocks data files, but `git status` before every commit anyway.
- **Clear outputs before committing** (Kernel → Restart & Clear Output) — outputs contain real data.
- Only edit your own notebook. If `00`'s cleaning needs to change, tell Patrick.
- PR your branch into `team-07-main`, not `main`.
