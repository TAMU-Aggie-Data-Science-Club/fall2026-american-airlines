# Team 08 — Final forecasting submission

Forecasts monthly passengers per origin–destination route two years ahead.

| Path | Contents |
|---|---|
| `notebooks/eda_khoa-ho-nguyen.ipynb` | Deliverable 3 EDA |
| `notebooks/AA_02_modeling_and_evaluation_v3_final.ipynb` | Final features, models and forecast (outputs cleared) |
| `scripts/origin_check.py` | Re-scores the model under different training cut-offs |
| `results/model-comparison.md`, `results/baseline-decision.md` | Deliverable 4 |
| `results/final-model-report.md`, `results/forecast-evaluation.md` | Deliverable 5 |

## Run it

Put the AA files plus the FAA, Census CBSA and county gazetteer files in the git-ignored `data/raw/`. Needs
pandas, numpy, scipy, scikit-learn, matplotlib and openpyxl. Open the notebook from its own folder and run all
cells, or run `python team-08-modeling/scripts/origin_check.py` from the repo root.

This repository is publicly readable and the AA data is under NDA, so only code, method and percentage errors are
committed. Charts, passenger-level errors and forecasts stay local, and `.gitignore` here blocks the notebook's
outputs.
