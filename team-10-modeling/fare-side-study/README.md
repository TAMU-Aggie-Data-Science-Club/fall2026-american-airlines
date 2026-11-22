# Team 10 — fare side study (not the team submission)

A side analysis on the compact synthetic **fare** file `data/raw/aa_fares_compact_SYNTHETIC.csv`
(200 quotes, target `total_price`). The team's Deliverable 3–5 submission is the two-year passenger
forecast in the parent folder (`team-10-modeling/README.md`). This study answers a different
question (what fare does AA quote?) and cannot support a demand forecast: the file has no dates
and no passenger counts. Paths below are relative to this folder.

## Run order

Run each notebook top to bottom (Kernel → Restart & Run All). Each one regenerates its files
in `results/` and `visuals/`.

| # | notebook | deliverable | writes |
|---|---|---|---|
| 1 | `notebooks/01_schema_saideep.ipynb` | D3: schema, types, consistency checks | `results/data_dictionary.md` |
| 2 | `notebooks/02_eda_saideep.ipynb` | D3: EDA → feature set | `visuals/eda_*.png` |
| 3 | `notebooks/03_model_selection_saideep.ipynb` | D4: candidate comparison | `results/cv_scores.md`, `visuals/d4_*.png` |
| 4 | `notebooks/04_final_model_saideep.ipynb` | D5: tuning, test, diagnostics | `results/test_scores.md`, `visuals/d5_*.png`, `models/final_ridge.joblib` |

Shared code: `notebooks/load_data.py` (read and type the file) and `notebooks/features.py`
(`clean()` + `build_features()`, the D3 feature set). Later notebooks import these and never copy the logic.

Dependencies: pandas, numpy, matplotlib, seaborn, scikit-learn, xgboost (all already required by D1–D2).

## Results

| file | contents |
|---|---|
| `results/data_dictionary.md` | Columns, types, counts, consistency checks, open PM questions |
| `results/feature-set.md` | D3: every kept, engineered and dropped feature, with the reason |
| `results/model-comparison.md` | D4: candidates, validation, side-by-side scores, trade-offs |
| `results/baseline-decision.md` | D4: the chosen baseline and why |
| `results/forecast-evaluation.md` | D5: test design, held-out results, interval coverage, why a two-year forecast is not supportable |
| `results/final-model-report.md` | D5: training procedure, diagnostics, segment errors, business interpretation |

## Key findings

- **Model:** Ridge regression on log price (16 features). Fallback when the competitor price
  is unavailable: XGBoost without it.
- **Headline:** about 6% typical error (test MAE $22). That is **no better than pricing at the
  lowest competitor quote**, which carries almost all the signal in this data.
- **Two-year demand forecast:** not delivered. The file has no dates and no demand variable.
  What would be needed is in `results/final-model-report.md` §6.
