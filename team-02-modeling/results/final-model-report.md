# Team 02 — Final model report (Deliverable 5) — Aidan

Code: `notebooks/two_year_forecast.ipynb`. Charts: `data/visuals/` (local only — real numbers, never committed).

## Training procedure
- **Data:** D3 feature table — one row per route per month; dropped `OD_REV_USD` (leak), `SF_LOAD_TMS` (load
  timestamp), `Unnamed: 0` (row index).
- **Model:** XGBoost, Poisson loss, 2000-tree cap with early stopping (100 rounds) on the last 6 training months;
  learning rate 0.05, depth 8, row/column subsampling 0.8, airports as native categoricals.
- **Loss selection:** Poisson was selected over log1p because it performed better across MAE, RMSE, and WAPE. However, the seasonal baseline outperformed both XGBoost models in the model comparison.
- **Intervals:** a second XGBoost with quantile loss (10th / 50th / 90th percentile) on `log1p(OD_PAX)`.
- **Reproducible:** fixed random seed; re-run `feature_set.ipynb` then this notebook top to bottom.

## How the two-year horizon is represented
- Forecast date = start of the last 24 months; the model sees only months before it.
- All 24 months are predicted **at once** — this is not next-month prediction.
- "Same month last year" is dropped because it does not exist yet for months 13–24; "same month two years ago" is
  used instead and is known for every forecast month.
- Final forecast: refit on all data, then predict the 24 months after the data ends for every route active in the
  last 12 months.

## Model statistics
- Trees used: 2000 (point model) / 2000 (quantile model). Both models reached the tree cap.
- Top features by gain: Passenger demand from the same month two years ago, whether the route is domestic or international, and the origin airport.
- Overall WAPE vs baseline, year 1 vs year 2: see `forecast-evaluation.md`.

## Business interpretation
The two-year forecast can help American Airlines estimate future passenger demand for individual routes to improve capacity and fleet planning. It can also help identify seasonal demand patterns for airport staffing and determine which routes may experience growth or decline. Since the seasonal baseline outperformed both XGBoost models in the model comparison, historical passenger demand should still be considered when making these decisions. However, the model should not be used for short-term pricing decisions, predicting demand for brand-new routes, or accounting for unexpected events such as the COVID-19 pandemic.

## Assumptions and risks
- The future looks like the past (no new shocks, no big network changes).
- A missing route-month is treated as "no record", not zero passengers (open PM question).
- No price, capacity or competitor data, so the model can't react to fare or schedule changes.
- Recommended monitoring: retrain monthly, compare against "same month two years ago", alert when WAPE drifts up.
