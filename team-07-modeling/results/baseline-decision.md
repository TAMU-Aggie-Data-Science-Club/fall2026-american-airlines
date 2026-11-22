# Team 07 — Baseline decision (Deliverable 4)

> **Status:** proposed by Kingston. Final once teammates' candidates are added to
> [`model-comparison.md`](model-comparison.md) and the team agrees.

## Selected model

**XGBoost, one global model across all routes, reconciled top-down to a seasonal-naive network total.**
Code: [`xgboost_model.py`](../xgboost_model.py).

## Why

1. **It beats the baseline at both horizons, at every forecast date tested.** On the 2024-07 test it
   scored 17.3% WAPE vs 18.3% for seasonal naive (year 1 13.9% vs 15.1%, year 2 21.0% vs 21.7%). Across 5
   rolling forecast dates it averaged 16.1% vs 16.9% and won in both years at every date.
2. **It is better on every route size,** including the top 10% of routes that carry 62% of passengers.
3. **It fits the data the EDA described:** ~17,000 short route histories, a wide size range,
   categorical airports, missing months, and strong seasonality (notebooks 01–04).
4. **It is leakage-safe for the two-year horizon.** Every feature comes from months up to the
   forecast date, which was checked by multiplying all later data by 10 and confirming no feature changed.
5. **It gives calibrated 80% ranges** (82.6% / 80.6% coverage in years 1 / 2), which Deliverable 5 requires.
6. **It is cheap and explainable:** it trains in under a minute and reports feature importance.

## What carries forward into Deliverable 5

- The forecast-date setup: one forecast date, predictions for months 1–24, test on 2024-07.
- The feature set from `notebooks/05_feature_set.ipynb`, plus Xiaojie's airport attributes and Ranbin's
  COVID-history flag.
- Training on log passengers with square-root route-size weights, early stopping on validation.
- Top-down reconciliation to the seasonal-naive network total, and symmetric 80% ranges.
- The real forecast for **2026-08 to 2028-07**, produced by the same script from all data through 2026-07.

## Known limits (to address in Deliverable 5)

- **No network growth or decline.** The network total is seasonal naive, so the forecast total is flat
  against the last 12 months. In the 2024-07 test this overpredicted year 2 by 7.6%, because traffic
  dipped in 2025–26. AA's data alone gave no reliable way to forecast that; a better network-total
  method was tested and did not hold up.
- **Little clean two-year training history.** COVID sits in the middle of 91 months, so the only full
  post-COVID two-year window is the test period itself. All validation is from the recovery years.
- **Missing route-months** are treated as "no data", not zero passengers. This needs to be confirmed with
  Patrick (notebook 01).
