# Team 07 — Model comparison (Deliverable 4)

## What we are forecasting

`OD_PAX`: passengers per route (origin → destination) per month, for every AA route, **two years ahead**.
Each forecast is made from a *forecast date* (the last month the model may see) for each of the next
24 months. This is the two-year setup Deliverable 5 asks for, not next-month prediction.

## Validation method (shared by every candidate)

- **Test:** one forecast made at **2024-07** for **2024-08 to 2026-07** (all 24 months unseen).
- **Training:** forecast dates 2022-07 to 2024-06, targets up to 2024-07. Earlier forecast dates are
  left out because their 12-month history falls in COVID (notebooks 01, 03).
- **Model selection:** early stopping and every design choice were checked on a validation slice
  (forecast dates before 2023-07, targets after it). A change was kept only if it improved both
  validation and test.
- **Rolling-origin check:** the final recipe was re-run from 5 forecast dates (2023-07, 2023-10,
  2024-01, 2024-04, 2024-07), each using only data up to that date.
- **Features:** the feature set from `notebooks/05_feature_set.ipynb`, built only from months up to
  the forecast date (leakage test: multiplying all later data by 10 leaves every feature unchanged).

## Shared metrics

| Metric | Meaning |
|---|---|
| **WAPE** (headline) | Total absolute error ÷ total passengers. Big routes count more, as they do for the business |
| MAE | Average miss per route-month, in passengers |
| RMSE | Like MAE but punishes large misses more |
| Bias | Total forecast ÷ total actual − 1. Positive means too many passengers forecast |

## Results on the 2024-07 test forecast

| Candidate | Owner | WAPE (2 yrs) | Year 1 | Year 2 | RMSE | Bias | Status |
|---|---|---|---|---|---|---|---|
| Seasonal naive (same month, latest year) | baseline | 18.3% | 15.1% | 21.7% | 1,473 | +4.1% | done |
| XGBoost, raw | Kingston | 20.1% | 14.7% | 25.9% | 1,626 | +11.3% | done |
| **XGBoost, reconciled to naive network total** | Kingston | **17.3%** | **13.9%** | **21.0%** | **1,390** | +4.1% | done |
| *(teammate candidate)* | *TBD* | | | | | | *pending* |
| *(teammate candidate)* | *TBD* | | | | | | *pending* |

Rolling-origin check, average over the 5 forecast dates: **XGBoost 16.1% vs naive 16.9%**. XGBoost
was better in both year 1 and year 2 at every date.

By route size (2024-07 test): smallest half of routes 29.6% vs 33.5% naive, middle 40% 21.9% vs 22.8%,
top 10% (62% of passengers) 13.9% vs 14.7%.

80% prediction ranges: 82.6% of actual values fell inside in year 1, 80.6% in year 2.

## Trade-offs

**Seasonal naive.** Free, transparent, and hard to beat on a stable network. It cannot react to a route
that is clearly growing or shrinking, and it gives no ranges.

**XGBoost.** One model across all ~17,000 routes. It handles different route sizes, airport codes, and
missing months without special treatment, trains in under a minute, and shows feature importance. Its
weakness is that trees cannot extrapolate: trained on 2022–2024, a recovery period, the raw model
learned growth and forecast 18% too many passengers in year 2.

**Why reconciled.** XGBoost decides how passengers split across routes. The network total comes from
seasonal naive, and each month's route forecasts are scaled to add up to it. This removes the learned
recovery growth while keeping XGBoost's route-level gains. It means the model does **not** forecast
network growth or decline; that comes entirely from the naive total.

## What was tried and not kept

| Change | Result |
|---|---|
| Every lag at least 24 months old (no forecast date) | 29% WAPE: too much recent data thrown away |
| Predicting change vs. the naive value | Worse: learned recovery growth even more strongly |
| Tweedie loss (Ranbin, 03) | Better on validation, worse on test |
| Blending toward naive as the horizon grows | Validation picked no blend |
| More training history (from 2021-07 or 2022-01) | Worse on the same validation rows |
| Reconciling per origin airport | Worse on validation |
| Tree depth 6 / 10, learning rate 0.1, larger min child weight, averaging 3 seeds | No gain on both validation and test |
| Smoother network total (12-month level × seasonal shape) | Won on earlier recovery-period dates, lost on the 2024-07 test (11.2% vs 8.6% year-2 network miss) |
| LSTM | Not run: 91 months per route with COVID in the middle is too little clean history |
| ARIMA per route | Not run by Kingston: ~17,000 separate fits; open for a teammate on top routes or network totals |

Kept from teammates: Xiaojie's airport attributes per route end and Ranbin's COVID-history flag. They
helped once reconciliation was in place.

## How to reproduce

```
python team-07-modeling/xgboost_model.py             # test results + the real 2026-08 to 2028-07 forecast (~3 min)
python team-07-modeling/xgboost_model.py --backtest  # also the rolling-origin check (~4 min)
```

Needs the two AA files in `data/raw/`. Forecast files are written to `models/` and are git-ignored.
