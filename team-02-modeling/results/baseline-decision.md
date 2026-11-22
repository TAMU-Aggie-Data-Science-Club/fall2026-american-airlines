# Team 02 — Baseline decision (Deliverable 4)

**Recommendation:** keep the seasonal baseline ("same month last year" for one year out, "same month two years ago"
for two years out) as the forecast to use today. **XGBoost with the Poisson loss** is our candidate model, carried
into Deliverable 5 and documented so it can be improved.

## Why
1. **Accuracy:** the baseline had the lowest WAPE on the 12-month test (about 18.0%), just ahead of XGBoost Poisson
   (about 18.9%) and `log1p` (about 19.3%). Poisson beat `log1p` among the XGBoost runs, and Ridge was far behind (see [`model-comparison.md`](model-comparison.md)). On the 24-month test
   the two-years-ago baseline also won overall and in both years ([`forecast-evaluation.md`](forecast-evaluation.md)).
2. **Why we still carry XGBoost:** it handles hundreds of airports and thousands of routes as native categoricals,
   learns month x route interactions (`eda_categoricals_akshithreddy-sheri.ipynb`), uses only information known at
   forecast time, and gives prediction intervals through quantile loss. It is the model worth improving.
3. **Why it lost (our read):** both XGBoost fits used the full 2,000-tree budget, so they had not finished learning;
   the forecast runs high overall (about +12% on network total over two years); and the intervals are
   slightly narrow (about 77% coverage for an 80% band). Each has a clear next step below.

## Next steps to beat the baseline
- Larger tree budget or higher learning rate, then confirm early stopping triggers.
- Correct the upward bias (check the trend feature and Poisson settings).
- Model the gap from the baseline (residual on top of the seasonal value) instead of raw passengers.
- Evaluate on several forecast origins with model selection outside the final test window.

## What we gave up
- Trees cannot extrapolate a trend past what they have seen — long-run growth is a limitation (see D5 report).
- New routes with no history get weaker forecasts.

Decided by: Akshith Reddy Sheri (lead) with Aidan Chance and Sahith Yamani, Nov 22, 2026.
