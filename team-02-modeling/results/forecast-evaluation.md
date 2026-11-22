# Team 02 — Forecast evaluation (Deliverable 5) — Sahith

Code: `notebooks/two_year_forecast.ipynb`, sections 3–4. **NDA:** rounded numbers only.

Final run: November 22, 2026 (headline results, interval coverage, network total, feature importance). The
horizon, residual, segment and history diagnostics below are from the November 21 reproducibility run; the
baseline's network total differs between the two runs and has not been reconciled.

## Test design

Hold out the last 24 months; forecast all of them from a single forecast date using calendar, airport, and
two-years-ago features available then. Baseline: **same month two years ago** (the honest seasonal naive for a
two-year horizon). Missing seasonal values fall back to the route's training median, then the overall training
median. The loss-selection limitation is documented below.

The point forecast is **XGBoost with Poisson loss**, which beat `log1p` among the tree candidates in Deliverable 4.
The same-month-last-year baseline won that comparison overall. Ridge is the separate 12-month comparison in
[`model-comparison.md`](model-comparison.md); the two-year notebook does not fit Ridge. A D4 result should not be
compared directly with a D5 result because the holdout and lag features differ.

## Results

WAPE is a percentage; MAE and RMSE are rounded passengers per route-month. Lower is better.

| Period | Model WAPE | Baseline WAPE | Model MAE | Baseline MAE | Model RMSE | Baseline RMSE |
|---|---|---|---|---|---|---|
| All 24 months | 23.5% | 19.8% | 717 | 621 | 1,859 | 1,652 |
| Year 1 (months 1–12) | 19.0% | 18.0% | 581 | 565 | 1,422 | 1,365 |
| Year 2 (months 13–24) | 28.0% | 21.7% | 853 | 677 | 2,211 | 1,895 |

**The baseline wins overall and in both years on all three metrics.** Model WAPE is about 3.7 percentage points worse
overall; year 1 is close (about 1 point behind) but the gap widens to about 6 points in year 2. Winning the comparison between tree losses did not mean
beating the seasonal baseline.

- **80% prediction interval coverage:** 77.3%, close to the nominal 80% (slightly narrow). The median band width
  is about 0.92 times the median prediction.
- **Network total over 24 months:** model **+12.1%** (over-forecast), baseline **−5.6%** (under-forecast). Net totals allow
  over- and under-forecasts to cancel, so they do not replace route-month WAPE.

## Diagnostics (charts in `data/visuals/`, local only)

- **Observed vs forecast (`observed_vs_forecast.png`):** the forecast follows some monthly variation (correlation
  0.62 with observed network totals), but exceeds actual demand in 23 of 24 months. The main issue is upward bias,
  consistent with the +12.1% total error.
- **Error by months ahead (`error_by_horizon.png`):** error generally increases with the horizon. Monthly model
  WAPE ranges from 15.36% to 37.14%, with the worst result at month 23. It beats the baseline in only 5 of 24 months;
  year 2 is materially worse than year 1.
- **Residuals (`residuals.png`):** log residuals are not centred on zero: median −0.13 and mean −0.17. Negative
  values mean over-forecasting; only 33.6% of route-months are under-forecast. The histogram clips residuals at
  ±3, affecting about 0.2%, so its displayed tails are truncated.
- **Segments (`error_by_segment.png`):** small routes have the highest relative error (41.23% WAPE), followed by
  medium routes (32.16%) and large routes (21.69%). International routes are harder than domestic routes (31.89%
  versus 20.24%). The baseline wins in every reported segment. Large routes carry 89.4% of passengers, so aggregate
  WAPE gives little weight to the small-route problem.
- **Feature importance (`feature_importance.png`, final run):** the two-years-ago passenger value has the highest
  share of gain (about 46%), then the international flag (about 13%) and the origin airport (about 10%). These gain scores describe model reliance, not causal effects or passenger shares.
- **Forward forecast (`two_year_forecast.png`):** the next 24 months are model projections. Forecast year 1 is
  +1.1% versus the last observed 12 months, and year 2 is −3.3% versus forecast year 1. Neither change has been
  validated against future observations.

## Does the data support a two-year forecast?

There are about **7.6 years of history** (January 2019–July 2026). At the August 2024 forecast origin, the training
pool spans about **5.6 years**, including the six months reserved for early stopping. A two-years-ago value exists
for 63.1% of training route-months and 99.9% of test route-months.

That supports constructing the lagged features, but **does not establish that this model improves a two-year
forecast**. Year-2 WAPE is 28.0% versus the baseline's 21.7%, although interval coverage is close to nominal. The honest
claim is that the pipeline produces a 24-month forecast for a specified route roster; this run does not justify
replacing the seasonal baseline.

## Limitations

- Trees cannot extrapolate long-run growth; trend beyond the training range is flattened.
- Missing route-months are not filled with zero, so scoring covers recorded route-months rather than every possible
  route-month. Whether a missing record means no service or no observation remains unresolved.
- The backtest uses the routes actually recorded in the held-out months. The forward forecast instead includes only
  routes active in the last observed 12 months; it does not predict route openings, closures, or the future route roster.
- No separate new-route segment appeared in this backtest, so it does not establish cold-start accuracy.
- The network band sums route-level quantiles. It is an approximate display, not a calibrated 80% network interval;
  the reported coverage is measured on individual route-months.
- The D4 loss decision uses the last 12 months, which overlap this 24-month holdout. This is not an untouched final
  test after model selection. One fixed forecast origin also cannot establish reliability across different regimes.
- The tree helper reserves the last six training months for early stopping and does not refit at the selected tree
  count on the entire training period. The same limitation applies to the final forward forecast.
- Both backtest fits exhausted the 2,000-tree budget: the best point iteration used 1,997 trees and the quantile fit
  used 2,000. Convergence and the effect of a larger search budget have not been established.
- Calendar, airport, and lag features omit future prices, capacity, competition, and disruptions. A forecast based on
  historical demand cannot anticipate those changes on its own.

## Business use

For capacity and fleet planning, the seasonal naive remains the benchmark this candidate must beat. Treat the
forward model output as a provisional scenario, not a validated demand commitment. Before operational use, evaluate
multiple forecast origins with model selection outside the final test window, check interval calibration, and
monitor monthly performance against the same seasonal baseline. This is not a short-term revenue-management model.
