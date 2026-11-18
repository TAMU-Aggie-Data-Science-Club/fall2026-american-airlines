# Forecast evaluation

Numeric appendix to `final-model-report.md`. Every figure in this file comes from `notebooks/two-year-forecast.ipynb`. The carried forecast is the route-level seasonal naive. The gradient-boosted tree is the rejected candidate, shown so the decision can be checked.

## Design

A forecast made at origin T uses only rows dated before T and predicts T through T+23 in one shot. It is not refreshed with actuals from inside the window. Horizons 1–12 are year 1. Horizons 13–24 are year 2.

| Origin | Train | Score | Role |
|---|---|---|---|
| 2024-08 | 2019-01 … 2024-07 | 2024-08 … 2026-07 | Primary |
| 2023-08 | 2019-01 … 2023-07 | 2023-08 … 2025-07 | Stability check. Overlaps the primary window by 12 months. |
| 2026-08 | rows through 2026-07 | none | Forward forecast. No actuals. |

The forecast set is every route-month in the 12 months before T, moved forward 12 months and 24 months. The naive copies that prior-year passenger count into both years. The tree, when scored, uses the 22 Deliverable 3 columns and no lagged target.

Scoring outer-joins the forecast set to the actual rows in the window. A missing actual is 0 passengers. A route-month outside the forecast set is forecast 0. The join preserves the actual total. Routes outside the forecast set were 0% of actual passengers at both origins.

Route-month WAPE is the sum of absolute errors divided by the sum of actuals. Bias is (forecast total − actual total) / actual total. The same bias falls out at route grain and at network grain, because both are sums. Network MAE and RMSE are passengers per month.

## Primary window, August 2024

Actual network total over the 24 months: 1,291,963,949.62 passengers.

| Metric | Naive | Tree |
|---|---:|---:|
| Route-month WAPE, 24 months | 18.29% | 19.18% |
| Route-month WAPE, year 1 | 15.10% | 16.15% |
| Route-month WAPE, year 2 | 21.69% | 22.42% |
| Bias, 24 months | +4.09% | +6.40% |
| Bias, year 1 | +0.75% | +2.99% |
| Bias, year 2 | +7.65% | +10.04% |
| Network monthly WAPE | 5.78% | 6.44% |
| Network MAE (passengers / month) | 3,113,174.10 | 3,465,121.15 |
| Network RMSE (passengers / month) | 4,319,584.87 | 4,537,239.11 |
| Route-month MAE (passengers) | 575.02 | 603.17 |
| Route-month RMSE (passengers) | 1,472.52 | 1,478.39 |
| WAPE of route 24-month totals | 11.32% | 13.71% |

A positive bias means the forecast total is above the actual total.

## Stability window, August 2023

| Metric | Naive | Tree |
|---|---:|---:|
| Route-month WAPE, 24 months | 16.77% | 16.82% |
| Route-month WAPE, year 1 | 15.44% | 15.75% |
| Route-month WAPE, year 2 | 18.11% | 17.90% |
| Bias, 24 months | −6.25% | −2.76% |
| Bias, year 1 | −6.60% | −3.12% |
| Bias, year 2 | −5.90% | −2.40% |
| Network monthly WAPE | 6.25% | 3.53% |
| Network MAE (passengers / month) | 3,490,428.18 | 1,969,727.88 |
| Network RMSE (passengers / month) | 3,836,714.39 | 2,339,818.56 |
| Route-month MAE (passengers) | 546.81 | 548.53 |
| Route-month RMSE (passengers) | 1,329.83 | 1,284.53 |
| WAPE of route 24-month totals | 11.15% | 11.64% |

The tree's network MAE is lower on this earlier window and higher on the primary window. The bar required a lower route-month WAPE in both years at both origins. Year 1 at this origin is 15.75% against 15.44%, so the window does not carry the tree.

## Route-month WAPE by horizon

`visuals/error-by-horizon.png`. Dashed lines are the August 2023 origin.

| h | Naive, 2024-08 | Tree, 2024-08 | Naive, 2023-08 | Tree, 2023-08 |
|---:|---:|---:|---:|---:|
| 1 | 13.93% | 15.18% | 16.84% | 15.21% |
| 2 | 14.87% | 14.65% | 15.71% | 15.05% |
| 3 | 15.38% | 16.42% | 16.12% | 14.57% |
| 4 | 14.81% | 15.61% | 15.39% | 14.55% |
| 5 | 14.53% | 15.26% | 15.65% | 15.06% |
| 6 | 15.50% | 16.38% | 15.66% | 16.65% |
| 7 | 16.40% | 17.56% | 16.16% | 17.02% |
| 8 | 15.61% | 16.15% | 15.64% | 16.30% |
| 9 | 15.40% | 16.15% | 15.04% | 15.29% |
| 10 | 15.78% | 16.16% | 15.06% | 15.98% |
| 11 | 13.91% | 16.66% | 13.95% | 16.09% |
| 12 | 15.40% | 17.73% | 14.48% | 17.23% |
| 13 | 17.03% | 18.23% | 20.70% | 17.67% |
| 14 | 17.42% | 19.33% | 18.86% | 17.53% |
| 15 | 16.33% | 17.63% | 19.55% | 18.36% |
| 16 | 20.03% | 21.88% | 17.25% | 17.16% |
| 17 | 18.59% | 18.43% | 19.80% | 16.80% |
| 18 | 21.81% | 22.50% | 17.81% | 18.12% |
| 19 | 18.90% | 19.98% | 17.31% | 18.04% |
| 20 | 24.77% | 24.55% | 16.95% | 17.39% |
| 21 | 18.98% | 19.61% | 16.89% | 17.31% |
| 22 | 31.96% | 30.47% | 17.00% | 17.93% |
| 23 | 31.43% | 32.18% | 17.00% | 18.50% |
| 24 | 24.54% | 25.63% | 17.88% | 19.79% |

On the primary window both models stay near 14–18% through the first year and then rise. Horizons 22 and 23 (May and June 2026) are above 30% for both. The tree does not separate from the naive there. The earlier window does not show that jump, which is one reason a single 24-month window cannot be treated as a stable error distribution.

## Segments

Quartiles use route passengers in the 12 months before the origin, so the label is known at forecast time. Q1 is the smallest quartile. Domestic means both airports have country code US. `visuals/error-by-segment.png` is the August 2024 cut.

August 2024, route-month WAPE:

| Segment | Naive year 1 | Naive year 2 | Tree year 1 | Tree year 2 |
|---|---:|---:|---:|---:|
| Q1 | 41.63% | 46.69% | 55.99% | 53.92% |
| Q2 | 27.67% | 31.84% | 30.29% | 32.56% |
| Q3 | 22.22% | 28.27% | 24.62% | 29.37% |
| Q4 | 13.35% | 20.02% | 14.02% | 20.61% |
| Domestic | 13.55% | 20.03% | 14.47% | 20.27% |
| International | 20.18% | 27.15% | 21.65% | 29.49% |

August 2023:

| Segment | Naive year 1 | Naive year 2 | Tree year 1 | Tree year 2 |
|---|---:|---:|---:|---:|
| Q1 | 44.86% | 45.80% | 57.96% | 56.63% |
| Q2 | 30.44% | 32.24% | 30.92% | 33.56% |
| Q3 | 24.56% | 27.59% | 25.49% | 27.58% |
| Q4 | 13.42% | 15.93% | 13.53% | 15.51% |
| Domestic | 13.29% | 16.18% | 13.39% | 15.66% |
| International | 22.73% | 24.41% | 23.76% | 25.24% |

The carried model's route-level uncertainty is this table. Small routes and international routes miss by more, in percent, than large domestic routes. There is no model interval around a naive forecast. The tree's P10/P90 fits were not produced, because the tree was not carried.

## Network picture

`visuals/backtest-network.png` plots the network actual from August 2022 through July 2026 against both forecasts on the primary test window, and the monthly percent error of each forecast. Percent error is (forecast − actual) / actual. Both forecasts sit above the actual through much of 2026. That is the +7.65% year-2 bias of the naive and the +10.04% year-2 bias of the tree.

## Intervals

The carried model has no fitted interval. The network band is approximate and is not a validated prediction interval. It multiplies the forecast by one plus the 10th and 90th percentiles of the monthly network percent errors, pooled from both backtests, separately for each horizon year. That pool is 24 months per year, from two windows that share 12 months.

| Horizon year | 10th percentile of network error | 90th percentile of network error |
|---|---:|---:|
| Year 1 | −7.44% | +3.38% |
| Year 2 | −9.37% | +13.87% |

Applied to the forward forecast, the year-1 band sums to 578,131,661.04 … 645,724,156.37 around a point total of 624,607,043.78. The year-2 band sums to 566,091,242.42 … 711,228,035.54 around the same point total. Coverage of this band cannot be tested on a third window. Do not read it as an 80% interval.

## Comparison with the published network scores

On August 2024 through July 2026, in passengers per month, the route naive's network MAE and RMSE match the seasonal naive in PR #77 and in `pm-modeling/initial-modeling.ipynb`: 3,113,174.10 and 4,319,584.87.

Those notebooks' other network models, transcribed rather than re-run, are all worse than that naive on MAE. Their WAPE, using the same actual total of 1,291,963,949.62, is 17.07% for ARIMA(1,1,1), 16.97% for SARIMAX(1,1,1)(1,1,1,12), 7.45% for recursive XGBoost, 10.55% for recursive random forest, and 7.46% for their simple average. Those WAPEs are network WAPEs. They are not route-month WAPEs, and they are not comparable with Lukas's 12-month route MAE.

The route tree's network MAE on this window is 3,465,121.15, between the naive and the recursive XGBoost, and its route-month WAPE is higher than the naive. That is the result the bar uses.

## Limitations

There is one clean 24-month window. The second origin overlaps it and is trained mostly on 2019–2023. Reporting both is as far as this file goes.

The network band is not validated out of sample. Route-level quantile coverage was not computed, because the tree was not carried.

Neither forecast projects growth. The forward path repeats August 2025 through July 2026 twice. Growth is an assumption American Airlines would set on top of this baseline.

Learning curves were not produced. The tree's settings were fixed rather than tuned, and the naive has none, so a learning curve would not have changed the decision.

If `OD_PAX` counts origin-to-destination trips, connecting passengers at a hub are not in an airport sum. Airport totals from this file are originating demand only.

The naive has no hyperparameters to finalize. The tree's settings are recorded in `baseline-decision.md` as the settings that were scored, not as a selected configuration. Both scored fits stopped at the iteration cap.
