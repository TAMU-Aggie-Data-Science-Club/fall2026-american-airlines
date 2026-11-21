# Team 02 — Forecast evaluation (Deliverable 5) — Sahith

Code: `notebooks/two_year_forecast.ipynb`, sections 3–4. **NDA:** rounded numbers only.

## Test design
Hold out the last 24 months; forecast all of them from a single forecast date using only information available
then. Baseline: **same month two years ago** (the honest seasonal naive for a two-year horizon).

## Results

| Period | Model WAPE | Baseline WAPE | Model MAE | Baseline MAE |
|---|---|---|---|---|
| All 24 months | | | | |
| Year 1 (months 1–12) | | | | |
| Year 2 (months 13–24) | | | | |

- **80% prediction interval coverage:** *(target about 80%; above = too wide, below = too narrow)*
- **Network total over 24 months:** model off by *(±%)*, baseline off by *(±%)*.

## Diagnostics (charts in `data/visuals/`, local only)
- **Observed vs forecast (`observed_vs_forecast.png`):** *(does the forecast follow the seasonal shape? bias up or down?)*
- **Error by months ahead (`error_by_horizon.png`):** *(does error grow with the horizon?)*
- **Residuals (`residuals.png`):** *(centred on zero? skewed toward under-forecasting?)*
- **Segments (`error_by_segment.png`):** *(which segments are hardest — new routes, small routes, international?)*
- **Feature importance (`feature_importance.png`):** *(what drives the forecast)*

## Does the data support a two-year forecast?
*(Years of history available, share of rows with a two-years-ago value, whether year-2 error is still clearly better
than the baseline. If not, state the most honest claim we can make.)*

## Limitations
- Trees cannot extrapolate long-run growth; trend beyond the training range is flattened.
- New routes have no history and the largest errors.
- Network-level interval is a rough sum of route intervals (wider than a true network interval).
