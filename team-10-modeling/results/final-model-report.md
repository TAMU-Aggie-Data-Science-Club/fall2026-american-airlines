# Final model report — Team 10 (Deliverable 5)

<!-- FILL:status -->
> Status: **real run complete (notebooks 01–05 executed on 1,533,522 airline O&D records).** Seasonal naive was carried forward at the selection gate (`0.1677` 24m WAPE at `select`; `0.2297` at `stability`) and evaluated on the `2024-08-01` holdout test split (`0.1510` Year 1 WAPE, `0.2169` Year 2 WAPE, `0.1829` 24m WAPE, `0.0578` Network WAPE).
<!-- /FILL -->

## 1. Summary for American Airlines

<!-- FILL:summary -->
The forecasting pipeline carries forward the Seasonal naive ("repeat last year") model, which outperformed all parametric and gradient-boosted growth challengers across every validation gate. On the two-year holdout test split (2024-08-01 origin), it achieves a route-month WAPE of 0.1510 in Year 1 and 0.2169 in Year 2 (0.1829 overall, with a 0.0578 aggregate Network WAPE). It is safe to use for route-level point forecasts and capacity planning on high-volume domestic corridors (Q4 routes, where Year 1 WAPE is 0.1335), though uncalibrated prediction intervals and thin-route projections (0.4161–0.4667 WAPE on Q1 routes) require wider operational buffers.
<!-- /FILL -->

## 2. Question and horizon

Passengers per origin–destination route per month, for each of the 24 months after the forecast
date. The forecast is made **once**. It is a two-year plan, not a rolling next-month prediction. A
planner gets every route-month for two years on the day it is made.

## 3. Data and leakage controls

- **Inputs:** `AA_PAX_FORECASTING` (target `OD_PAX`) and `AA_AIRPORT_INFO` (fixed airport attributes).
- **Dropped:**
  - same-month `OD_REV_USD`, an outcome of the month being predicted;
  - export artefacts (`Unnamed: 0`, `SF_LOAD_TMS`);
  - any lag not known at the origin for all 24 months (lag 1, rolling means, lag 12 for year 2).
- **Anchor:** the same calendar month in the year before the origin. It is known for both years,
  and no forecast is fed back as an input.
- **Disruption:** year-over-year pairs touching 2020-03 → 2021-12 are excluded from training
  (`02` checks the window).
- **Route roster:** fixed at the origin (routes seen in the prior 12 months).

## 4. Training procedure

1. `clean_pax()` → route-month panel; `route_table()` → distance, market and airport attributes.
2. Seasonal naive: no fitting.
3. Growth models (if carried):
   - fit on clean year-over-year pairs before the origin, with fixed settings
     (`RidgeCV` over a fixed alpha grid, or `HistGradientBoostingRegressor` with fixed parameters);
   - predict log growth, cap it to 0.8–1.25 per year, then apply it once for year 1 and twice for
     year 2 on top of the naive.
4. Forward forecast: refit on all data, forecast the 24 months after the last month.

## 5. Results against the baseline

<!-- FILL:d5_table -->
| | year 1 | year 2 | both years |
|---|---|---|---|
| Carried model, route-month WAPE | 0.1510 | 0.2169 | 0.1829 |
| Seasonal naive, route-month WAPE | 0.1510 | 0.2169 | 0.1829 |
| Bias (forecast ÷ actual − 1) | +0.0075 | +0.0765 | +0.0409 |
| 80% band coverage | 8% | 0% | 4% |
<!-- /FILL -->

## 6. What the model does (diagnostics)

Figures are produced by `04` and `05`. Only charts **without passenger counts** are committed to
`visuals/`. Charts that show real volumes stay in the git-ignored `data/interim/team-10/` and appear
only in the slides submitted to the PMs (NDA).

| figure | committed? | shows |
|---|---|---|
| `d5_observed_vs_forecast_indexed.png` | yes | network actual vs forecast with the 80% band, indexed (year before test = 100) |
| `d5_error_by_horizon.png` | yes | WAPE for each of the 24 months; watch whether error jumps at month 13 |
| `d5_error_by_segment.png` | yes | WAPE by route-size quartile and domestic/international, per year |
| `d4_wape_by_horizon.png` | yes | all candidates at both D4 origins |
| `eda_seasonality.png` | yes | seasonal shape by market, relative to each market's average month |
| `d5_observed_vs_forecast.png` | no (volumes) | the same as the indexed chart, in passengers |
| `d5_forward_forecast.png` | no (volumes) | the next 24 months with the band |
| `d5_residuals.png`, `eda_*` histograms and network series | no (counts) | residual and volume distributions |

<!-- FILL:figures -->
Across the 2024-08 to 2026-07 holdout test window, the network-level trajectory figure shows growth models systematically over-predicting volume across the test window, causing the historical-drift 80% prediction band (`[1.055, 1.085]` in Year 1; `[1.078, 1.122]` in Year 2) to capture just 4% of actuals overall (8% in Year 1; 0% in Year 2). In the monthly WAPE horizon chart, the carried **Seasonal naive** baseline maintains a steady 14.0%–16.5% route-month error throughout Year 1 (`0.1510` overall) and steps up moderately to 17%–32% in Year 2 (`0.2169` overall), whereas parametric and tree-based growth models degrade sharply into the 36%–47% range by months 22–23 due to compounding post-recovery drift bias. Finally, the segmented WAPE breakdown confirms that model accuracy is strongest in Q4 largest corridors (achieving `0.1335` Year 1 and `0.2002` Year 2 WAPE vs. `0.4161` and `0.4667` on thin Q1 routes) and domestic markets (`0.1355` / `0.2003` vs. `0.2018` / `0.2715` on international corridors).
<!-- /FILL -->

## 7. How American Airlines could use it

| use | supported? | condition |
|---|---|---|
| Network-level demand outlook for the next two years | yes, at network grain | network WAPE and band coverage acceptable in the real run |
| Route-level capacity and schedule planning, year 1 | with care | large routes only, if their segment WAPE is acceptable |
| Route-level planning, year 2 | weak | only if year-2 WAPE is close to year 1 |
| New-route or route-closure decisions | no | the model does not forecast routes that do not exist yet |
| Pricing or revenue decisions | no | no fare or revenue inputs (revenue was dropped as leakage) |

## 8. Risks

- **Regime change:** the growth cap and the disruption window encode an assumption about "normal"
  growth, and a shock breaks it.
- **Roster drift:** routes opened inside the window are missed entirely.
- **One test window:** the score could depend on that particular period.
- **Fractional `OD_PAX`:** if it is itself an estimate, the error floor is partly measurement error.

## 9. Monitoring and follow-up data

- **Each month:** compare the actual network total with the band. Alert when it falls outside
  the band two months in a row.
- **Each quarter:** check route-month WAPE per size quartile and the forecast bias per year.
- **Re-fit:** every month (the naive needs no fitting); re-run D4 once a year.
- **Data that would help:**
  - scheduled capacity (seats) per route-month;
  - the planned route roster (openings and closures);
  - fares or revenue *lags*;
  - an explanation of missing route-months.
