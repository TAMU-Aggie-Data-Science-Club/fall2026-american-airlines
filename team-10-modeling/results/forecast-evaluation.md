# Forecast evaluation — Team 10 (Deliverable 5)

**Notebook:** `notebooks/05_two_year_forecast.ipynb` · **Scores:** `forecast_evaluation_scores.md`
(FAKE mode: `results/generated/`; REAL mode: `data/interim/team-10/`) · **Figures:** `visuals/`
(FAKE) or `data/interim/team-10/` (REAL)

## Test design

| item | design |
|---|---|
| Forecast origin | first month of the last 24 months of data |
| Horizon | 24 months, forecast once, never updated inside the window |
| Model | the one D4 carried (`decision.json`); not re-chosen |
| Training data | rows before the origin only; growth pairs outside the disruption window |
| Baseline | seasonal naive on the same forecast set (if the naive is carried, the other candidates are shown for context only) |
| Scoring | outer join with actual rows, zero-filled (see `model-comparison.md`) |
| Metrics | route-month WAPE (year 1, year 2, both), bias per year, network WAPE, WAPE by horizon month, WAPE by route-size quartile and by market |

## Uncertainty: the 80% network band

- **Year 1:** the 10th and 90th percentiles of log(actual ÷ forecast) for the monthly network total
  over the **first 12 months from the select origin**. Those months all fall before the test window.
- **Year 2:** no clean calibration months exist before the test window, so the year-1 log band is
  widened by √2, as if errors accumulate like a random walk.
- **Check:** the share of test months whose actual network total falls inside the band, reported per
  year. Coverage well below 80% means the band is too narrow and must not be shown as an 80% range.

## Results

<!-- FILL:d5_table -->
| | year 1 | year 2 | both |
|---|---|---|---|
| Carried model, route-month WAPE | 0.1510 | 0.2169 | 0.1829 |
| Seasonal naive, route-month WAPE | 0.1510 | 0.2169 | 0.1829 |
| Carried model, bias | +0.0075 | +0.0765 | +0.0409 |
| 80% band coverage | 8% | 0% | — |
<!-- /FILL -->

### Segments

Segments reported: route-size quartiles, domestic vs international, and "new route". New routes
have no forecast, so by construction they score 100% WAPE. Error metrics only: each segment's share
of passengers is business data and is kept out of the repo (NDA).

<!-- FILL:d5_segments -->
Across route volume tiers, forecast accuracy is strongest in the largest Q4 corridors, which achieve the lowest route-month WAPE at **0.1335** in Year 1 and **0.2002** in Year 2, whereas thin Q1 corridors exhibit substantially higher volatility at **0.4161** and **0.4667** WAPE (with Q2 at **0.2767 / 0.3184** and Q3 at **0.2223 / 0.2828**). By market type, domestic routes track closely with Q4 core accuracy (**0.1355** Year 1 WAPE and **0.2003** Year 2 WAPE), while international routes carry **6.6 to 7.1 percentage points** higher error across both forecast years (**0.2018** in Year 1 and **0.2715** in Year 2).
<!-- /FILL -->

## Does the data support a two-year forecast?

The data has a real monthly time axis (2019–2026 per the project docs), so a 24-month hold-out can
be built and scored honestly. That shows a two-year forecast **can be evaluated**. It does not show
that one **can be trusted**. To say anything stronger, the real run has to show three things:

1. year-2 WAPE close to year-1 WAPE;
2. the carried model at least matching the naive in year 2;
3. band coverage near 80%.

If any of these fail, the honest claim is: "the pipeline produces a 24-month forecast; at this
horizon it is no better than repeating last year, and its band is not calibrated."

<!-- FILL:d5_verdict -->
The holdout test evaluation supports a two-year forecast for **point predictions on high-volume core corridors**, but **does not support uncalibrated two-year uncertainty intervals or confident thin-route forecasting**. Using the carried **Seasonal naive** baseline, route-month WAPE remains steady at **0.1510** in Year 1 (`0.1335` on Q4 core routes; `0.1355` on domestic routes) and steps up moderately to **0.2169** in Year 2 (`0.2002` on Q4; `0.2003` on domestic), while aggregate Network WAPE holds at **0.0578** across the full 24-month horizon—substantially outperforming parametric and tree-based growth models that spike to 36%–47% WAPE late in Year 2 due to post-pandemic drift bias. However, empirical 80% prediction band coverage collapses to **8% in Year 1 and 0% in Year 2** (4% overall) because positive historical growth multipliers (`[1.055, 1.085]` in Year 1; `[1.078, 1.122]` in Year 2) systematically over-predicted volume across the test window, and error on thin Q1 corridors reaches **0.4667** in Year 2. Consequently, a two-year forecast is operationally viable for capacity planning on Q4 domestic routes if prediction intervals are recalibrated around zero net growth, whereas Year 2 projections for Q1/Q2 and international routes should be treated as directional baselines with wider operational buffers.
<!-- /FILL -->

## Limitations

- Only one test window exists, since 24 months cannot be split again. Stability is judged with
  earlier origins that sit closer to the disruption.
- The select and test windows share 12 months (see `model-comparison.md`). No setting was tuned on
  them.
- New routes are not forecast. Closures inside the window count as forecast error.
- Demand drivers outside the data (fares, capacity, competition, the economy) are not modelled.
- Everything committed so far is from FAKE mode.
