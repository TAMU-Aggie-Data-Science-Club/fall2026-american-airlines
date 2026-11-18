# Two-year passenger forecast

Handover for American Airlines, and the Deliverable 5 report. The numbers sit in `forecast-evaluation.md`. The route-month file is local only, at `data/processed/forecast-2026-08-to-2028-07.csv`, and a PM sends it through the NDA channel. It is not in git.

## 1. Summary for American Airlines

The forecast is passengers on one origin–destination route in one departure month, for August 2026 through July 2028, made once in August 2026. The network total is the sum of the routes.

The carried forecast repeats each route's most recent year. On the test that matches this use — trained through July 2024, scored on August 2024 through July 2026 without seeing those months — it missed route-month passengers by 18.29% (15.10% in the first year, 21.69% in the second). The two-year network total was 4.09% above actual. A gradient-boosted tree on the team's feature frame was less accurate on that window and was not carried.

The forward network total is 624,607,043.78 passengers in August 2026–July 2027 and the same number again in August 2027–July 2028. An approximate band, wider in the second year, is drawn around that path in `visuals/forward-forecast.png`.

Use it as the seasonal baseline for route capacity and frequency, and for a network demand path in fleet and schedule planning. Put growth, new routes, and capacity changes in by hand. The main caveat is that one historical window, which includes the 2020–21 collapse, cannot tell American Airlines how large the error would be after another shock.

## 2. Question and horizon

The forecast unit is the route-month. `OD_PAX` is an estimated count and can be fractional. A network or airport total is a sum of route forecasts, not a separate model.

The origin T is the first day of a month. Everything dated before T is known. Nothing dated T or later is. "Two years ahead" means one forecast of all 24 months, judged on every month and on the 12- and 24-month totals. It is not a next-month score, and it is not the score of month 24 alone.

August 2024 is the latest origin whose 24 months are fully observed. August 2023 is a second origin on the same design. August 2026 is the forward origin. All three have January–July of the latest year in the history, which is the position the forward forecast is in.

## 3. Data and leakage controls

The frame is the Deliverable 3 feature set in `notebooks/pri-biswal.ipynb`: 22 columns, 11 of them categorical airport and calendar attributes, none of them `OD_PAX` or `OD_REV_USD`. The carried forecast does not use that frame. The tree that was rejected does. The functions are copied into `notebooks/two-year-forecast.ipynb` and were not changed.

What is excluded, and why:

| Input | h = 1 | h = 2–12 | h = 13–24 |
|---|---|---|---|
| Same-month `OD_REV_USD` | Leak | Leak | Leak |
| `OD_PAX` lag 1, or a rolling mean ending last month | Known | Leak | Leak |
| `OD_PAX` lag 12 | Known | Known | Leak. The year-ago month sits inside the test window. |
| `OD_PAX` lag 24 | Known | Known | Known |
| Earlier forecasts fed back as lags | Errors compound | Errors compound | Errors compound |
| The 22 Deliverable 3 columns | Known | Known | Known |
| Which route-months will exist inside the window | Leak | Leak | Leak. The forecast set is fixed at T. |
| Settings chosen after seeing the test score | Leak | Leak | Leak |

The naive uses the year before T for both forecast years, which is the safe row of that table. The tree uses only the 22 columns. Lukas's forest uses a year-ago lag, which is why it stops at 12 months and is not in this score.

## 4. Training procedure

The carried model is not trained. For each route-month in the forecast set, the forecast is that route's passengers in the same calendar month of the year before the origin. Both forecast years use that year. There is no parameter, no seed, and no iteration.

The rejected tree was fit only so the bar could be scored. `HistGradientBoostingRegressor(loss="poisson", max_iter=1000, early_stopping=True, random_state=9)`, with every other argument at the scikit-learn 1.9.1 default: learning rate 0.1, 31 leaves, 20 rows per leaf, a 10% random validation slice drawn from the training period, and a stop after 10 rounds without improvement. Training rows are realized route-months dated before the origin, including 2020 and 2021. Both backtest fits ran all 1,000 iterations. Early stopping did not end them. The settings were not changed after the score.

The notebook was run with Python 3.14.7, pandas 3.0.6, numpy 2.5.4, scikit-learn 1.9.1, matplotlib 3.11.2, and statsmodels 0.15.0 from the project's `.venv`. No requirements file is tracked. To rerun from the repository root:

```text
.venv/bin/jupyter nbconvert --to notebook --execute --ExecutePreprocessor.timeout=-1 team-09-modeling/notebooks/two-year-forecast.ipynb --output-dir <scratchpad>
```

## 5. Test design

Origins, windows, the forecast set, the zero-fill rule, the metrics, and the baseline are in `forecast-evaluation.md`. In short: fit or copy only from dates before T, forecast 24 months at once, score an outer join that keeps the actual total, and compare the tree with the naive inside that same frame. The August 2024 network naive matches the published MAE and RMSE to the passenger.

## 6. Results against the baseline

The baseline and the carried forecast are the same model. The table is the test, not a second contest.

August 2024 through July 2026:

| | Year 1 | Year 2 | Both years |
|---|---:|---:|---:|
| Route-month WAPE | 15.10% | 21.69% | 18.29% |
| Network total versus actual (bias) | +0.75% | +7.65% | +4.09% |
| Network monthly WAPE |  |  | 5.78% |

The rejected tree on the same window: route-month WAPE 16.15%, 22.42%, and 19.18%; network bias +2.99%, +10.04%, and +6.40%; network monthly WAPE 6.44%. Network MAE 3,465,121.15 against the naive's 3,113,174.10.

August 2023 through July 2025, naive: route-month WAPE 15.44%, 18.11%, and 16.77%; network bias −6.60%, −5.90%, and −6.25%. The tree was closer on the network total in this window and still higher on year-1 route-month WAPE. Detail is in the appendix.

## 7. Diagnostics

`visuals/backtest-network.png` shows the network actual from August 2022 through July 2026, both forecasts on the test window, and the monthly percent error. The second year of the test runs high: the naive's year-2 network total is 7.65% above actual.

`visuals/error-by-horizon.png` is route-month WAPE by horizon for both models and both origins. The primary window gets worse after month 12, and months 22 and 23 are above 30% for both models. The earlier window does not repeat that spike. The tree tracks the naive rather than correcting it.

`visuals/error-by-segment.png` splits August 2024 by the route's prior-year volume and by domestic versus international. The bottom volume quartile misses by 41.63% in year 1 and 46.69% in year 2. The top quartile misses by 13.35% and 20.02%. Domestic is 13.55% and 20.03%. International is 20.18% and 27.15%. The same pattern shows up at the August 2023 origin. Because the naive was carried, this table is the route-level uncertainty statement. There is no feature-importance chart and no P10/P90 coverage.

Across a route's own 24-month total, WAPE is 11.32% on the primary window. That is the number closer to a capacity plan than the route-month WAPE, because months inside a route cancel.

## 8. The two-year forecast

The forward origin is August 2026. The path is August 2025–July 2026, repeated. Both 12-month network totals are 624,607,043.78. The two-year total is 1,249,214,087.56.

The band is the approximate network band from the two backtests. It is not an 80% interval. Summing the monthly endpoints gives 578,131,661.04 to 645,724,156.37 for the first year and 566,091,242.42 to 711,228,035.54 for the second. The second year is wider because the backtest errors were wider there.

| Month | Forecast | Band low | Band high |
|---|---:|---:|---:|
| 2026-08 | 56,721,019.37 | 52,500,556.10 | 58,638,679.70 |
| 2026-09 | 50,557,191.40 | 46,795,362.51 | 52,266,461.11 |
| 2026-10 | 57,478,420.11 | 53,201,600.63 | 59,421,687.13 |
| 2026-11 | 50,864,205.84 | 47,079,532.81 | 52,583,855.29 |
| 2026-12 | 56,053,345.45 | 51,882,562.06 | 57,948,432.64 |
| 2027-01 | 45,147,304.83 | 41,788,011.51 | 46,673,673.65 |
| 2027-02 | 45,974,031.46 | 42,553,223.56 | 47,528,350.78 |
| 2027-03 | 50,866,217.12 | 47,081,394.43 | 52,585,934.57 |
| 2027-04 | 53,886,166.08 | 49,876,636.86 | 55,707,983.89 |
| 2027-05 | 49,808,897.98 | 46,102,747.66 | 51,492,868.91 |
| 2027-06 | 49,673,870.64 | 45,977,767.35 | 51,353,276.48 |
| 2027-07 | 57,576,373.50 | 53,292,265.56 | 59,522,952.19 |
| 2027-08 | 56,721,019.37 | 51,407,156.94 | 64,587,134.55 |
| 2027-09 | 50,557,191.40 | 45,820,782.17 | 57,568,502.12 |
| 2027-10 | 57,478,420.11 | 52,093,601.21 | 65,449,572.22 |
| 2027-11 | 50,864,205.84 | 46,099,034.21 | 57,918,093.56 |
| 2027-12 | 56,053,345.45 | 50,802,033.51 | 63,826,867.09 |
| 2028-01 | 45,147,304.83 | 40,917,716.41 | 51,408,368.26 |
| 2028-02 | 45,974,031.46 | 41,666,991.83 | 52,349,745.98 |
| 2028-03 | 50,866,217.12 | 46,100,857.06 | 57,920,383.76 |
| 2028-04 | 53,886,166.08 | 48,837,884.57 | 61,359,141.60 |
| 2028-05 | 49,808,897.98 | 45,142,592.01 | 56,716,434.78 |
| 2028-06 | 49,673,870.64 | 45,020,214.59 | 56,562,681.74 |
| 2028-07 | 57,576,373.50 | 52,182,377.92 | 65,561,109.86 |

The chart is `visuals/forward-forecast.png`: actual network passengers from January 2019 through July 2026, then this path and band.

The route-month table has columns `ORG_AIRPRT`, `DES_AIRPRT`, `DEP_YEAR`, `DEP_MONTH`, and `FORECAST_PAX`. It lives at `data/processed/forecast-2026-08-to-2028-07.csv` and is git-ignored. A PM should send it to American Airlines through the NDA channel. This report does not contain route-level values.

## 9. What American Airlines can and cannot do with it

Can: plan route capacity and frequency by season, using the route-month path as the no-growth baseline. Use the network total as the demand baseline for fleet and schedule planning.

Can, with care: add routes up to an airport and treat the sum as originating demand. If these counts are origin-to-destination trips, a connecting passenger at a hub is not in the hub's sum. Ask American Airlines before using an airport total for staffing.

Cannot: set a fare. There are no fares and no elasticities, so this is a seasonal demand outlook only. Cannot open a route that is not in the last year, and cannot see a route that exits, because the forecast set is the last year's route-months. Cannot read unconstrained demand: `OD_PAX` is carried traffic and is capped by seats offered. Cannot forecast revenue. Cannot plan a week or a day. The grain is the month.

## 10. Does the data support two years?

What this forecast can and cannot claim. We tested the forecast the way it would be used: trained on data through July 2024, then scored on all 24 months from August 2024 to July 2026 without seeing any of them. Over those two years it missed route-month passengers by 18.29% (year 1 15.10%, year 2 21.69%) and the two-year network total was 4.09% above actual. Simply repeating the last year missed by those same amounts, because that repeat is the forecast. The gradient-boosted tree missed route-month passengers by 19.18% (year 1 16.15%, year 2 22.42%) and its two-year network total was 6.40% above actual. That is one two-year window. A second window starting August 2023, which overlaps it, missed route-month passengers by 16.77% (year 1 15.44%, year 2 18.11%) and its two-year network total was 6.25% below actual. The history includes the 2020–21 collapse and recovery, so we cannot say how large the errors would be after another shock. The forecast carries the most recent year's level and seasonal pattern forward. It does not project growth, new routes, route exits, or capacity changes. Use it as the baseline for two-year planning: set growth as an explicit assumption, plan with the band rather than the point, and re-score it every month as actuals arrive.

## 11. Risks

A shock on the scale of 2020 is outside both test windows. The forecast assumes zero growth and will be high for two years if traffic falls, or low if it keeps rising. Route churn is invisible: a new route is forecast at 0 only if it is outside the set, and in the two test windows those routes were none of the actual passengers, which will not stay true if the network changes. American Airlines' own capacity decisions shape future `OD_PAX`, so the forecast is not an unconstrained ask. The counts are estimates. The file covers the airports in the extract, not every airport American flies. One test window, plus a second window that overlaps it, is the whole evidence.

## 12. Monitoring and follow-up data

Each month, score the new actual with the same functions: network percent error against the band, route-month WAPE by horizon year, and bias of the total. If the network stays outside the band for three months running, re-forecast. Refit the comparison, and refresh the repeated year, every August.

Ask American Airlines for seats offered by route-month, past fares or revenue to be used only as lags, bookings on hand, a small set of macro indicators, and competitor capacity. Any of those would be a new source and needs a license and sensitivity check before it enters the pipeline.
