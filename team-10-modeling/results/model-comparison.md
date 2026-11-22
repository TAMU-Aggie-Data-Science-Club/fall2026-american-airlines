# Model comparison — Team 10 (Deliverable 4)

**Notebook:** `notebooks/04_model_comparison.ipynb` · **Code:** `src/forecasting.py`
**Scores:** written by the notebook to `model_comparison_scores.md` (FAKE mode: `results/generated/`;
REAL mode: `data/interim/team-10/`).

## The question being scored

A **two-year path**, not next month. At an origin T, one forecast covers every route-month from T
to T+23. It is made once and never refreshed with actuals from inside the window.

## Validation

| origin | definition | role |
|---|---|---|
| test | first month of the last 24 months of data | D5 only |
| select | test − 12 months | main D4 comparison |
| stability | test − 24 months (its window ends the month before the test window) | robustness check |

- **Forecast set at T:** every route seen in the 12 months before T × the 24 months from T.
  The route roster is fixed at T, because which routes will fly is unknown then.
- **Scoring join:** forecasts are outer-joined to the actual rows of the window. A forecast route-month
  with no actual counts as 0 passengers, and an actual route-month with no forecast (a new route)
  counts as forecast 0.
- **Why the select window overlaps the test window:** an origin early enough to avoid any overlap
  puts all of its training history inside the 2020–21 disruption, so the growth models would have
  no clean pairs to learn from. To stop the 12 shared months from steering the result:
  - every model setting is fixed in `src/forecasting.py`;
  - nothing is searched or tuned;
  - the carry rule was written before any score existed.

## Metrics (same for every candidate)

- **Route-month WAPE** for year 1 (months 1–12), year 2 (months 13–24) and both. This is the primary
  metric: total absolute error ÷ total passengers, so routes count in proportion to their size.
- **Bias:** forecast total ÷ actual total − 1, per year.
- **Network WAPE:** the same measure on the monthly network totals. It is smaller, because route
  errors cancel.
- **Route-month MAE and RMSE**, in passengers.

## Candidates

| candidate | how it forecasts | included because | limitation |
|---|---|---|---|
| Seasonal naive | Same route, same month, year before T, for both years. A month the route skipped falls back to the route's median month that year. | The reference every model must beat. No settings, and leakage-safe for all 24 months. | No growth: any level shift becomes year-2 bias |
| Trend-adjusted naive | Naive × g (year 1) and × g² (year 2), where g blends route and network growth, capped 0.8–1.25 | The simplest way to add growth | A growth rate measured over one year can be noise |
| Ridge growth | Ridge on log growth: airport one-hots, month, distance, market, coordinates | Readable, shares growth across routes through airports | Additive in log space. Cannot capture route-specific interactions. |
| Gradient-boosted growth | Histogram GBT on the same inputs, airports as native categories | Captures interactions (market × month × size) | Harder to explain. Can overfit thin airports. |

**Not fitted, with reasons:**
- **Network ARIMA / SARIMAX:** one series cannot produce a route plan.
- **Per-route ARIMA:** thousands of short series broken by 2020.
- **LSTM:** short clean history, and TensorFlow would be a new dependency.
- **Ensembles and hyperparameter search:** nothing may be tuned on the select window.

## Results

<!-- FILL:d4_table -->

| origin | candidate | WAPE y1 | WAPE y2 | bias 24m |
|---|---|---|---|---|
| select | Seasonal naive | 0.1544 | 0.1811 | -0.6425 |
| select | Trend-adjusted naive | 0.1733 | 0.2944 | +0.1317 |
| select | Ridge growth | 0.1874 | 0.2718 | -0.0740 |
| select | Gradient-boosted growth | 0.1621 | 0.2522 | +0.0729 |
| stability | Seasonal naive | 0.2098 | 0.2483 | -0.1586 |
| stability | Trend-adjusted naive | 0.2457 | 0.3930 | +0.1831 |
| stability | Ridge growth | 0.2170 | 0.2722 | -0.1553 |
| stability | Gradient-boosted growth | 0.2104 | 0.2540 | -0.0800 |
<!-- /FILL -->

**Why Random Forest and XGBoost are not separate candidates:** the gradient-boosted growth model
is the tree-ensemble candidate. It uses scikit-learn's histogram gradient boosting, which is the
same family as XGBoost and needs no extra dependency. A Random Forest on the same inputs would add
a second tree ensemble with no new idea to test, and it cannot extrapolate growth either. ARIMA and
LSTM are excluded above.

<!-- FILL:d4_readout -->
Across both the select (2023-08-01) and stability (2022-08-01) forecast origins, Seasonal naive achieves the lowest 24-month route-level error (0.1677 WAPE at select and 0.2297 WAPE at stability), outperforming every parametric and trend-adjusted candidate while executing in under one second (0.8s–0.9s). At the select origin, Seasonal naive leads across both forecast years (0.1544 in y1 and 0.1811 in y2), whereas Gradient-boosted growth (0.2070 24m WAPE) and Ridge growth (0.2294) degrade sharply in Year 2 (0.2522 and 0.2718), and Trend-adjusted naive suffers severe upward drift bias (+0.1317, 0.2944 y2 WAPE). On the harder stability split—where the 12-month anchor falls inside the 2021–2022 recovery period—all models exhibit negative volume bias (-0.0800 to -0.1586) except Trend-adjusted naive (+0.1831), yet Seasonal naive still retains the top rank on overall 24-month WAPE (0.2297 vs. 0.2330 for Gradient-boosted growth), failing all challenger models at the selection gate and establishing Seasonal naive as the carried baseline.
<!-- /FILL -->

## Trade-offs to discuss once the real scores exist

1. **Does any model beat "repeat last year" in year 2?** Completed teams on this data found that
   the seasonal naive is hard to beat over two years. A tie should be read as a win for the naive:
   it has no settings and nothing to break.
2. **Bias vs WAPE:** a model can lower WAPE while adding a systematic over-forecast. Report both.
3. **Size segments:** a model that helps large routes but hurts small ones may still lower total
   WAPE. `05` reports both.
4. **Cost and explainability:** the naive and Ridge can be explained in one sentence. Trees need
   feature-importance tooling.
