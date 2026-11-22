# Team 3 evaluation brief

These six questions are the final review. Each section states the decision, the module that carries it, and the result of the completed local run `20261122T054112Z` at source revision `cbc0ca8`. That run scored horizons 12–18 on matched cases. It is a retrospective evaluation of periods that had already been inspected. It is not an untouched holdout.

`python -m aaforecast submission-report` writes the same six headings from a completed backtest. This file is the reviewer copy of that run.

## 1. Problem-Solving Approach

A single regression on every market-month is a poor fit for this problem. Planners commit capacity about a year before departure, every market has its own season, the pandemic is a structural break, and the markets have to add up to a network the planner can stand behind.

The approach was to separate those pieces instead of asking one model to learn all of them:

1. A seasonal anchor for each directional market: what the market looks like if its recent level and its own season continue.
2. One network growth number, defaulting to zero, because the planner usually holds that view and a few years of history do not identify the next network-wide shock.
3. One pooled model for relative market movement only, scored against the simple rules on the same cases.

That order is the implementation in `src/aaforecast/model.py`, wired by `pipeline.py` and scored by `backtest.py`. The meeting-by-meeting chronology of how the team arrived here is not stored in the code.

## 2. Data Aggregation

The passenger extract is directional origin-destination-month traffic. `src/aaforecast/data.py` reads `DEP_MONTH`, `DEP_YEAR`, `ORG_AIRPRT`, `DES_AIRPRT`, `OD_PAX`, `OD_REV_USD`, and `SF_LOAD_TMS`. It builds a month, drops invalid months, negative values, and same-airport rows, and sums duplicate market-months so each origin-destination-month appears once. Absent target months are not scored as zeros. Historical gaps used as inputs are still zero-filled, and that rule still needs source-owner confirmation.

Airport attributes are joined for country, hub, runway, and coordinates. `validate.py` stops the run when the extract fails required checks. Airport and network scores are sums of the same market forecasts. Hub totals overlap, so hubs are not added together.

The release run used airport metadata, left external macro features off, and assumed a one-month label lag.

## 3. Initial Data Exploration

`src/aaforecast/eda.py` produced these findings on the supplied extract:

- Trailing-twelve-month network passengers are at 98.3% of 2019 volume.
- Across eligible markets, the 10th to 90th percentile of log movement is about −24.6% to +19.2%. Markets did not recover together.
- A Chow test places a structural break at March 2020 (F = 2.8, p = 0.0029). Training targets from March 2020 through December 2021 are excluded.
- Seasonal strength since 2022 is 0.81 for the network and 0.80 at the median of the top 200 markets, which is why each market keeps its own seasonal anchor.
- Network passengers since 2022 look stationary on ADF and KPSS. The twelve-month difference does not. Differencing is not used as the modeling transform.
- Trailing-twelve-month revenue per passenger is about $282. The scored target is passengers.

## 4. Creativity & Uniqueness

The forecast is a planning control, not only a score. A planner can set network growth without refitting, and the model is only allowed to say which markets outgrow that outlook. `model.py` trains one histogram gradient-boosted model on the log gap versus the anchor, demeaned within each origin and horizon, so the post-pandemic rebound does not look like a market-specific signal.

Three choices keep that claim checkable:

- The weight on the learned mix is chosen only from earlier forecasts whose outcomes have already matured (`backtest.py`). If that evidence does not exist, the weight is zero and the forecast is the anchor.
- Every method is scored on the same finite market-origin-horizon cases (`metrics.py`). A missing key fails the run.
- Feature ablations in `experiments.py` may turn off a named feature group. They cannot change the split, the horizon set, or the metric.

The domestic definition in the model is US/PR/VI/GU. The older business scripts also treat Canada and other territories as domestic. Those outputs are not interchangeable.

## 5. Methodology

Documented in `docs/methodology.md` and executed from `configs/release.json`.

- Target: observed directional O&D passengers per month.
- Horizons: 12 through 18 months. The command accepts 1–24, but those other horizons do not have this run's evidence. No 36-month model is claimed.
- Origins: January, April, July, and October 2024, plus January 2025. Each origin trains only on labels available at that origin minus the configured lag.
- Model: scikit-learn `HistGradientBoostingRegressor` on relative log movement, then rescaled so markets sum to the seasonal network outlook.
- References on the same cases: seasonal anchor, same month last year, and last year times trend.
- Uncertainty: intervals use only residuals that had matured by that origin. Later folds do not donate residuals to earlier folds.
- External census, FAA, and FRED features are implemented and excluded from this run. A current public file is not a historical vintage.

## 6. Accuracy & Performance

598,654 matched market cases. WAPE is total absolute error divided by total actual passengers. Positive bias is overprediction. MAE and RMSE are in passengers per market-month.

| Method | Market WAPE | Market bias | MAE | RMSE |
| --- | ---: | ---: | ---: | ---: |
| Anchor and Adjust | 18.0% | +2.4% | 565 | 1,409 |
| Seasonal anchor only | 18.2% | +2.4% | 571 | 1,428 |
| Same month last year | 19.0% | +3.2% | 595 | 1,494 |
| Last year × trend | 24.3% | +12.5% | 762 | 1,947 |

The learned mix improves the anchor by about 0.2 WAPE points at market level. It is strongest on larger markets (13.4% WAPE above 200k trailing passengers, 53.4% below 1k). At network level, same-month-last-year is better: 5.3% WAPE versus 5.6% for Anchor and Adjust. With network growth set to zero, the published network total equals the seasonal anchor by construction.

Error grows with horizon: market WAPE is 15.8% at 12 months and 19.0% at 18 months.

Nominal 80% intervals covered 82.9% of cases for the January 2024 origin and 68.8% for the January 2025 origin. Volume-weighted coverage on that last origin is 60.8%. The intervals are too narrow on the later folds. The production forward run, July 2027 through January 2028, uses the same anchor-scaled outlook and does not include the optional explanation charts.

These figures describe the retrospective run above. They are aggregate scores, not a license to treat the model as approved for scheduling.
