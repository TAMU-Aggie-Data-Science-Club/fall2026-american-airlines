# Team 3 evaluation brief

These six questions are the final review. Each section states the decision, the module that carries it, and the result of the completed local run `20261122T062603Z` at source revision `7b5618c`. That run scores horizons 12–24 on matched cases, and it drops a horizon when the target month is not in the extract yet. It is a retrospective evaluation of periods that had already been inspected. It is not an untouched holdout.

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

The supplied extract has 1,533,522 rows from January 2019 through July 2026. It has no duplicate origin-destination-month keys and no key with two load timestamps, so summing duplicate keys does not change this file. The release run used airport metadata, left external macro features off, and assumed a one-month label lag.

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

`geo.MODEL_DOMESTIC` is US/PR/VI/GU and is what the scored forecast uses. `geo.BUSINESS_US_CA` also includes Canada, American Samoa, and the Northern Mariana Islands. The COVID, business, and macro scripts use that second set on purpose. Their “domestic” totals are not the model’s domestic segment.

## 5. Methodology

Documented in `docs/methodology.md` and executed from `configs/release.json`.

- Target: observed directional O&D passengers per month.
- Horizons: 12 through 24 months. Months 19–24 are scored for the January, April, and July 2024 origins, because those target months fall on or before July 2026. The October 2024 and January 2025 origins are scored through the months that exist and are not deleted for lacking a 24-month actual. No 36-month model is claimed.
- Origins: January, April, July, and October 2024, plus January 2025. Each origin trains only on labels available at that origin minus the configured lag.
- Model: scikit-learn `HistGradientBoostingRegressor` on relative log movement, then rescaled so markets sum to the seasonal network outlook.
- References on the same cases: seasonal anchor, same month last year, and last year times trend.
- Uncertainty: intervals use only residuals that had matured by that origin. Later folds do not donate residuals to earlier folds.
- External census, FAA, and FRED features are implemented and excluded from this run. A current public file is not a historical vintage.

## 6. Accuracy & Performance

957,727 matched market cases. WAPE is total absolute error divided by total actual passengers. Positive bias is overprediction.

| Method | Market WAPE | Network WAPE |
| --- | ---: | ---: |
| Anchor and Adjust | 19.6% | 7.0% |
| Seasonal anchor only | 19.6% | 7.0% |
| Same month last year | 20.2% | 6.8% |
| Last year × trend | 26.6% | 15.3% |

On every scored fold the matured evidence selected a mix weight of zero, so the evaluated Anchor and Adjust forecast is the seasonal anchor. A grid that peeks at the same folds prefers a weight of 0.50 and a market WAPE of 19.62% against 19.99% at weight zero. That peek is not the result. The forward forecast, made after more outcomes had matured, uses a weight of 0.25. About 95% of its market rows move off the anchor, by a median of about 12 passengers, and the network total is then scaled back to the seasonal outlook.

Error grows with horizon. Market WAPE is 15.9% at 12 months, 19.2% at 18 months, and 22.1% at 24 months. At 24 months, same-month-last-year is slightly better (21.8%). Domestic market WAPE is 18.0%. International is 25.1%. Markets above 200,000 trailing passengers are at 15.0%. Markets under 1,000 are at 55.1%.

Nominal 80% intervals covered 82.8% of cases for the January 2024 origin and 69.0% for the January 2025 origin. Volume-weighted coverage on that last origin is 60.9%. The bands are too narrow on the later folds. They were not widened with those folds' own outcomes.

The forward network outlook runs from July 2027 through July 2028. With network growth left at zero, each month's network total equals that month's seasonal anchor. Same-month-last-year remains the stronger network benchmark in the backtest (6.8% versus 7.0%).

These figures describe the retrospective run above. They are aggregate scores, not a license to treat the model as approved for scheduling.
