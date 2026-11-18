# Model comparison

The question is a two-year path, not the next month. One forecast is made at an origin T for every route-month from T through T+23, and it is not updated with actuals from inside that window. Route-month WAPE is the primary metric. Network MAE and RMSE, in passengers per month, are the metrics shared with the earlier network models.

Two comparisons are not one comparison. A network WAPE is smaller than a route-month WAPE because route errors cancel when the routes are summed. Models are ranked only inside a grain.

## Validation

The passenger file runs from January 2019 through July 2026, 1,533,522 route-months over 91 months, with a unique origin, destination, year, and month.

The primary origin is August 2024. Training uses rows dated before that month (January 2019 through July 2024, 67 months). The score window is August 2024 through July 2026. This is the window in PR #77 and in `pm-modeling/initial-modeling.ipynb`. A stability origin at August 2023 scores August 2023 through July 2025. The two windows overlap by 12 months. That is the only second window the file can support.

The forecast set is fixed at T: every route-month in the 12 months before T, copied forward 12 months and 24 months. An actual row missing from that set counts as 0 passengers. A route-month that appears in the window but not in the forecast set counts as forecast 0. In both windows, routes outside the forecast set contributed 0% of actual passengers.

The route-level seasonal naive copies each route's passengers from the same calendar month in the year before T into both forecast years. Summed to the network, the August 2024 score is the published seasonal naive: MAE 3,113,174.10 and RMSE 4,319,584.87.

## Network candidates

Grain: one network total per month. Window: August 2024 through July 2026. Horizon: 24 months, forecast once. None of these models uses the 22-column Deliverable 3 frame as the model input. Deekshitha Kumari's notebook calls `build_features`, then sums `OD_PAX` and drops the features. The PM notebook aggregates the raw file to the same monthly series. Their MAE and RMSE are comparable with each other and with the network row of the route-level models below. The WAPE column is MAE × 24 ÷ 1,291,963,949.62, the actual network total over that window.

| Candidate | Uses D3 frame | MAE | RMSE | WAPE | Comparable on this grain |
|---|---|---:|---:|---:|---|
| Seasonal naive | No. It repeats the last 12 network totals. | 3,113,174.10 | 4,319,584.87 | 5.78% | Yes |
| ARIMA(1,1,1) | No. Fit on the network series. | 9,187,860.27 | 10,301,932.82 | 17.07% | Yes |
| SARIMAX(1,1,1)(1,1,1,12) | No. Fit on the network series. | 9,135,953.67 | 10,790,937.09 | 16.97% | Yes |
| Recursive XGBoost | No. Lag and calendar features on the network series. | 4,012,361.82 | 5,256,153.92 | 7.45% | Yes |
| Recursive random forest | No. Same design as the recursive XGBoost. | 5,680,803.49 | 6,761,747.84 | 10.55% | Yes |
| Simple average | No. The average of the network forecasts in the PM notebook. | 4,013,350.49 | 5,184,236.17 | 7.46% | Yes |

Numbers other than the seasonal naive are transcribed from `deekshitha-arima.ipynb` and `pm-modeling/initial-modeling.ipynb`. They were not re-run. The seasonal naive was recomputed from route forecasts and matched the published network figure to the passenger.

**Seasonal naive.** Included because a 12-month seasonal series needs a "repeat last year" reference before any fitted model is worth carrying. On this window it has the lowest network MAE and RMSE of the network candidates. It does not estimate a trend, so a level shift becomes a bias for the whole second year. That is also what makes it leakage-safe: the only history it needs is the year before the origin.

**ARIMA(1,1,1).** Included as the small non-seasonal baseline from the modeling plan. It has no yearly cycle, so the summer peak is a large residual every year. The network error is about three times the naive. A network ARIMA also cannot say which route the passengers are on.

**SARIMAX(1,1,1)(1,1,1,12).** Included because the series has a yearly cycle and the non-seasonal ARIMA misses it. The seasonal term recovers the shape and still loses to the naive, because the differenced trend keeps climbing. Same limit as ARIMA: one series, no route.

**Recursive XGBoost.** Included as the tree model on the network series, with lags fed back one month at a time. Errors compound across 24 steps, and a lag of 12 months reaches into the test window after the first year. It beats ARIMA and loses to the naive. It is not a route forecast.

**Recursive random forest.** Included beside XGBoost on the same lags. It has the same leakage and compounding limits and a larger network error. It is not Lukas's route-level forest.

**Simple average.** Included because the modeling plan averages the available network forecasts. Averaging models that are already worse than the naive does not beat the naive. An average also has no single leakage story: it inherits the worst member.

## Route-level candidates

Grain: one origin–destination route in one departure month. The August 2024 window is the primary score. The August 2023 window is the stability check. These two rows use the same forecast set, the same zero-fill, and the same metrics, so they are comparable with each other. They are not comparable with the network table above, and they are not comparable with Lukas's forest.

| Candidate | Window | Horizon | Uses D3 frame | Route-month WAPE | Network MAE | Network RMSE | Comparable with the route naive |
|---|---|---|---|---:|---:|---:|---|
| Seasonal naive | 2024-08 … 2026-07 | 24 months, one shot | No lagged frame. The forecast is last year's route-month, repeated. | 18.29% | 3,113,174.10 | 4,319,584.87 | Yes, this is the reference |
| Gradient-boosted tree | 2024-08 … 2026-07 | 24 months, one shot | Yes. All 22 columns. | 19.18% | 3,465,121.15 | 4,537,239.11 | Yes |
| Seasonal naive | 2023-08 … 2025-07 | 24 months, one shot | No | 16.77% | 3,490,428.18 | 3,836,714.39 | Yes, same design, earlier origin |
| Gradient-boosted tree | 2023-08 … 2025-07 | 24 months, one shot | Yes | 16.82% | 1,969,727.88 | 2,339,818.56 | Yes, same design, earlier origin |
| Lukas's random forest | 2025-08 … 2026-07 | 12 months | No. Five features of its own, including a year-ago passenger lag. | Not computed here. Reported route-month MAE 523.9 against a last-year MAE of 547.4. | Not this grain | Not this grain | No |

**Seasonal naive.** Same idea as the network naive, applied per route before the sum. Both forecast years use the year before T, so month 13 does not read month 1 of the test window. Route-month WAPE is 18.29% on the primary window and 16.77% on the earlier window. Large routes are closer in percent than small ones: at August 2024 the top volume quartile is 13.35% in year 1 and 20.02% in year 2, against 41.63% and 46.69% in the bottom quartile. The forecast has no settings and no trend.

**Gradient-boosted tree.** This is the Deliverable 3 candidate. `HistGradientBoostingRegressor(loss="poisson", max_iter=1000, early_stopping=True, random_state=9)` is fit on realized rows before T and applied to the whole forecast set at once. Poisson loss is a mean on a log link, so route forecasts add up and route, season, and year effects multiply. The 22 columns are known at every horizon, so the model does not recurse and does not use a year-ago passenger lag. On the primary window it is worse than the naive on route-month WAPE in both years and on the network monthly WAPE. On the earlier window it is slightly worse on year-1 route WAPE (15.75% against 15.44%) and better on year-2 route WAPE and on the network. Both fits ran to the 1,000-iteration cap; early stopping did not end them. The settings were not searched, and they were not changed after these scores.

**Lukas's random forest.** Included because it is a merged route-level candidate. It is listed apart because the test is the last 12 months of realized rows only, the features are not the Deliverable 3 frame, and the year-ago lag would fall inside a 24-month window. The notebook also joins the airport file without dropping duplicate codes. Those numbers are his reported MAE and RMSE, transcribed from the last cell of `lukas_rf.ipynb`, and they are not used to choose the forecast. On its own 12-month test the forest is about 4% under the last-year MAE (523.9 against 547.4). That is a different question from a two-year path.

## Excluded

**LSTM.** TensorFlow is not installed, so it would be a new dependency. Sixty-seven training months is a short series for a network LSTM, and on that series the seasonal naive already beat every fitted model. A route-level LSTM would be a new architecture. It was not fit.

**Per-route ARIMA.** Not built. The network ARIMA and SARIMAX already lose to the naive, and a separate ARIMA per route would be a different design from the one shared frame.

**Lukas's forest with a 24-month lag.** Not re-scored. Replacing the year-ago feature with a lag of 24 months would be a new model, fit after seeing that the current one cannot stretch to two years.

**Linear lag baseline.** Named in `docs/plans/2026-11-12-initial-modeling-design.md` and never built.

**Ensembles and a hyperparameter search.** Not run. The network average already lost to the naive, and the tree's settings were fixed before the score so the test window could not choose them.
