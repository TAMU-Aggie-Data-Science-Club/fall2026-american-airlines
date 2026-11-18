# Baseline decision

The model carried into Deliverable 5 is the route-level seasonal naive. It is also the comparison baseline. The gradient-boosted tree was scored once on the Deliverable 3 frame and rejected.

## The bar

The bar was fixed before either score was computed. The tree is carried only if its error is lower than the naive on all six checks:

- route-month WAPE in year 1 (horizons 1–12)
- route-month WAPE in year 2 (horizons 13–24)
- network monthly WAPE over the 24 months

at both August 2023 and August 2024. Otherwise the naive is carried and the tree is reported as scored and rejected.

| Origin | Check | Naive | Tree | Tree lower |
|---|---|---:|---:|---|
| 2023-08 | Route-month WAPE, year 1 | 15.44% | 15.75% | No |
| 2023-08 | Route-month WAPE, year 2 | 18.11% | 17.90% | Yes |
| 2023-08 | Network monthly WAPE | 6.25% | 3.53% | Yes |
| 2024-08 | Route-month WAPE, year 1 | 15.10% | 16.15% | No |
| 2024-08 | Route-month WAPE, year 2 | 21.69% | 22.42% | No |
| 2024-08 | Network monthly WAPE | 5.78% | 6.44% | No |

The tree wins two checks, both on the overlapping earlier window, and loses every check on the primary window. It is not carried.

Both tree fits used `HistGradientBoostingRegressor(loss="poisson", max_iter=1000, early_stopping=True, random_state=9)` and otherwise the scikit-learn defaults. Both ran to the 1,000-iteration cap. Early stopping did not end training. No setting was changed after the scores were seen.

## What is carried

The carried forecast assigns each route-month the passengers that route carried in the same calendar month of the year before the origin, and it uses that same year for both forecast years. There is no fitted model and no hyperparameter. The August 2024 network sum matches PR #77 to the passenger (MAE 3,113,174.10, RMSE 4,319,584.87).

The comparison baseline is this same forecast. Deliverable 5 does not train a second model to sit underneath it.

## Why this one

These are the reasons that are not "it had the lowest score":

1. It stays leakage-safe for all 24 months. Both years copy the year before the origin, so the second year never reads a month inside the test window, and nothing is fed back as a lag.
2. Route forecasts add up to the network seasonal naive that PR #77 and the PM notebook already use. The notebook checks that sum. A route plan and the network plan cannot disagree about the total.
3. It has no settings. The test window cannot be used to pick a learning rate, a tree depth, or a feature list, because there is nothing to pick.

## What was ruled out

The gradient-boosted tree was the Deliverable 3 candidate and the one route model fit on that frame. Poisson loss, a direct 24-month prediction, and columns that are known at every horizon were the reasons to score it. It did not clear the bar above. It stays in `model-comparison.md` as scored and rejected. Because it is not carried, Deliverable 5 does not ship quantile intervals or a feature-importance chart for it.

Also ruled out, and not fit here: an LSTM, a per-route ARIMA, a re-score of Lukas's forest with a lag of 24 months, the linear lag baseline from the November 12 modeling plan, an ensemble, and a hyperparameter search. The reasons are in `model-comparison.md`. Lukas's published 12-month forest is a different test and is not a candidate for this forecast.
