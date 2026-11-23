# Model comparison (Deliverable 4)

Every candidate uses the same test year (2025), rows (route-months with a value 24 months earlier) and metrics.
WAPE is total absolute error divided by total actual passengers; lower is better.

| Candidate, trained on 2022–2024 | WAPE (%) | Bias (%) |
|---|---|---|
| Seasonal naive (same month, 24 months earlier; no training) | 17.49 | −1.29 |
| Ridge regression | 17.44 | +1.67 |
| Gradient-boosted trees, AA data only | 16.08 | −0.63 |
| Gradient-boosted trees, AA + FAA + Census | 16.05 | −0.08 |
| Ensemble: 70% trees (AA + FAA + Census) + 30% naive | 15.69 | −0.44 |

These scores rank candidates on equal terms, but **they are not two-year-ahead accuracy**, because training uses
2024 outcomes. See [forecast-evaluation.md](forecast-evaluation.md).

## Trade-offs

- **Seasonal naive:** transparent and low bias, but cannot adapt to a route's changing share or growth.
- **Ridge:** readable, but no meaningful gain over naive.
- **Trees:** capture non-linear effects and accept missing values; cannot anticipate a new demand shock.
- **External data:** improved the trees by only 0.03 points, so it is not a main driver of accuracy.
- **Ensemble:** lowest error; the naive share limits drift. The weight is a fixed constant and the notebook shows no
  validation-based selection, so it may be tuned to the test year.

## Considered, not benchmarked

- **ARIMA:** one series per route, with a short history broken by COVID and no sharing across routes.
- **LSTM:** short history and a regime change; harder to explain to planners.
- **XGBoost, Random Forest:** other tree ensembles. Scikit-learn's histogram gradient boosting was used instead. The
  Deliverable 3 EDA named XGBoost, which is also gradient-boosted trees.
