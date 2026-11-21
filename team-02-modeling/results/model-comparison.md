# Team 02 — Model comparison (Deliverable 4)

Code: `notebooks/model_comparison.ipynb` (reads the D3 feature table from `notebooks/feature_set.ipynb`).
**NDA:** rounded numbers only; no rows, no exact counts.

## Validation method (same for every candidate)
- **Time split:** the last 12 months of the data are the test set; everything before is training. Never shuffled,
  because a random split lets the model learn from the future (shown in `eda_akshithreddy-sheri.ipynb`, section 7).
- **Early stopping** for tree models uses the last 6 training months, so the test months are never seen while fitting.
- **Encoders** (one-hot, categorical codes) learn their categories from training months only; unseen airports become
  "missing" / "rare".
- **Shared metrics:** MAE and RMSE (passengers per route-month) and **WAPE** (total absolute error / total passengers).
  WAPE is the headline metric because it is comparable across big and small routes and reads as "% of passengers missed".

## Results (last 12 months held out)

| Candidate | Owner | MAE | RMSE | WAPE | vs baseline |
|---|---|---|---|---|---|
| Same month last year (baseline) | Akshith | | | | — |
| Ridge on `log1p` | Sahith | 2,506 | 8,493 | 82.32% | 357.8% higher WAPE |
| XGBoost, `log1p` + squared error | Aidan | | | | |
| XGBoost, Poisson | Aidan | | | | |

*(fill from section 6 of the notebook, rounded: MAE/RMSE to whole passengers, WAPE to 2 decimals)*

**Ridge result — Sahith (Nov 21, 2026):** the supplied pipeline (`alpha=1.0`, training-only encoding and scaling)
ran successfully, but its WAPE was 4.58 times the same-month-last-year baseline's 17.98%. Ridge did not beat the
baseline on this holdout. This is the fixed-alpha comparison, not a tuned Ridge result.

**Segments** (section 7): *(one or two sentences: domestic vs international, small vs large routes, new routes)*

## Trade-offs

| Candidate | Strengths | Limitations | Included because |
|---|---|---|---|
| Seasonal naive (last year) | no training, impossible to overfit, easy to explain | no answer for new routes; copies last year's shocks | every model must beat it |
| Ridge (`log1p`) | fast, coefficients are explainable, stable | needs hand-built interactions; one-hot of hundreds of airports; can't bend | simple, explainable comparison (Sahith) |
| XGBoost `log1p` | handles high-cardinality airports natively, finds month x route interactions | `log1p` + squared error predicts a median-like value, so totals can come out low | main candidate (Aidan) |
| XGBoost Poisson | made for counts; predicts the mean, so route totals add up | more sensitive to very large routes | settles the loss debate (Aidan) |

**Loss decision:** *(log1p or Poisson — the one with lower WAPE; one sentence why — Aidan)*

## Considered but not run — Pearl
*(Pearl's paragraph: why not Random Forest, ARIMA/SARIMA, LSTM. Suggested points: Random Forest is the same tree
family as XGBoost but usually weaker and slower on large tables; ARIMA fits one route at a time, so thousands of
separate models with no sharing across routes, and it cannot use airport information; LSTM needs long sequences per
route and much more data and tuning than our monthly history gives, with little explainability for AA.)*
