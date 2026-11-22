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
| Same month last year (baseline) | Akshith | 547 | 1,366 | 17.98% | — |
| Ridge on `log1p` | Sahith | 2,506 | 8,493 | 82.32% | 357.8% higher WAPE |
| XGBoost, `log1p` + squared error | Aidan | 588 | 1,502 | 19.3% | about 7% higher WAPE |
| XGBoost, Poisson | Aidan | 575 | 1,425 | 18.9% | about 5% higher WAPE |

*(rounded; test = Aug 2025 – Jul 2026, train = Jan 2019 – Jul 2025)*

**Result:** the same-month-last-year baseline had the lowest error of every candidate. Poisson beat `log1p`
among the XGBoost runs, but neither XGBoost model beat the baseline. The gap is small (about one WAPE point for Poisson).

**What the trees rely on** (Poisson, share of gain): same month last year (about half), same month two years
ago (about a quarter), then whether the route existed last year (under 10%). The model is mostly re-learning
the seasonal naive, which explains why it lands close to the baseline but not below it.

**Ridge result — Sahith (Nov 21, 2026):** the supplied pipeline (`alpha=1.0`, training-only encoding and scaling)
ran successfully, but its WAPE was 4.58 times the same-month-last-year baseline's 17.98%. Ridge did not beat the
baseline on this holdout. This is the fixed-alpha comparison, not a tuned Ridge result.

**Segments (12-month test, WAPE):** domestic routes (about 77% of passengers) are much easier than international
(about 23%): baseline about 16% vs 24%, XGBoost Poisson about 17% vs 25%. Error grows as routes get smaller: large
routes (about 89% of passengers) about 17–18%, medium (about 9%) about 27%, small (about 2%) about 36–40%. Ridge is
far behind in every segment. Because large routes carry most passengers, the headline WAPE mostly reflects them.

## Tuning and learning curves
A hyperparameter search (learning rate, depth, `min_child_weight`, `reg_lambda`, and Poisson's `max_delta_step`,
scored on the last 12 training months, never the test year) and two learning curves (error vs trees, error vs
years of history) are built in sections 9–10 of the notebook and tested. The search did not finish before the
deadline, so no tuned result is reported. Both original XGBoost fits used the full 2,000-tree budget, so they had
not finished learning; finishing this search is the first next step.

## Trade-offs

| Candidate | Strengths | Limitations | Included because |
|---|---|---|---|
| Seasonal naive (last year) | no training, impossible to overfit, easy to explain | no answer for new routes; copies last year's shocks | every model must beat it |
| Ridge (`log1p`) | fast, coefficients are explainable, stable | needs hand-built interactions; one-hot of hundreds of airports; can't bend | simple, explainable comparison (Sahith) |
| XGBoost `log1p` | handles high-cardinality airports natively, finds month x route interactions | `log1p` + squared error predicts a median-like value, so totals can come out low | main candidate (Aidan) |
| XGBoost Poisson | made for counts; predicts the mean, so route totals add up | more sensitive to very large routes | settles the loss debate (Aidan) |

**Loss decision:** Poisson. It had lower MAE, RMSE and WAPE than `log1p` (Aidan), and it predicts the mean, so
route totals add up.

## Considered but not run — Pearl
We went with gradient boosted trees (XGBoost) instead of Random Forest, ARIMA, or LSTM because our EDA showed
strong non-linear seasonality and route-level patterns across lots of origin-destination pairs. Fitting ARIMA per
route gets messy when some routes have intermittent or missing monthly records, LSTMs are overkill and heavier to
train on tabular calendar/route data, and boosted trees handle those interactions and splits better than Random
Forest.
