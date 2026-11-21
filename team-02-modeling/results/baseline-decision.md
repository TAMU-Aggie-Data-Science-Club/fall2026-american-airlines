# Team 02 — Baseline decision (Deliverable 4)

**Model carried into Deliverable 5:** XGBoost with the *(Poisson / log1p)* loss, trained on the D3 feature set.

## Why
1. **Accuracy:** *(WAPE on the 12-month test, and how much better than "same month last year" — rounded)*
2. **Fits the data shape:** hundreds of airports and thousands of routes are handled as native categoricals; month x
   route interactions (domestic vs international seasonality, `eda_categoricals_akshithreddy-sheri.ipynb`) are learned
   automatically.
3. **Works for the two-year horizon:** it runs on "same month two years ago" plus calendar and airport features, all
   known at forecast time, and supports prediction intervals through quantile loss.
4. **Explainable enough:** feature importance shows what drives the forecast.

**If Ridge scored best:** *(say so honestly, give the gap, and explain the choice — e.g. we still carry XGBoost
because ___, or we switch to Ridge.)*

## What we gave up
- Trees cannot extrapolate a trend past what they have seen — long-run growth is a limitation (see D5 report).
- New routes with no history get weaker forecasts.

Decided by: Akshith Reddy Sheri (lead) with Aidan Chance and Sahith Yamani, *(date)*.
