# Baseline decision — Team 10 (Deliverable 4)

## Decision

**Baseline model carried into Deliverable 5: Ridge regression on `log1p(total_price)`, all 16 D3
features, standardised inside a pipeline.**

**Fallback (competitor feed unavailable): XGBoost on log price without `log_competitor_price`.**

## Why Ridge (in the D2 order: shape, loss, stakeholder, cost, and only then score)

1. **Data shape.** This is a cross-section of quotes with no time index, so ARIMA and LSTM are out.
   The dominant driver, competitor price, is close to proportional to AA's price. A linear model on
   log scale represents a proportional relationship exactly. Trees can only approximate it with
   steps, and that showed up as 2× worse RMSE.
2. **Loss.** Squared error on log price is approximately squared *relative* error. A $20 miss on a
   $90 fare and a $200 miss on a $900 fare are treated alike, which matches how fare errors matter.
3. **Stakeholder need.** A revenue-management analyst can read the coefficients directly ("+1 SD of
   distance ≈ +6.6% price, holding competitor price fixed"). A pricing benchmark has to be explainable.
4. **Cost.** It fits in milliseconds, has one hyperparameter (`alpha`), and is easy to monitor.
5. **Score.** Best CV MAE ($20.1) and clearly best RMSE ($33 vs $40 for the rule and $72 for
   XGBoost), with the smallest fold-to-fold spread.

## What we are *not* claiming

- Ridge does **not** clearly beat the match-the-competitor rule on MAE (11 / 25 folds). The honest
  reading: on this data, almost all predictable variation in AA's fare is already in the competitor
  quote. Cabin, distance and booking window add a few dollars on top.
- The D3 hypothesis was gradient-boosted trees. **D4 overturned it** for the full feature set.
  Trees win only once the competitor price is removed.

## Carry-forward to Deliverable 5

- Tune `alpha` and test pairwise interaction terms on training rows only.
- Test once on a held-out 20% split against the same baselines (cabin median, competitor rule,
  XGBoost fallback).
- Add prediction intervals, diagnostics, segment errors, and the business interpretation.
