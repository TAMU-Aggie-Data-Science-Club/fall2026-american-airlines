# Forecast evaluation — Team 10 (Deliverable 5)

**Notebook:** `notebooks/04_final_model_saideep.ipynb` · **Generated tables:** `test_scores.md`

## Can this data support a two-year-ahead demand forecast?

**No.** We state this up front rather than present a number that only looks like one.

| requirement for a two-year demand forecast | this dataset |
|---|---|
| A **demand** target (passengers or bookings) | Target is a **fare** (`total_price`) |
| Dated observations to order past → future | **No dates at all.** `month` is month-of-year with no year. |
| ≥ 2–3 years of history for a 24-month hold-out | Not applicable, since there is no time axis |

Because nothing orders the rows in time, there is no train-on-past / test-on-future boundary at
*any* horizon. A next-step forecast cannot be evaluated either. The most honest result the data
supports is a **cross-sectional pricing model**: the fare AA quotes for an itinerary at quote time.
Its only "horizon" is the booking window (0–119 days before departure). That model is what we
evaluate below.

## Test design

- **Hold-out:** 40 of 199 cleaned rows (20%), random, stratified by cabin, `random_state=10`.
  Random because there is no time column.
- **Leakage controls:** all tuning (alpha, interactions) uses only the 159 training rows via
  repeated 5-fold CV. The scaler is fit inside each fold. No feature is derived from the target.
  The competitor quote is assumed to be known at quote time (open PM question).
- **Baselines on the same rows:** cabin median, match-the-competitor rule, and XGBoost without the
  competitor price.

## Results on the held-out test

| model | MAE ($) | RMSE ($) | R² | MAPE (%) | worst miss ($) |
|---|---|---|---|---|---|
| Final Ridge (log price) | 21.97 | 37.82 | 0.98 | 6.37 | 170.68 |
| Match-the-competitor rule | **16.75** | **25.59** | **0.99** | **5.44** | **97.00** |
| Cabin median | 159.85 | 221.68 | 0.37 | 77.73 | 888.50 |
| XGBoost, no competitor (fallback) | 76.83 | 113.64 | 0.83 | 23.91 | 423.16 |

**On these 40 rows, the simple competitor rule beats the final Ridge model.** In 25-fold CV the
two were tied on MAE ($20.1 vs $20.9) and Ridge was better on RMSE, so the test result is within
the range of fold-to-fold noise (fold std ≈ $4–6). Neither source says Ridge is reliably more
accurate than the rule. Our conclusion: in this data, the competitor quote carries almost all the
predictable signal. The other features change the fare by a few percent at most.

## Uncertainty: prediction-interval coverage

Split-conformal intervals from out-of-fold log residuals on the training rows:

| nominal | observed coverage (40 test rows) | interval as a price multiplier |
|---|---|---|
| 80% | 82.5% (33 / 40) | × 0.899 – 1.113 |
| 90% | 85.0% (34 / 40) | × 0.884 – 1.126 |

The 80% band is well calibrated. The 90% band under-covers by 2 rows out of 40, which is within
sampling noise at this size but leans narrow. In practice, use the 90% band as roughly an
85% band, or widen it until more data confirms it.

## Limitations

- **Synthetic data, 199 rows.** All numbers are for building and testing the pipeline, not
  conclusions about real AA pricing.
- **No time axis:** no trend, no drift detection, and no out-of-time validation. A random split
  is optimistic compared with a real forward test.
- **Test-set hygiene:** the 40 test rows were scored once with the final configuration. After
  seeing that result we checked one alternative (Ridge on the price/competitor ratio), but judged
  it **only on training CV**, where it was not better, so the final model did not change.
- **Small segments:** only 4 Business/First rows in the test set. Segment errors in the report
  use out-of-fold predictions on all 199 rows instead.
