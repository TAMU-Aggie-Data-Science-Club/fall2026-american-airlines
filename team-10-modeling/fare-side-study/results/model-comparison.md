# Model comparison — Team 10 (Deliverable 4)

**Notebook:** `notebooks/03_model_selection_saideep.ipynb` · **Generated scores:** `cv_scores.md`
**Data:** D3 feature set (`features.py`) on 199 cleaned rows of `aa_fares_compact_SYNTHETIC.csv`.

## Validation method and metrics

- **Repeated 5-fold CV, 5 repeats, `random_state=10`:** 25 fits per model, the same 25 folds for
  every model. A single 80/20 split would leave only 40 test rows, too few to rank models on.
- **Random folds:** the file has no date column, so a chronological split is impossible.
- **Target:** every learned model trains on `log1p(price)`. All metrics are in **dollars**
  after back-transforming.
- **Metrics:** MAE (headline), RMSE, R², MAPE, and fit time.

## Candidates and why each was included or excluded

| candidate | included? | reason |
|---|---|---|
| Cabin median | yes | Floor: charge the cabin's typical price. |
| Match-the-competitor rule | yes | The obvious business heuristic, and a very strong one given the 0.99 correlation. |
| Ridge on log price | yes | Readable and regularised. D3 made the effects roughly additive on the log scale. |
| Ridge on log(price / competitor) | yes | Tests whether the other features add anything *on top of* the competitor quote. |
| Random Forest | yes | Bagged trees: non-linear, little tuning. |
| XGBoost | yes | The family D3 pointed to (the 14-day step, interactions). |
| ARIMA | **no** | Needs an ordered time series. There is no date column. |
| LSTM | **no** | No sequence to learn from, only 199 rows, and TensorFlow is not a team dependency. |

## Results (mean ± std over 25 folds)

| model | features | MAE ($) | RMSE ($) | R² | MAPE (%) | fit s/fold |
|---|---|---|---|---|---|---|
| Cabin median | all | 148.4 ± 24.5 | 234.3 ± 78.1 | 0.046 ± 0.437 | 63.4 ± 11.5 | 0.001 |
| Match-the-competitor rule | all | 20.9 ± 5.8 | 39.5 ± 22.7 | 0.976 ± 0.011 | 6.3 ± 0.8 | 0.001 |
| Ridge on log(price / competitor) | all | 20.6 ± 5.2 | 37.8 ± 19.0 | 0.977 ± 0.009 | 6.3 ± 0.9 | 0.004 |
| **Ridge (log price)** | all | **20.1 ± 4.1** | **33.3 ± 10.2** | **0.980 ± 0.010** | 6.4 ± 0.8 | 0.005 |
| Random Forest | all | 32.9 ± 16.8 | 81.5 ± 73.3 | 0.903 ± 0.101 | 8.4 ± 1.2 | 0.271 |
| XGBoost | all | 30.6 ± 15.1 | 72.3 ± 65.0 | 0.926 ± 0.076 | 8.0 ± 1.1 | 0.443 |
| Ridge (log price) | no competitor | 62.7 ± 17.8 | 124.4 ± 68.9 | 0.773 ± 0.086 | 17.9 ± 1.9 | 0.006 |
| Random Forest | no competitor | 69.6 ± 23.9 | 134.3 ± 80.6 | 0.743 ± 0.115 | 20.2 ± 2.4 | 0.283 |
| **XGBoost** | no competitor | **52.9 ± 21.5** | **110.0 ± 79.4** | **0.833 ± 0.114** | **14.4 ± 1.8** | 0.440 |

Paired, fold-by-fold (same folds):

| A | B | mean MAE diff (A−B) | folds where A is better |
|---|---|---|---|
| Ridge (log price) | Match-the-competitor rule | −0.76 | 11 / 25 |
| Ridge on log(price / competitor) | Match-the-competitor rule | −0.28 | 15 / 25 |
| XGBoost | Ridge (log price) | +10.46 | 4 / 25 |
| Random Forest | Ridge (log price) | +12.81 | 5 / 25 |
| XGBoost [no competitor] | Ridge [no competitor] | −9.79 | 23 / 25 |

Figure: `visuals/d4_cv_mae_boxplot.png`.

## Trade-offs

1. **The competitor quote is almost the whole story.** The one-line rule (competitor × median
   ratio) is within $1 of the best model's MAE. Ridge and the rule are a statistical tie on MAE:
   Ridge is better in only 11 of 25 folds. Ridge's real edge is **RMSE** ($33 vs $40) and a
   smaller fold-to-fold spread, so it makes fewer large misses.
2. **Trees lose when the competitor price is in.** Price is close to proportional to the
   competitor quote. A tree approximates that straight line with steps and cannot extrapolate
   past the training range. That costs it on the expensive Business/First fares (RMSE 72–82 vs 33,
   with large fold variance).
3. **Without the competitor price, the ranking flips.** XGBoost beats Ridge in 23 of 25 folds
   (MAE $53 vs $63). With the dominant linear signal gone, what is left (the 14-day step, cabin ×
   distance) is non-linear, which is exactly where boosted trees earn their keep.
4. **Cost:** Ridge fits in milliseconds and its coefficients can be read. XGBoost is ~100× slower
   to fit (still under a second here), has more hyperparameters, and needs SHAP-style tools to explain.
5. **Small-data caveat:** with 199 rows, differences under ~$2 MAE are inside the noise (fold std
   is $4–6 for the linear models).

## Recommendation

Carry **Ridge on log price (all features)** forward as the baseline, and keep **XGBoost without
the competitor price** as the fallback for when the competitor feed is unavailable. Rationale:
`baseline-decision.md`.
