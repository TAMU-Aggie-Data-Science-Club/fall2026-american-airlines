# Team 13 model comparison

Run [the comparison notebook](../../team-13/deliverable4/notebooks/01-model-comparison.ipynb) locally with the approved D3 inputs. The measured side-by-side table and per-window scores are written to `data/interim/team-13-d4/model-comparison.md`. Keep these derived results in approved local storage.

Predict OD_PAX for each supplied directional route and departure month. Reserve the most recent 24 months for D5. From two earlier forecast origins spaced 24 months apart, fit on all earlier observations and predict the following 24 months without refreshing the model. Score all three candidates on identical observed rows. Missing records have unknown demand.

| Candidate | Rationale | Limitation |
|---|---|---|
| Seasonal reference | Repeats the last available same-calendar-month observation for the route. Falls back to route median, then global median. | For sparse markets an old observation or fallback can dominate. |
| Ridge | Regularized additive airport, year and cyclic month effects share information across markets. | Cannot capture route-specific seasonality without interaction features. |
| Gradient boosting | Shallow trees capture interactions among D3 inputs; Huber loss limits the influence of large residuals. | Rare airports have limited support. Future year values lie outside the training range. |

The primary measure is pooled MAE in passengers per observed route-month. RMSE emphasizes large errors, and WAPE shows total absolute error relative to passenger volume. Route identifiers and calendar inputs are assumed known at forecast issuance. One-hot encoding and scaling are fitted within each training window. Predictions are clipped at zero for every candidate.
