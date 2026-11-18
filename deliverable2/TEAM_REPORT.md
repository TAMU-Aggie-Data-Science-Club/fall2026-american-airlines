# Team 13 · Deliverable 2

Rahul’s measured contribution, pending team review. Member branch `team-13-Rahul`, integration branch `team-13`. The notebook reproduces the results from the public EaseMyTrip CSV using seed 42. Fit times are machine-dependent.

## EDA decisions

| Observation | Modeling decision |
|---|---|
| Business fares average ₹52,540 and economy ₹6,572. | Retain cabin and inspect percentage errors as well as rupee errors. |
| The cabin indicator correlates 0.938 with fare. | Include cabin explicitly in the preprocessing coverage check. |
| The booking curve steepens close to departure, particularly in economy relative to its usual price. | Allow a nonlinear lead-time effect and interactions with cabin. |
| Stops and time bands have an order. Cities and airlines are labels. | Encode stops and time bands in their observed order and one-hot encode nominal fields. Clock time wraps at midnight, a limitation of the ordinal encoding. |

## Leakage audit

| Configuration | R² | MAE (INR) | MAPE (%) |
|---|---:|---:|---:|
| D1 baseline (3 leaks) | 0.9782 | 1,910.67 | 14.43 |
| No counter | 0.9739 | 2,100.27 | 15.00 |
| Deduplicated | 0.9741 | 2,106.64 | 15.23 |
| XGBoost | 0.9767 | 1,975.72 | 14.31 |
| XGBoost, five folds (mean ± sample SD) | 0.97628 ± 0.00031 | 1,990.11 ± 9.35 | 14.27 ± 0.09 |

The largest drop followed removal of the export counter. For this practice population, the leakage-safe XGBoost estimate is ₹1,990 MAE across five shuffled folds. The folds do not establish accuracy on future departures.

## Regularization

Ridge and Lasso both favored the smallest tested alpha, 0.1, on the inner validation split. Lasso had marginally lower holdout MAE, while OLS and both penalized fits rounded to the same R² of 0.9092. There are many observations for a small feature space, leaving little variance for shrinkage to remove. At alpha 0.1 Lasso zeroed two features, and at alpha 100 it zeroed eleven, including the Indigo indicator and departure time. A zero coefficient for one dummy can reflect redundancy among encoded categories. Ridge is a sensible explanatory baseline, with the trade-off that one lead-time slope misses the booking curve.

## Training objectives and intervals

| Objective | MAE (INR) | RMSE (INR) | R² | MAPE (%) |
|---|---:|---:|---:|---:|
| XGBoost | 1,975.72 | 3,461.01 | 0.9767 | 14.31 |
| Absolute | 2,061.77 | 4,074.85 | 0.9677 | 13.42 |
| Huber | 2,629.71 | 4,593.41 | 0.9589 | 17.89 |

The squared-error XGBoost fit scored lower on RMSE than the absolute-error fit, while absolute error reduced percentage error. Across all three fits, the lowest MAE came from XGBoost and the lowest RMSE from XGBoost. Huber used a different boosting implementation, so that comparison also includes training differences. For a typical-fare benchmark I would optimize absolute error and report MAE alongside cabin-level percentage errors. The 10th–90th percentile band covered 81.71% of held-out fares, slightly wider in aggregate than the 80% target. If quoting too low is assumed to cost more because a traveler budgets below the eventual fare, a 0.75 quantile would be a candidate to validate with stakeholders.

## Candidate comparison

| Model | MAE (INR) | R² | Fit time (s) | Interpretable? | Use |
|---|---:|---:|---:|---|---|
| Dummy | 19,779.74 | -0.0000 | 0.36 | Yes | Mean-fare reference |
| Ridge | 4,518.26 | 0.9092 | 0.27 | Yes | Signed effects, stable baseline |
| Lasso | 4,518.21 | 0.9092 | 0.30 | Yes | Sparse explanation |
| Random Forest | 1,284.76 | 0.9852 | 29.40 | Limited | Batch, nonlinear cabin interactions |
| XGBoost | 1,975.72 | 0.9767 | 2.41 | Limited | Fast boosted-tree refits |
| ARIMA | 1,626.03 (curve) | — | 0.05 | Partly | Aggregate booking curve |

The forest reduced ticket MAE by ₹691 relative to XGBoost. It took 29.4 seconds to fit versus 2.4 seconds for XGBoost on this machine. Against the mean-fare predictor, the MAE reductions were ₹18,495 for the forest and ₹17,804 for XGBoost. The forest is a reasonable first handoff for batch work because a teammate can start with depth and leaf size. XGBoost is attractive when frequent retraining matters.

On the last nine lead times, ARIMA(1,1,0) had MAE ₹1,626.03 and the last-value forecast had MAE ₹1,647.26. A ₹21.23 advantage on nine points is weak evidence for repeatable improvement. ADF failed to reject a unit root on the training levels and on their first differences, so one difference is a working assumption with limited support. The series summarizes average fare as departure approaches. Aggregation removes the carrier, route, cabin, and stops needed to price an individual itinerary. Use it as a descriptive booking-curve benchmark only.

## Recommendation for discussion

Random Forest is the candidate for a batch fare benchmark. Its splits can represent the sharp rise near departure, its trees can learn cabin-specific responses without hand-built interactions, and bagging provides a straightforward training recipe for teammates to repeat. The main cost is slower fitting and a larger collection of trees to maintain. A chronological, market-specific evaluation would be necessary before operational use.

For AA, the method could support a retrospective fare-benchmarking exercise. An operational decision needs AA-specific dated quotes, inventory, and an agreed tolerance for pricing errors. The observed ₹1,285 forest MAE describes the Indian practice sample only. The forest result comes from the comparison holdout, so use an independent evaluation before making a deployment claim.

## Review and sign-off

Rahul’s observed setup issue was the collision between the shared trunk name and a slash-prefixed member branch. Agreeing on publishable member names and checking write access before modeling would simplify the next deliverable. Teammates still need to record the review ring and their own findings, agree on a team recommendation, and complete the team submission.
