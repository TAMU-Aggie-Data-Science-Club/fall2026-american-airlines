# Slide text drafts — Aditi

Draft wording for the final slides (problem approach and data aggregation). Words only, no numbers.

## How we approached the problem
- Defined passenger volume as our prediction target.
- Treated the task as a regression problem due to the continuous variation in passenger demand.
- Selected XGBoost gradient-boosted trees as our primary model because they handle complex relationships and
  high-cardinality categorical features.
- Planned comparisons against Ridge regression and a seasonal naïve baseline.
- Used time-based validation to ensure the model predicts future demand without learning from future data.
- Evaluated different loss functions to determine which produced the most accurate predictions.
- Preserved unusually high passenger volumes because they represent genuine demand on busy routes.

## How we organized the data
- Removed columns that could cause data leakage or provide no predictive value.
- Combined passenger and airport datasets to add airport and country information.
- Organized records by origin airport, destination airport, and departure month.
- Created date-based features to capture seasonality, long-term trends, and recurring travel patterns.
- Developed historical passenger-demand features using previous years of route activity.
- Distinguished international and domestic routes to capture differences in travel demand.
- Applied categorical encoding to airport and country features based on model requirements.
- Handled missing values based on their meaning rather than automatically replacing them with zero.
- Kept all preprocessing and encoding within the training pipeline to prevent data leakage.
