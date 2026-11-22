# Initial Modeling Notebook Design

## Goal

Create a beginner-friendly notebook under `pm-modeling/` that uses the local
American Airlines passenger-demand data to explore and forecast `OD_PAX` with
simple ARIMA/SARIMAX, LSTM, XGBoost, Random Forest, and ensemble baselines.

## Design

- Read `data/AA_PAX_FORECASTING.csv` and aggregate `OD_PAX` to a single monthly
  demand series. The source contains monthly observations, so the requested
  “day” forecasts are represented as one-step-ahead monthly forecasts rather
  than invented daily observations.
- Sort the monthly series chronologically and reserve the most recent 24 months
  as the test set. Use the earlier observations for training.
- Explain and show basic data shape, missing values, descriptive statistics,
  trend/seasonality, and a train/test plot.
- Fit intentionally small, surface-level baselines:
  - seasonal-naive and linear lag features as reference models;
  - ARIMA and SARIMAX with a simple 12-month seasonal period;
  - Random Forest and XGBoost using lag and calendar features;
  - a small one-layer LSTM using scaled lag windows.
- Evaluate all models with MAE and RMSE.
- Show two forecast modes:
  1. recursive one-step forecasts, where each next period is predicted and then
     fed into the next prediction;
  2. a direct two-year future forecast from the end of the available data,
     using each model's native/simple recursive mechanism to produce 24 periods.
- Build a simple ensemble by averaging available model predictions for each
  period and evaluate/plot it alongside the individual models.
- Keep markdown comments throughout, explicitly calling out that this is a
  teaching baseline rather than a production forecast.

## Scope and limitations

- Do not add a new data source or commit the confidential CSVs.
- Do not use reinforcement learning; it is unnecessary for this introductory
  forecasting exercise.
- Do not tune hyperparameters extensively or model every origin-destination
  route separately.
- Deep-learning cells should fail clearly with an install note if TensorFlow
  is unavailable, rather than silently substituting another model.

## Validation

- Restart the notebook kernel and run all cells from top to bottom.
- Confirm the notebook discovers the CSV via a path relative to the repository.
- Confirm the latest 24 monthly periods are excluded from training.
- Confirm every model and the ensemble produces a 24-period forecast table and
  evaluation metrics when its dependency is available.
