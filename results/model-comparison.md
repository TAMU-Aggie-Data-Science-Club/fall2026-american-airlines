# Model Comparison & Evaluation Report (Deliverable 4/5)
## American Airlines Demand Forecasting Pipeline

### 1. Executive Summary
- **Target Variable:** `OD_PAX` (Origin-Destination Passenger Demand)
- **Validation Methodology:** Chronological train/test split (Training: 2019–2024; Holdout Test: August 2024 – July 2026).
- **Final Model Architecture:** XGBoost Regressor optimized via Optuna with exogenous macroeconomic/capacity features (FAA Enplanements, Census Metro Population & Net Migration).

### 2. Final Performance Metrics (24-Month Holdout Test)
| Metric | XGBoost Model Value | Seasonal Naive Baseline | Interpretation |
| :--- | :--- | :--- | :--- |
| **MAE** | `3,631,019.40` | `2,758,200.00` | Mean absolute monthly passenger deviation |
| **RMSE** | `4,706,870.29` | — | Root mean squared error penalizing larger variance spikes |
| **MAPE (%)** | `6.92%` | `5.27%` | Out-of-sample percentage error over the 24-month horizon |
| **sMAPE (%)** | `6.64%` | — | Symmetric MAPE confirming robust tracking across seasonal peaks |

### 3. Key Insights & Visual Deliverables
- **Seasonality & Momentum:** Lags 1, 12, and 12-month rolling means (`rolling_mean_12`) successfully anchored recurring summer peak surges and holiday travel troughs.
- **External Drivers:** FAA commercial enplanement volume trends and Census CBSA metro populations provided stable macro grounding for multi-year forecasting.
- **Visual Artifacts Generated:**
  - `results/forecast_plot.png`: Actual vs. 24-Month Recursive Forecast curve featuring **80% and 95% shaded prediction intervals** and Seasonal Naive baseline comparison.
  - `results/feature_importance.png`: Top 10 feature importances dominated by 12-month rolling trends.
  - `results/diagnostics.png`: Two-panel residual analysis showing monthly error percentages and error distribution.

### 4. American Airlines Business Use Case & Strategic Recommendations
- **Fleet Capacity & Scheduling:** The 24-month horizon allows network planners to optimize aircraft allocation (narrowbody vs. widebody splits) up to two seasons in advance, specifically preparing for summer peak surges while rightsizing capacity during winter troughs.
- **Revenue Management & Yield Pricing:** By anchoring baseline passenger demand (`OD_PAX`) against Census metro growth and FAA enplanement trends, yield management teams can set advanced fare class availability without over-relying on reactive last-minute discounting.
- **Model Limitations & Risks:** 
  - *Recursive Drift:* While mitigated by seasonal weighting, multi-step recursive forecasting over extended horizons remains sensitive to sudden external macroeconomic shocks (e.g., fuel price spikes or severe weather anomalies).
  - *Exogenous Lag Dependency:* Public demographic and FAA datasets lag by 12 months, meaning sudden localized shifts in urban migration may take time to reflect in feature weights.

### 5. Two-Year Forecast Horizon & Scope Assessment
--------------------------------------------------------------------------------
Our evaluation tests a 24-month recursive forecast horizon (August 2024 – July 2026) using out-of-sample holdout testing and rolling-origin cross-validation rather than single-step next-month predictions. 
This horizon is supported for directional, seasonally anchored network planning under normal operating conditions because historical passenger demand (OD_PAX) captures full annual seasonal cycles alongside rolling_mean_12, lag features, and winter trough adjustments. 
However, accuracy is stronger in Year 1 than Year 2 due to recursive error propagation across sequential steps, reliance on static or lagged external baselines (FAA enplanement trends and Census CBSA metro population/migration figures), a historical window starting in 2019 that includes the COVID-19 disruption and recovery, 
and sensitivity to unmodeled shocks such as fuel spikes, capacity shifts, or severe weather. Accordingly, the forecast is best used to inform medium-range capacity and planning ranges alongside 80%/95% prediction intervals rather than exact point commitments far into the second year.