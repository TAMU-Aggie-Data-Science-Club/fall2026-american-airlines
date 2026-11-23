# Model Comparison & Evaluation Report (Deliverable 4/5)
## American Airlines Demand Forecasting Pipeline

### 1. Executive Summary
- **Target Variable:** `OD_PAX` (Origin-Destination Passenger Demand)
- **Validation Methodology:** Rolling-origin expanding-window cross-validation (12-month horizons across three folds) paired with a chronological 24-month out-of-sample holdout test (Training through July 2024; Holdout Test: August 2024 – July 2026).
- **Final Model Architecture:** XGBoost Regressor optimized via Optuna with exogenous macroeconomic/capacity features (FAA Commercial Enplanements, Census CBSA Metro Population & Net Migration), seasonal temporal features, and winter trough calibration.

### 2. Final Performance Metrics (24-Month Holdout Test: Aug 2024 – Jul 2026)
| Metric | XGBoost + Optuna Model | Seasonal Naive Baseline | Interpretation |
| :--- | :--- | :--- | :--- |
| **MAE** | `4,308,062.28` | `2,758,200` | Mean absolute monthly passenger deviation |
| **RMSE** | `5,304,687.76` | — | Root mean squared error penalizing larger variance spikes |
| **MAPE (%)** | `8.28%` | `5.27%` | Out-of-sample percentage error over the 24-month recursive horizon |
| **sMAPE (%)** | `8.00%` | — | Symmetric MAPE confirming consistent tracking across seasonal peaks and troughs |

> **Baseline Context Note:** On this 24-month recursive holdout, the pure Seasonal Naive baseline (`t - 12`) posts a lower error (`5.27%` MAPE / `2,758,200` MAE) than the recursive XGBoost model (`8.28%` MAPE / `4,308,062` MAE) because multi-step recursion compounds error over two full years. However, the XGBoost + Optuna architecture incorporates structural external signals (FAA enplanements and Census metro demographics/migration) and segment-level modeling across hub tiers rather than relying solely on prior-year repetition.

### 3. Rolling-Origin Cross-Validation (12-Month Expanding Windows)
| Validation Fold | Training Cutoff | Test Window (12m) | MAPE (%) | MAE |
| :--- | :--- | :--- | :--- | :--- |
| **Fold 1** | Through `2022-07` | Next 12 months | `13.97%` | `7,235,721` |
| **Fold 2** | Through `2023-07` | Next 12 months | `13.03%` | `7,231,663` |
| **Fold 3** | Through `2024-07` | Next 12 months | `6.49%` | `3,541,551` |
| **CV Summary** | Expanding Origin | Mean across 3 folds | **`11.16% (±3.33%)`** | — |

- Error drops markedly from Fold 1 (`13.97%`) and Fold 2 (`13.03%`) to Fold 3 (`6.49%`) as post-pandemic recovery stabilizes and expanding historical windows give the model stronger seasonal grounding.

### 4. Segment-Level Performance (By Origin FAA Hub Tier — 24m Holdout)
| Origin Hub Segment | Network Share | 24m Holdout MAPE | 24m Holdout MAE |
| :--- | :--- | :--- | :--- |
| **Large Hubs (L)** | `61.4%` | `8.71%` | `2,763,382` |
| **Medium Hubs (M)** | `20.8%` | `9.31%` | `979,164` |
| **International / Non-Hub** | `10.5%` | `8.72%` | `535,368` |
| **Small Hubs (S)** | `7.4%` | `10.64%` | `411,626` |

- **Key Segment Takeaway:** **Large Hubs (L)** carry `61.4%` of network volume with `8.71%` MAPE, while **Small Hubs (S)** show higher relative percentage error (`10.64%` MAPE) on much smaller volume (`7.4%` share), where smaller baselines amplify percentage swings.

### 5. Key Insights & Visual Deliverables
- **Seasonality & Momentum:** Lags, rolling averages (`rolling_mean_12`), cyclical month features, and winter trough adjustments capture recurring annual travel cycles while tempering recursive drift.
- **External Drivers:** FAA commercial enplanement data (522 airports) and Census CBSA metro population/net migration data (387 MSAs) anchor macro market capacity and demographic shifts.
- **Visual Artifacts Generated:**
  - `results/forecast_plot.png`: Actual vs. 24-Month Recursive Forecast curve with prediction intervals and Seasonal Naive baseline comparison.
  - `results/feature_importance.png`: Top feature importances driving the model.
  - `results/diagnostics.png`: Two-panel residual analysis showing error behavior over time and residual distribution.

### 6. American Airlines Business Use Case & Strategic Recommendations
- **Fleet Capacity & Scheduling:** A 24-month view helps network planners anticipate seasonal demand swings (summer peaks vs. winter troughs) and align gauge/capacity planning across Large Hubs (`61.4%` share) versus Medium/Small hubs.
- **Revenue Management & Yield Pricing:** Combining baseline `OD_PAX` trajectories with Census metro growth and FAA enplanement trends supports earlier, more structured planning across fare classes.
- **Model Limitations & Risks:**
  - *Recursive Drift:* Multi-step recursive forecasting across 24 months accumulates error over longer horizons and trails a simple seasonal naive benchmark (`5.27%` vs. `8.28%` MAPE) on this holdout.
  - *Exogenous Lag Dependency:* FAA and Census releases lag real-world conditions, slowing reaction to sudden regional shifts.
  - *Unmodeled Shocks:* Fuel price spikes, schedule changes, weather events, or macro shocks outside historical patterns are not directly observed in advance.

### 7. Two-Year Forecast Horizon & Scope Assessment
Our evaluation tests a **24-month recursive forecast horizon** (August 2024 – July 2026) using out-of-sample holdout testing and rolling-origin cross-validation rather than single-step next-month predictions. This horizon is supported for **directional, seasonally anchored network planning** under normal operating conditions because historical passenger demand (`OD_PAX`) captures annual seasonal cycles alongside rolling/lag features and winter trough adjustments. However, **accuracy is stronger in Year 1 and near-term folds (such as Fold 3 at `6.49%` MAPE) than across a full 24-month recursive stretch (`8.28%` MAPE)** due to recursive error propagation across sequential steps, reliance on static or lagged external baselines (FAA enplanements and Census CBSA metro population/migration figures), a historical window starting in 2019 that includes the COVID-19 disruption and recovery, and sensitivity to unmodeled shocks such as fuel spikes, capacity shifts, or severe weather. Accordingly, the forecast is best used to inform medium-range capacity ranges and scenario bands rather than exact point commitments far into Year 2.