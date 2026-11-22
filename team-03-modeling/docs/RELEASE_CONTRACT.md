# Current forecasting and evaluation contract

- Quantity: observed directional O&D passengers by month (or revenue when explicitly selected). Do not claim unconstrained demand or flown seats/load factor.
- Origin: latest input observation month available to the run. True sponsor release/revision timestamps still require verification; monthly index checks alone cannot establish historical availability.
- Horizons: default 12–18. CLI accepts 1–24; alternate horizons require their own evaluated evidence. No validated 36-month model is claimed.
- Fitting/selection: training labels and selection residuals must mature by origin minus configured `label_lag_months` (default one month, an explicit assumption). Earlier calibration origins are fit separately. Without matured selection evidence, weight zero uses the seasonal anchor.
- Baselines/challenger: forecast normalization occurs before excluding unavailable scoring targets. All reported methods use the same nonmissing finite evaluation cases. Duplicate forecast keys fail.
- Missingness: absent rows remain zero-filled historical inputs under an unresolved extract assumption; absent target observations are excluded, not scored as actual zeros. Reported zeros are preserved. Model feature infinities become missing values.
- Duplicates: current ingestion sums duplicate market-months, preserving imported behavior. The separate business EDA removes exact duplicates before summing. Source-owner confirmation is required before comparing those outputs or declaring either rule correct.
- Geography: model domestic definition is US/PR/VI/GU; legacy EDA also includes Canada/AS/MP. These are not equivalent. Passenger-only mode uses unknown geographic attributes and labels segments Unknown; it never invents coordinates or countries.
- Pandemic: forecast training excludes March 2020–December 2021 targets; descriptive scripts use broader masks for different analyses. Treat this as a documented modeling assumption.
- Selection/calibration: grid scores across all outer folds are diagnostic only. Historical weights use earlier matured predictions; historical interval coverage uses earlier matured residuals or is unavailable. Production calibration uses matured residuals available at the final origin.
- Evaluation status: retrospective. Prior results/periods have been inspected, so no untouched final holdout is asserted. Independent reproduction is still required.
- Metrics: WAPE, MAE, RMSE, signed mean error and normalized bias; matched-case counts by horizon/fold. Market-level bootstrap and Wilcoxon are exploratory because airport/time dependence is not fully modeled.
- Network control: forecasts are rescaled to the seasonal-anchor network outlook with user-specified growth. This gives coherence with that chosen outlook, not proof of accurate network-level demand prediction.
- External sources: opt-in. Current revised files plus lag assumptions do not establish historical vintages. Use them for approved descriptive exploration or explicitly retrospective sensitivities until availability is verified.
- Outputs: each backtest/forecast has a unique local directory, status/failure manifest, config, input/source hashes and runtime timestamps. Calibration must explicitly identify a compatible completed backtest. Dirty working-tree runs remain labeled.
- Review: automated gates do not constitute independent scientific approval. No automatic model promotion, final merge, external publication or sponsor contact.
