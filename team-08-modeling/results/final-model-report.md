# Final model report (Deliverable 5)

## Method

1. **Data:** AA passengers and revenue by route and month, AA airport attributes, FAA enplanements, and Census
   population mapped from counties to the nearest US airport.
2. **Features:** passengers 24, 25 and 36 months earlier, seasonality, 2019 same-month baseline, share of metro-pair
   traffic, fare, distance, domestic and hub flags, population, gravity and FAA hub size.
3. **Target:** log change from 24 months earlier, minus the year's volume-weighted average change.
4. **Model:** histogram gradient boosting (200 iterations, depth 7, seed 42), weighted by the square root of earlier
   passengers, with a smearing correction. The forecast is 70% trees and 30% same month two years earlier.
5. **Production run:** the notebook retrains through July 2026 and forecasts August 2026 to July 2028. For 2028, the
   yearly-average feature uses only January–July 2026, so those months carry extra seasonal risk.

## Does the data support a two-year forecast?

Only modestly. Trained through 2023, the model scores **16.73% WAPE against 17.49%** for seasonal naive, but with a
larger negative bias, and a harsher feature check removes the advantage. COVID leaves few normal training years.
Details are in [forecast-evaluation.md](forecast-evaluation.md).

## Business use

- **Supported:** a second opinion beside the seasonal baseline when screening capacity on high-volume or domestic
  routes.
- **Not supported:** pricing, new routes, thin routes, reacting to shocks, or system totals without a bias correction.
- **Risks:** persistent under-forecasting would under-allocate capacity.
- **Monitoring:** compare with naive monthly by segment, and track bias.
- **Before deployment:** rolling-origin backtests, prediction intervals, feature importance and residual diagnostics.

## Open items for the PM

- Whether the aggregate percentages here suit a public repository.
- Whether `OD_PAX` covers AA only or the whole market.
- Approval to add the FAA, Census and gazetteer sources to `DATA.md`.
- Deliverable 5 charts can be shared privately; they are not committed because of the NDA.
