# Final model report

## Training

The release model is a seasonal anchor, a planner-set network growth rate, and a pooled histogram gradient-boosted model of relative market movement. Training uses observed passenger labels through the origin minus a one-month lag. Targets from March 2020 through December 2021 are excluded. External macro, demographic, and exchange-rate features are off. Settings are in `configs/release.json`.

The scored weight on the learned mix is chosen only from forecasts whose actuals had already matured. On this run that rule selected zero at every backtest origin. The forward forecast, after more of those outcomes had matured, uses a weight of 0.25 and then rescales markets so each month's network total matches the seasonal outlook. Network growth is zero.

## Horizon

The evaluated range is 12 to 24 months, which is the two-year planning window the extract can support. Months beyond July 2026 are not scored because the extract ends there. No 36-month model is claimed. The forward network outlook is July 2027 through July 2028.

## Business reading

American Airlines can use the seasonal anchor as the capacity baseline a year to two years ahead, and can replace the flat network outlook with a corporate growth rate without refitting. The learned mix is a small reallocation around that total. It did not beat the anchor under the matured-weight rule. Same-month-last-year is the stronger network benchmark in this backtest, 6.8% WAPE against 7.0%.

The intervals are too narrow on the later origins, so they are not a sufficient staffing or fleet band on their own. The forecast is observed origin-destination passengers, not seats, load factor, or unconstrained demand. Hub totals overlap and should not be added across hubs.

## What this does not establish

There is no untouched holdout. Duplicate keys were absent in this extract, and missing historical inputs are still zero-filled by assumption. Canada is international in the scored model and domestic in the older descriptive scripts.
