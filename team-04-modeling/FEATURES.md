# Team 04 feature set

Written by `02_feature_set.ipynb` from **REAL** data.

- **Model family:** Gradient-boosted trees, absolute-error loss on log1p(target), forecasting 1 month(s) ahead.
- **Training window:** only after the shock.
- **Always kept:** `LAG_1`, the route's latest month that is known at forecast time.
- **Dropped as a leak:** REVENUE, known only after passengers fly.
- **How groups were chosen:** refits on date-based backtests. A group stays only if removing it worsens WAPE by at least 0.001.

| Group | Columns | Decision | Why we tested it |
|---|---|---|---|
| airport attributes | ORG_LAT, ORG_LON, ORG_RUNWAY_FT, ORG_ELEVATION_FT, DES_LAT, DES_LON, DES_RUNWAY_FT, DES_ELEVATION_FT | drop | Size and location of each airport. Could help routes with little history. |
| region | ORG_REGION, DES_REGION | drop | Broad demand differences between parts of the world. |
| distance and international | DISTANCE_MI, IS_INTL | drop | Fixed per route, so it could help when route history is thin. |
| trend | TREND | drop | A date counter, though trees can't extrapolate past the dates they trained on. |
| recent lags and route mean | LAG_2, LAG_3, ROUTE_MEAN | drop | Smooths month-to-month noise in the latest value. |
| route IDs | ORIGIN, DEST | keep | Origin and destination as categories, for lasting differences between airports. |
| month | MONTH_SIN, MONTH_COS | keep | Seasonality with no jump between December and January. |
| year-over-year | YOY_LOG_RATIO, LAG_13 | keep | Recent change against a year earlier, so trees can follow a level shift. |
| same month last year | LAG_12 | keep | The classic seasonal signal, known at any horizon up to 12. |
