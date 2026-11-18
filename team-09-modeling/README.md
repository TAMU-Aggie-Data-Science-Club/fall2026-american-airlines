# Team 9 modeling

Deliverable 3 feature set for gradient-boosted trees. The target is `OD_PAX`: passengers on one origin-destination route in one departure month. The cutoff is the start of that month.

`notebooks/pri-biswal.ipynb` builds the frame. `build_features` returns `X` and `y`. Deliverable 4 tunes this frame and does not refit the feature logic.

## Where the EDA is

| Notebook | What it covers |
|---|---|
| `lukas.ipynb` | Shape, column meanings, and a first look at the target distribution |
| `anish.ipynb` | Missingness, including the non-US blanks in state and country |
| `distributions.ipynb` | Target skew, the month pattern, and why this panel is not one ARIMA series |
| `leakage.ipynb` | What is known at the cutoff, and what is dropped |
| `deekshitha.ipynb` | Passenger-volume histograms, the month pattern, and why same-month revenue is not a feature |
| `pri-biswal.ipynb` | The feature set those notes add up to |
| `two-year-forecast.ipynb` | The two-year route forecast, the seasonal-naive baseline, and the scored gradient-boosted tree |
| `results/model-comparison.md` | Candidate results, what is comparable, and what was excluded |
| `results/baseline-decision.md` | The six-check bar and the model carried forward |
| `results/forecast-evaluation.md` | Held-out scores, segments, and the approximate network band |
| `results/final-model-report.md` | The handover: design, forecast, use, and limits |
| `visuals/` | Backtest, horizon, segment, and forward-forecast charts |

The carried forecast is the route-level seasonal naive. The gradient-boosted tree was scored on the Deliverable 3 frame and was not carried.

## The model

Gradient-boosted trees were the Deliverable 3 candidate. The target is skewed, the useful inputs are airport categories plus a few fixed numbers, and a route-month table is a panel rather than one ordered series. The two-year test carried the seasonal naive instead.

Eleven columns of `X` are pandas categories. Eleven are numbers. Deliverable 4 can pass `X` to XGBoost with `enable_categorical=True`, to LightGBM, or to `HistGradientBoostingRegressor`.

## State code

`ORG_STATE_PROVNC_CD` and `DES_STATE_PROVNC_CD` stay. Blanks are filled with `MISSING`. Those blanks are non-US airports, so filling in a state would invent one. About half the airport lookup is blank. `anish.ipynb` and `lukas.ipynb` both keep the column for that reason.

## Dropped

`OD_REV_USD` is that same month's revenue. `Unnamed: 0` is the export row number. `SF_LOAD_TMS` is one warehouse timestamp on every row of each file. Airport names and the raw degree, minute, and second fields are not separate features. The keep and drop tables are in `leakage.ipynb` and `pri-biswal.ipynb`.
