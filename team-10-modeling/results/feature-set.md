# Feature set — Team 10 (Deliverable 3)

**Grain:** one directional route (`ORG_AIRPRT` → `DES_AIRPRT`) in one departure month.
**Cutoff:** the forecast origin T. An input is allowed only if it is known at T for **every**
horizon from 1 to 24 months.
**Code:** `src/features.py`. `clean_pax()`, `airport_table()` and `route_table()` build the inputs.
`training_pairs()` builds the training frame and `model_inputs()` builds the forecast frame.
**Named model family:** seasonal naive as the reference, with year-over-year **growth** models
(Ridge, gradient-boosted trees) as challengers. The evidence is in `02_eda_demand.ipynb` §4.

## Target

Fitted models learn `log_growth = log(OD_PAX in month m / OD_PAX in month m − 12)` for the same
route. At forecast time the anchor is the seasonal naive (the same calendar month in the year
before T), and the predicted growth (capped to 0.8–1.25 per year) is applied once for months 1–12
and twice for months 13–24. Why growth and not levels: the passenger series has a 2020 collapse and
recovery. A level model trained across it reads the rebound as a trend and extrapolates it.

## Rows

| rule | why |
|---|---|
| Drop rows with a missing date, route or `OD_PAX`, or `OD_PAX ≤ 0` | Nothing to learn from |
| Drop origin = destination | Not a route |
| Sum any duplicate route-months | Protects the grain if the export ever repeats a row (01 checks for this) |
| Training pairs touching **2020-03 → 2021-12** (target or anchor month) are excluded | Their year-over-year change reflects the disruption, not normal demand. `02` prints a CHECK if months after the window still look abnormal. |
| Missing route-months are **not** filled with zero for training | A missing row may be "no service" or "not recorded" (open PM question) |

## Inputs

| input | source | type | why |
|---|---|---|---|
| `anchor_log` | `OD_PAX`, same month one year earlier (forecast time: the year before T) | numeric | Route size. Growth differs between big and small routes. |
| `month_sin`, `month_cos`, `month` | `DEP_MONTH` | numeric / categorical | Growth can be season-specific (e.g. a summer that grows faster than winter) |
| `log_distance` | great-circle distance from airport coordinates | numeric | Short-haul and long-haul markets grow differently |
| `is_international` | `CNTRY_CD` of both airports | 0/1 | International demand has its own cycle and recovery path |
| `org_lat/lng`, `des_lat/lng` | airport DMS fields → decimal degrees | numeric | Regional effects without one-hot encoding every region |
| `org/des_world_area` | `WRLD_AREA_DOT_CD` | numeric | DOT region code: a coarser regional grouping |
| `org/des_runway_ft`, `org/des_elevation_ft` | airport file | numeric | Fixed airport scale and constraints |
| `ORG_AIRPRT`, `DES_AIRPRT` | passenger file | categorical | Airport-specific growth (one-hot in Ridge, native categories in the tree model) |

## Dropped

| column | why |
|---|---|
| `OD_REV_USD` | Revenue from the same passengers in the same month. It is an outcome, not available at T (correlation with the target is printed in `03`). |
| `Unnamed: 0` (both files) | Export row number |
| `SF_LOAD_TMS` (both files) | One warehouse load timestamp for the whole extract |
| `AIRPRT_NM` | Text name of the code |
| `CITY_METRO_IATA_CD` | Mostly duplicates the airport code; a candidate for later metro-level pooling |
| `STATE_PROVNC_CD` | Kept in `airport_table()` (blank → `NONE` for non-US airports) but not a model input. Coordinates carry the regional signal without ~50 more levels. |
| Lag-1 passengers, rolling means, lag-12 for months 13–24 | Not known at T for the whole horizon |

## Open questions for the PM

- Is `OD_PAX` an estimate? Fractional values suggest so.
- Does a missing route-month mean "no service" or "not recorded"?
- When exactly did demand normalise? Set `DISRUPTION` from the real data's CHECK output.
