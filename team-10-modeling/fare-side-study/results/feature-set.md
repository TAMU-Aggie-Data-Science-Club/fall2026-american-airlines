# Feature set — Team 10 (Deliverable 3)

**Data:** `data/raw/aa_fares_compact_SYNTHETIC.csv` (200 fare quotes, 13 columns, synthetic).
**Target:** `total_price`. Models train on `log1p(total_price)` and are scored in dollars.
**Code:** `notebooks/features.py`. `clean()` drops bad rows and `build_features()` returns `X, y`.
The evidence for each decision is in `notebooks/02_eda_saideep.ipynb`.
**Model family the EDA points to:** gradient-boosted trees, with a regularised linear (Ridge) baseline
on log price that the trees have to beat. D4 then tested this. Ridge won, because the competitor
quote is a near-linear driver (see `baseline-decision.md`).

## Row cleaning (`clean()`)

| rule | rows removed | why |
|---|---|---|
| exact duplicate rows | 0 | Repeated quotes would be double-counted. Kept as a guard for future re-cuts of the file. |
| missing target | 0 | Cannot train or score without it. |
| `total_price ≤ 0`, or more than 3× (or under ⅓ of) `competitor_min_price` | 1 | Every other row sits within 0.81×–1.31× of the competitor quote. The one removed row is a $3,804 Main Cabin fare against a $326 competitor quote: a data-entry error. One such row dominates a squared-error fit. |

199 rows remain.

## Features (16)

| feature | from | type | why it is in |
|---|---|---|---|
| `log_competitor_price` | `competitor_min_price` | numeric | Correlation with price is 0.99 (0.994 log-log). Known at quote time (a market quote, not derived from our price). Logged because price effects are multiplicative. |
| `log_distance` | `distance_miles` | numeric | Price vs distance is roughly linear on a log-log scale. |
| `cabin_rank` | `cabin` | ordinal 0–3 | Basic < Main < Premium < Business/First. One ordered column instead of four dummies. Business/First's median is ~3× Basic. |
| `days_to_departure` | as is | numeric | Raw booking window, for any slope beyond the step below. |
| `last_minute` | `days_to_departure ≤ 14` | 0/1 | Fares are ~0.95–1.07× their cabin median beyond 14 days and ~1.8× inside. That is a step a linear term cannot draw. |
| `load_factor` | as is | numeric | Fuller flights price higher, though the effect is weak. Kept pending the PM question about when it is measured. |
| `departure_hour` | as is | numeric | Time-of-day effects. |
| `red_eye` | hour ≥ 21 or ≤ 5 | 0/1 | Red-eye and very early departures are typically discounted. |
| `month` | as is | numeric | Seasonality. |
| `is_summer` | month ∈ {6, 7, 8} | 0/1 | Summer peak, per the data notes. |
| `is_weekend_departure` | as is | 0/1 | Fri–Sun lift. |
| `is_holiday_period` | as is | 0/1 | Holiday lift. |
| `nonstop` | as is | 0/1 | Nonstop service premium. |
| `is_longhaul` | distance > 2200 | 0/1 | Transcontinental routes are a distinct product. |
| `org_is_hub`, `des_is_hub` | `origin`, `destination` ∈ AA hubs | 0/1 | Hub presence changes the competitive picture. This replaces ~50 sparse airport dummies. |

## Dropped or not used

| column | decision | why |
|---|---|---|
| `origin`, `destination` (raw codes) | replaced by hub flags + `log_distance` | 24 / 25 levels and 154 distinct routes in 199 rows. Dummies would memorise single rows. |
| route (`origin-destination`) | not built | Same problem, worse: almost every route appears once or twice. |
| `base_fare`, `taxes_fees` | not in this file | In the full synthetic sample they sum exactly to the target, so they would be leakage. |
| `comp_gap = price − competitor` | not built | Contains the target, so it would be leakage. |

## Open questions that could change this set

- **`load_factor`:** measured at quote time or at departure? If at departure, it is leakage and must go.
- **`competitor_min_price`:** captured at the same moment and for the same cabin as the AA quote?
  If it were captured *after* AA's price (competitors reacting to us), it would leak. D4 therefore
  also scores every model without it.
