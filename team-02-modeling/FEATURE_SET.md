# Team 02 — Deliverable 3 Feature Set

> Assembled by the lead from each member's hand-off. Built by code in `notebooks/feature_set.ipynb`.
> **NDA:** describe columns and decisions only — no raw values, sample rows, or charts with real numbers.

| Section | Owner |
|---|---|
| Target and loss | Aidan Chance |
| Leakage decisions | Sahith Yamani |
| Missingness handling | Chau Ha |
| Validation scheme and date features | Akshith Reddy Sheri |
| Encoding, relationships, feature-set code | Aditi Walvekar's area, completed by Akshith Reddy Sheri |
| Model family, feature register, data access | Akshith Reddy Sheri |

## Target and model family

- **Target:** `OD_PAX` — passengers on an origin–destination route in a departure month; regression on a
  right-skewed count. Large values are real busy routes and are kept. — **Aidan**
- **Loss:** `log1p` + squared error vs Poisson on raw counts — tested side by side in D4
  (`notebooks/model_comparison.ipynb`); the lower test WAPE wins. — **Aidan**
- **Model family:** gradient-boosted trees (XGBoost), with Ridge on `log1p` and a seasonal naive baseline as comparisons.
- **Why:**
  1. Airports and routes are high-cardinality categoricals; trees take them natively, one-hot does not scale
     (`eda_categoricals_akshithreddy-sheri.ipynb`).
  2. Seasonality differs between domestic and international routes and across airports — interactions trees learn
     on their own (`eda_categoricals_akshithreddy-sheri.ipynb`, section 5).
  3. The target is skewed with very large routes; trees with a count or log loss handle that
     (`eda_aidan-chance.ipynb`).
- **Validation scheme:** time-ordered split on `DEP_YEAR`/`DEP_MONTH` — last 12 months as test (D4), last 24 months
  for the two-year forecast (D5). Never shuffled K-fold: a random split lets the model learn from the future
  (`eda_akshithreddy-sheri.ipynb`, section 7). — **Akshith**

## Leakage decisions — Sahith

| Column | Decision | Why — would we have it at prediction time? |
|---|---|---|
| `OD_REV_USD` | leak → drop | realised revenue for the same route-month; only known after travel |
| `SF_LOAD_TMS` | drop | system load timestamp, not information about demand |
| `Unnamed: 0` | id → drop | row index from the export |
| `DEP_MONTH`, `DEP_YEAR`, `ORG_AIRPRT`, `DES_AIRPRT` | keep | known before the month happens |

## Missingness handling — Chau

| Column | Missing (rounded) | Random or tied to something? | Handling |
|---|---|---|---|
| Passenger table columns | none | — | none needed |
| `STATE_PROVNC_CD` (airport table) | about 83% | tied to airports outside US/Canada | fill `"INTL"` (not used in the model for now) |
| Missing route-months | a share of routes skip some months | tied to route (seasonal / new / dropped routes) | left missing, not zero (open PM question); lag features are missing + a `route_seen_*` flag |

## Date features — Akshith

| Feature | Built from | Why |
|---|---|---|
| `DEP_MONTH`, `month_sin`, `month_cos` | departure month | seasonality |
| `DEP_YEAR`, `months_since_start` | departure year/month | trend |
| `pax_same_month_last_year`, `route_seen_last_year` | same route, 12 months earlier | strongest single predictor (D4 only) |
| `pax_same_month_2_years_ago`, `route_seen_2_years_ago` | same route, 24 months earlier | known for every month of a two-year forecast (D5) |

## Encoding and relationships

| Column | Cardinality (rounded) | Encoding | Why |
|---|---|---|---|
| `ORG_AIRPRT`, `DES_AIRPRT` | hundreds | trees: native categorical (unseen → missing); Ridge: one-hot, rare airports grouped | high cardinality |
| `route` | thousands | not a model input (trees combine origin + destination); used for grouping | too many values |
| `ORG_CNTRY_CD`, `DES_CNTRY_CD` | a handful | one-hot / native categorical; `UNKNOWN` if airport not in reference table | domestic vs international demand |
| `is_international` | 2 | 0/1 | different seasonality |

Encoders are always fit inside the model pipeline on training months only.

## Feature register

| Feature | Kept / engineered / dropped | Encoding | Reason (owner) |
|---|---|---|---|
| `ORG_AIRPRT`, `DES_AIRPRT` | kept | categorical | route identity (Aditi's area / Akshith) |
| `ORG_CNTRY_CD`, `DES_CNTRY_CD`, `is_international` | engineered (airport table) | categorical / 0-1 | domestic vs international (Akshith) |
| `DEP_MONTH`, `DEP_YEAR`, `month_sin`, `month_cos`, `months_since_start` | kept / engineered | numeric | season + trend (Akshith) |
| `pax_same_month_last_year`, `route_seen_last_year` | engineered | numeric | seasonal lag (Akshith) |
| `pax_same_month_2_years_ago`, `route_seen_2_years_ago` | engineered | numeric | two-year horizon (Akshith) |
| `OD_REV_USD` | dropped | — | leak (Sahith) |
| `SF_LOAD_TMS`, `Unnamed: 0` | dropped | — | timestamp / index (Sahith) |
| `OD_PAX` | target | — | (Aidan) |

## Data access (for `DATA.md`) — Akshith

- Google Drive link shared privately by the PMs, downloaded to the git-ignored `data/raw/`, never committed.
- Files: `AA_PAX_FORECASTING.csv`, `AA_AIRPORT_INFO.csv`. Downloaded: *(date)*.

## Open questions for the PMs
- Does a missing route-month mean zero passengers or no record? (We treat it as no record.)
- Is `SF_LOAD_TMS` only a system load time? (We drop it either way.)
