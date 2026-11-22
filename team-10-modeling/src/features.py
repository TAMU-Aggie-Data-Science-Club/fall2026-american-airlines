"""Team 10 feature set (Deliverable 3): one function per step, imported by D4 and D5.

Prediction: passengers on one directional route (ORG -> DES) in one departure month.
Cutoff: the forecast origin T. A forecast for month m is made at T, and m can be up to 24 months
later. A feature is allowed only if its value is known at T for every horizon 1-24.

Model target: fitted models do not predict passengers directly. They predict the route's
**year-over-year log growth** relative to an anchor (same route, same calendar month, one year
earlier). At forecast time the anchor is the seasonal naive (the year before T), and the growth
is applied once for year 1 and twice for year 2. This keeps every input known at T, avoids
feeding forecasts back as lags, and stops a model from extrapolating a raw level trend.
Rationale per column: ``results/feature-set.md``.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

KEY = ["ORG_AIRPRT", "DES_AIRPRT"]

# Dropped from the passenger file (03_leakage_and_features.ipynb):
#   OD_REV_USD   - revenue from the same passengers in the same month: an outcome, not an input
#   Unnamed: 0   - export row number
#   SF_LOAD_TMS  - one warehouse load timestamp for the whole extract
DROP_PAX = ["OD_REV_USD", "Unnamed: 0", "SF_LOAD_TMS"]

# Months whose year-over-year change reflects the 2020 collapse and recovery, not normal demand.
# A training pair is dropped if its target month OR its anchor month falls inside. Assumption,
# checked against the monthly network plot in 02_eda_demand.ipynb; revisit on the real data.
DISRUPTION = (pd.Timestamp("2020-03-01"), pd.Timestamp("2021-12-01"))

# Annual growth multipliers outside this range are treated as implausible for a two-year plan.
GROWTH_CAP = (0.8, 1.25)


def shift_months(dates: pd.Series, months) -> pd.Series:
    """Shift month-start dates by a (vectorised) number of months."""
    idx = dates.dt.year * 12 + dates.dt.month - 1 + months
    return pd.to_datetime({"year": idx // 12, "month": idx % 12 + 1, "day": 1})


def in_disruption(dates: pd.Series) -> pd.Series:
    return (dates >= DISRUPTION[0]) & (dates <= DISRUPTION[1])


def clean_pax(pax: pd.DataFrame) -> pd.DataFrame:
    """Keep the route-month grain: drop leak/ID columns and bad rows, one row per route-month."""
    out = pax.drop(columns=[c for c in DROP_PAX if c in pax.columns])
    out = out.dropna(subset=["DATE", "OD_PAX", *KEY])
    out = out[out["OD_PAX"] > 0]
    out = out[out["ORG_AIRPRT"] != out["DES_AIRPRT"]]
    # If the grain is ever violated, sum duplicates rather than double-count them.
    out = out.groupby([*KEY, "DATE"], as_index=False)["OD_PAX"].sum()
    return out.sort_values([*KEY, "DATE"]).reset_index(drop=True)


def airport_table(apt: pd.DataFrame) -> pd.DataFrame:
    """One row per airport code: decimal coordinates plus fixed attributes."""
    a = apt.drop_duplicates("AIRPRT_CD").set_index("AIRPRT_CD")
    lat = a["LAT_DEGREE_QTY"] + a["LAT_MINUTE_QTY"] / 60 + a["LAT_SECOND_QTY"] / 3600
    lng = a["LNGTD_DEGREE_QTY"] + a["LNGTD_MINUTE_QTY"] / 60 + a["LNGTD_SECOND_QTY"] / 3600
    out = pd.DataFrame(index=a.index)
    out["lat"] = np.where(a["LAT_HEMSPHR_CD"].eq("S"), -lat, lat).astype(float)
    out["lng"] = np.where(a["LNGTD_HEMSPHR_CD"].eq("W"), -lng, lng).astype(float)
    out["country"] = a["CNTRY_CD"].fillna("UNKNOWN")
    out["state"] = a["STATE_PROVNC_CD"].fillna("NONE")   # blank = outside US/Canada, not unknown
    out["world_area"] = pd.to_numeric(a["WRLD_AREA_DOT_CD"], errors="coerce")
    out["runway_ft"] = pd.to_numeric(a["LNGST_RUNWAY_FT_QTY"], errors="coerce")
    out["elevation_ft"] = pd.to_numeric(a["ELEVATN_FT_QTY"], errors="coerce")
    return out


def route_table(routes: pd.DataFrame, airports: pd.DataFrame) -> pd.DataFrame:
    """Static route attributes for every (ORG, DES) pair in ``routes``."""
    r = routes[KEY].drop_duplicates().reset_index(drop=True)
    o = airports.add_prefix("org_").reindex(r["ORG_AIRPRT"]).reset_index(drop=True)
    d = airports.add_prefix("des_").reindex(r["DES_AIRPRT"]).reset_index(drop=True)
    r = pd.concat([r, o, d], axis=1)
    la1, lo1, la2, lo2 = (np.radians(r[c]) for c in ("org_lat", "org_lng", "des_lat", "des_lng"))
    h = np.sin((la2 - la1) / 2) ** 2 + np.cos(la1) * np.cos(la2) * np.sin((lo2 - lo1) / 2) ** 2
    r["distance_mi"] = 3959 * 2 * np.arcsin(np.sqrt(h))
    r["log_distance"] = np.log1p(r["distance_mi"])
    r["is_international"] = ((r["org_country"] != "US") | (r["des_country"] != "US")).astype(float)
    return r


STATIC_NUMERIC = ["log_distance", "is_international", "org_lat", "org_lng", "des_lat", "des_lng",
                  "org_runway_ft", "des_runway_ft", "org_elevation_ft", "des_elevation_ft",
                  "org_world_area", "des_world_area"]
MODEL_NUMERIC = ["anchor_log", "month_sin", "month_cos", *STATIC_NUMERIC]
MODEL_CATEGORICAL = ["ORG_AIRPRT", "DES_AIRPRT", "month"]
FEATURES = MODEL_NUMERIC + MODEL_CATEGORICAL


def model_inputs(rows: pd.DataFrame, routes: pd.DataFrame) -> pd.DataFrame:
    """Add model inputs to rows that carry ORG, DES, DATE and an ``anchor`` passenger count."""
    out = rows.copy()
    out["anchor_log"] = np.log1p(out["anchor"])
    m = out["DATE"].dt.month
    out["month"] = m
    out["month_sin"] = np.sin(2 * np.pi * m / 12)
    out["month_cos"] = np.cos(2 * np.pi * m / 12)
    return out.merge(routes[[*KEY, *STATIC_NUMERIC]], on=KEY, how="left")


def training_pairs(pax: pd.DataFrame, routes: pd.DataFrame, origin: pd.Timestamp) -> pd.DataFrame:
    """Year-over-year pairs (month m, same month a year earlier) realised before ``origin``,
    excluding pairs that touch the disruption window. Target column: ``log_growth``."""
    hist = pax[pax["DATE"] < origin][[*KEY, "DATE", "OD_PAX"]]
    rows = hist.copy()
    rows["anchor_date"] = shift_months(rows["DATE"], -12)
    lag = hist.rename(columns={"DATE": "anchor_date", "OD_PAX": "anchor"})
    rows = rows.merge(lag, on=[*KEY, "anchor_date"], how="inner")
    rows = rows[~in_disruption(rows["DATE"]) & ~in_disruption(rows["anchor_date"])]
    rows["log_growth"] = np.log(rows["OD_PAX"] / rows["anchor"])
    return model_inputs(rows, routes).reset_index(drop=True)
