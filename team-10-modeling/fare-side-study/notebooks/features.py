"""Team 10 feature set (Deliverable 3) -- the single source of truth.

Deliverables 4 and 5 call ``clean()`` and ``build_features()``; they never
re-implement this logic. The reason for every kept, engineered and dropped
column is in ``team-10-modeling/results/feature-set.md``.

    from load_data import load_data
    from features import clean, build_features
    df = clean(load_data())
    X, y = build_features(df)
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from load_data import CABIN_ORDER, TARGET

# AA hubs (origin or destination being a hub changes the competitive picture).
AA_HUBS = {"DFW", "CLT", "ORD", "PHX", "MIA", "PHL", "DCA", "LAX", "JFK", "LGA"}

# Outlier rule: a quote more than 3x (or under 1/3 of) the lowest competitor
# quote for the same itinerary is treated as a data-entry error, as is a
# non-positive price. Chosen in 02_eda_saideep.ipynb (section 4).
MAX_COMP_RATIO = 3.0

NUMERIC_FEATURES = [
    "days_to_departure",
    "log_distance",
    "departure_hour",
    "month",
    "load_factor",
    "log_competitor_price",
    "cabin_rank",
]
BINARY_FEATURES = [
    "is_weekend_departure",
    "is_holiday_period",
    "is_summer",
    "nonstop",
    "is_longhaul",
    "red_eye",
    "last_minute",
    "org_is_hub",
    "des_is_hub",
]
FEATURES = NUMERIC_FEATURES + BINARY_FEATURES

# Same set without the competitor quote -- used for the sensitivity check in
# case the competitor feed turns out not to be available at quote time.
FEATURES_NO_COMPETITOR = [f for f in FEATURES if f != "log_competitor_price"]


def outlier_mask(df: pd.DataFrame) -> pd.Series:
    """True for rows the outlier rule removes."""
    ratio = df[TARGET] / df["competitor_min_price"]
    return (df[TARGET] <= 0) | (ratio > MAX_COMP_RATIO) | (ratio < 1 / MAX_COMP_RATIO)


def clean(df: pd.DataFrame) -> pd.DataFrame:
    """Drop exact duplicates, rows with a missing target, and price outliers."""
    out = df.drop_duplicates()
    out = out[out[TARGET].notna()]
    out = out[~outlier_mask(out)]
    return out.reset_index(drop=True)


def build_features(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Return (X, y). y is the dollar price; models train on log1p(y)."""
    X = pd.DataFrame(index=df.index)
    X["days_to_departure"] = df["days_to_departure"].astype(float)
    X["log_distance"] = np.log(df["distance_miles"].astype(float))
    X["departure_hour"] = df["departure_hour"].astype(float)
    X["month"] = df["month"].astype(float)
    X["load_factor"] = df["load_factor"].astype(float)
    X["log_competitor_price"] = np.log(df["competitor_min_price"].astype(float))
    X["cabin_rank"] = df["cabin"].cat.codes.astype(float) if hasattr(df["cabin"], "cat") \
        else df["cabin"].map({c: i for i, c in enumerate(CABIN_ORDER)}).astype(float)

    X["is_weekend_departure"] = df["is_weekend_departure"].astype(float)
    X["is_holiday_period"] = df["is_holiday_period"].astype(float)
    X["is_summer"] = df["month"].isin([6, 7, 8]).astype(float)
    X["nonstop"] = df["nonstop"].astype(float)
    X["is_longhaul"] = (df["distance_miles"] > 2200).astype(float)
    X["red_eye"] = ((df["departure_hour"] >= 21) | (df["departure_hour"] <= 5)).astype(float)
    X["last_minute"] = (df["days_to_departure"] <= 14).astype(float)
    X["org_is_hub"] = df["origin"].astype(str).isin(AA_HUBS).astype(float)
    X["des_is_hub"] = df["destination"].astype(str).isin(AA_HUBS).astype(float)

    y = df[TARGET].astype(float)
    return X[FEATURES], y
