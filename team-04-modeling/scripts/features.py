"""Build candidate features. Run this file to self-check: python features.py"""
import numpy as np
import pandas as pd

import config as c

AIRPORT_AREAS = [f"{p}_{col}" for p in ("ORG", "DES") for col in c.AIRPORT_CATEGORICAL]
CATEGORICAL = c.ROUTE_KEYS + AIRPORT_AREAS

# Why each group is worth testing. Whether it stays is decided in 02_feature_set when it runs.
WHY = {
    "airport attributes": "Size and location of each airport. Could help routes with little history.",
    "region": "Broad demand differences between parts of the world.",
    "distance and international": "Fixed per route, so it could help when route history is thin.",
    "trend": "A date counter, though trees can't extrapolate past the dates they trained on.",
    "recent lags and route mean": "Smooths month-to-month noise in the latest value.",
    "route IDs": "Origin and destination as categories, for lasting differences between airports.",
    "month": "Seasonality with no jump between December and January.",
    "year-over-year": "Recent change against a year earlier, so trees can follow a level shift.",
    "same month last year": "The classic seasonal signal, known at any horizon up to 12.",
}


def feature_groups(horizon=c.HORIZON):
    """Candidate groups in testing order, most speculative first. LAG_<horizon> always stays."""
    groups = {
        "airport attributes": [f"{p}_{col}" for p in ("ORG", "DES") for col in c.AIRPORT_NUMERIC],
        "region": AIRPORT_AREAS,
        "distance and international": ["DISTANCE_MI", "IS_INTL"],
        "trend": ["TREND"],
        "recent lags and route mean": [f"LAG_{horizon + 1}", f"LAG_{horizon + 2}", "ROUTE_MEAN"],
        "route IDs": list(c.ROUTE_KEYS),
        "month": ["MONTH_SIN", "MONTH_COS"],
        "year-over-year": ["YOY_LOG_RATIO", f"LAG_{horizon + 12}"],
        "same month last year": ["LAG_12"] if horizon < 12 else [],
    }
    return {name: cols for name, cols in groups.items() if cols}


def split_columns(cols):
    """(numeric, categorical), in the order the model pipelines expect."""
    return [x for x in cols if x not in CATEGORICAL], [x for x in cols if x in CATEGORICAL]


def clean(pax):
    """Mechanical cleaning only: drop exact duplicates and rows with no usable target."""
    pax = pax.drop_duplicates()
    return pax[pax[c.TARGET].notna() & (pax[c.TARGET] >= 0)].copy()


def airport_attributes(airports):
    """Add signed decimal LAT and LON, and keep one row per airport."""
    df = airports.drop_duplicates(c.AIRPORT_KEY).copy()
    for axis, negative, valid in [("LAT", "S", {"N", "S"}), ("LON", "W", {"E", "W"})]:
        if not set(df[f"{axis}_HEMI"].dropna()) <= valid:
            raise ValueError(f"Unexpected hemisphere codes in {axis}_HEMI")
        decimal = df[f"{axis}_DEG"] + df[f"{axis}_MIN"] / 60 + df[f"{axis}_SEC"] / 3600
        df[axis] = decimal * np.where(df[f"{axis}_HEMI"] == negative, -1, 1)
    return df[[c.AIRPORT_KEY, "COUNTRY"] + c.AIRPORT_NUMERIC + c.AIRPORT_CATEGORICAL]


def miles_between(lat1, lon1, lat2, lon2):
    """Great-circle distance (haversine)."""
    lat1, lon1, lat2, lon2 = map(np.radians, (lat1, lon1, lat2, lon2))
    a = np.sin((lat2 - lat1) / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2) ** 2
    return 3958.8 * 2 * np.arcsin(np.sqrt(a))


def add_lag(df, k):
    """Match on the actual past date, not the previous row: routes can skip periods."""
    past = df[c.ROUTE_KEYS + [c.DATE, c.TARGET]].copy()
    past[c.DATE] = past[c.DATE] + k * c.PERIOD
    past = past.rename(columns={c.TARGET: f"LAG_{k}"})
    return df.merge(past, on=c.ROUTE_KEYS + [c.DATE], how="left", validate="one_to_one")


def build_features(pax, airports, horizon=c.HORIZON):
    """Return one row per route-month with the target, the date and every candidate feature."""
    df = pax[c.ROUTE_KEYS + [c.DATE, c.TARGET]].copy()
    for k in sorted({horizon, horizon + 1, horizon + 2, horizon + 12} | ({12} if horizon < 12 else set())):
        df = add_lag(df, k)

    df["ROUTE_MEAN"] = df[[f"LAG_{k}" for k in (horizon, horizon + 1, horizon + 2)]].mean(axis=1)
    df["YOY_LOG_RATIO"] = np.log1p(df[f"LAG_{horizon}"]) - np.log1p(df[f"LAG_{horizon + 12}"])
    month = df[c.DATE].dt.month
    df["MONTH_SIN"] = np.sin(2 * np.pi * month / 12)
    df["MONTH_COS"] = np.cos(2 * np.pi * month / 12)
    df["TREND"] = df[c.DATE].dt.year * 12 + month

    attrs = airport_attributes(airports)
    for prefix, key in zip(("ORG", "DES"), c.ROUTE_KEYS):
        renamed = attrs.rename(columns={col: f"{prefix}_{col}" for col in attrs.columns if col != c.AIRPORT_KEY})
        df = df.merge(renamed, left_on=key, right_on=c.AIRPORT_KEY, how="left", validate="many_to_one").drop(columns=c.AIRPORT_KEY)
    df["DISTANCE_MI"] = miles_between(df["ORG_LAT"], df["ORG_LON"], df["DES_LAT"], df["DES_LON"])
    df["IS_INTL"] = (df["ORG_COUNTRY"] != df["DES_COUNTRY"]).astype(float).where(
        df["ORG_COUNTRY"].notna() & df["DES_COUNTRY"].notna()
    )
    df = df.drop(columns=["ORG_COUNTRY", "DES_COUNTRY"])

    numeric, _ = split_columns([col for cols in feature_groups(horizon).values() for col in cols] + [f"LAG_{horizon}"])
    df[numeric] = df[numeric].astype(float)
    df[CATEGORICAL] = df[CATEGORICAL].astype(object).where(df[CATEGORICAL].notna(), "missing").astype(str)
    return df


def after_shock(df):
    """Rows after the last month with a total over 30% below the median month. All rows if there's no such month."""
    total = df.groupby(c.DATE)[c.TARGET].sum()
    last = total[total < 0.7 * total.median()].index.max()
    return df[c.DATE] > last if pd.notna(last) else pd.Series(True, index=df.index)


def time_folds(dates, folds=c.BACKTEST_FOLDS, length=c.HOLDOUT_PERIODS):
    """Yield (train_mask, test_mask): train on everything before each test window, never after."""
    periods = np.sort(dates.unique())
    for i in range(folds, 0, -1):
        start = len(periods) - i * length
        if start < 1:
            continue
        window = periods[start:start + length]
        yield dates < window[0], dates.isin(window)


def _check():
    """Smallest thing that fails if the time, join or leak logic breaks."""
    dates = pd.to_datetime(["2024-01-01", "2024-02-01", "2024-04-01", "2024-05-01"])
    pax = pd.DataFrame({"ORIGIN": "A", "DEST": "B", "DATE": dates, "PAX": [10.0, 20.0, 40.0, 50.0], "REVENUE": 1.0})
    airports = pd.DataFrame({
        "AIRPORT": ["A", "B"], "COUNTRY": ["AA", "NA"], "REGION": [1, 2],
        "LAT_DEG": [0, 0], "LAT_MIN": [0, 0], "LAT_SEC": [0, 0], "LAT_HEMI": ["N", "S"],
        "LON_DEG": [0, 1], "LON_MIN": [0, 0], "LON_SEC": [0, 0], "LON_HEMI": ["E", "W"],
        "RUNWAY_FT": [1, 2], "ELEVATION_FT": [1, 2],
    })

    df = build_features(pax, airports)
    feb, apr = df.DATE == "2024-02-01", df.DATE == "2024-04-01"
    assert df.loc[feb, "LAG_1"].item() == 10.0, "lag takes the previous period"
    assert np.isnan(df.loc[apr, "LAG_1"].item()), "a skipped period is missing, not the prior row"
    assert abs(df["DISTANCE_MI"].iloc[0] - 69.1) < 0.2, "one degree of longitude at the equator is about 69 miles"
    assert df["IS_INTL"].iloc[0] == 1.0, "different countries are international"

    candidates = [col for cols in feature_groups(1).values() for col in cols]
    assert not {c.TARGET, *c.LEAKY} & set(candidates), "target and leaks are never candidates"
    assert set(WHY) == set(feature_groups(1)), "every group has a reason"
    assert all(int(col[4:]) >= 3 for col in sum(feature_groups(3).values(), []) if col.startswith("LAG_")), "no lag shorter than the horizon"
    assert "LAG_12" not in sum(feature_groups(13).values(), []), "lag 12 is illegal at a 13-period horizon"

    shocked = pd.DataFrame({"DATE": dates, "PAX": [100.0, 1.0, 100.0, 100.0]})
    assert not after_shock(shocked).iloc[1] and after_shock(shocked).iloc[2], "only months after the shock are in"
    assert after_shock(pd.DataFrame({"DATE": dates, "PAX": 100.0})).all(), "no shock keeps every month"

    later = pax.assign(PAX=pax.PAX.where(pax.DATE < "2024-05-01", 9999.0))
    changed = build_features(later, airports)
    early, lags = df.DATE < "2024-05-01", [col for col in candidates if col.startswith(("LAG_", "ROUTE_MEAN", "YOY"))]
    assert df.loc[early, lags].equals(changed.loc[early, lags]), "the future can't change past features"

    # A route with an airport missing from the table is flagged, not dropped.
    orphan = build_features(pax.assign(DEST="ZZZ"), airports)
    assert len(orphan) == len(pax) and (orphan["DES_REGION"] == "missing").all() and orphan["IS_INTL"].isna().all()

    long_dates = pd.Series(pd.date_range("2020-01-01", periods=60, freq="MS"))
    folds = list(time_folds(long_dates))
    assert len(folds) == 3 and all(long_dates[tr].max() < long_dates[te].min() for tr, te in folds), "no time overlap"

    assert len(clean(pd.concat([pax, pax]))) == len(pax), "exact duplicates dropped"
    print("features.py self-check passed")


if __name__ == "__main__":
    _check()
