import argparse
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb

TARGET = "OD_PAX"
HORIZON = 24
FIRST_TRAIN_ORIGIN = pd.Timestamp("2022-07-01")
VAL_START = pd.Timestamp("2023-07-01")
TEST_ORIGIN = pd.Timestamp("2024-07-01")
BACKTEST_ORIGINS = pd.date_range("2023-07-01", TEST_ORIGIN, freq="3MS")
INTERVAL = (0.10, 0.90)

PARAMS = dict(
    objective="reg:squarederror", learning_rate=0.05, max_depth=8, min_child_weight=50,
    subsample=0.8, colsample_bytree=0.8, tree_method="hist", max_cat_to_onehot=1, seed=7,
)

FEATURES = [
    "h", "month_sin", "month_cos",
    "last_same_month", "level_12", "level_3", "same_vs_level12", "level3_vs_level12", "growth_12",
    "pax_2019_same_month", "level12_vs_2019", "yield_12",
    "org_level_12", "des_level_12", "months_seen_12",
    "distance_mi", "domestic", "ORG_AIRPRT", "DES_AIRPRT",
    "org_runway_ft", "des_runway_ft", "org_world_area", "des_world_area", "org_country", "des_country",
    "covid_months_in_history",
]
COVID_MONTHS = pd.date_range("2020-03-01", "2021-12-01", freq="MS")


def find_repo_root():
    for p in [Path(__file__).resolve().parent, *Path(__file__).resolve().parents]:
        if (p / "DATA.md").exists() and (p / ".git").exists():
            return p
    raise FileNotFoundError("Run this from inside the repo clone.")


def find_source(raw_dir, stem):
    matches = [p for p in raw_dir.iterdir() if stem.lower() in p.name.lower() and p.suffix.lower() == ".csv"]
    if len(matches) != 1:
        raise FileNotFoundError(f"Need exactly one CSV with '{stem}' in its name in {raw_dir}, found {len(matches)}.")
    return matches[0]


def load_data(repo_root):
    raw = repo_root / "data" / "raw"
    pax = pd.read_csv(find_source(raw, "PAX_FORECASTING"), low_memory=False,
                      dtype={"ORG_AIRPRT": str, "DES_AIRPRT": str})
    airports = pd.read_csv(find_source(raw, "AIRPORT_INFO"), low_memory=False, dtype={"AIRPRT_CD": str})
    for df in (pax, airports):
        for c in df.select_dtypes(include="object"):
            df[c] = df[c].str.strip()
    return pax.drop_duplicates(), airports


def _row_mean(a):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        return np.nanmean(a, axis=1)


def _decimal_coord(a, kind):
    deg = a[f"{kind}_DEGREE_QTY"] + a[f"{kind}_MINUTE_QTY"] / 60 + a[f"{kind}_SECOND_QTY"] / 3600
    return deg.where(~a[f"{kind}_HEMSPHR_CD"].isin(["S", "W"]), -deg)


def build_features(pax, airports, origins, horizon=HORIZON, max_target=None, future=False):
    """One row per (route, forecast date, target month); every feature uses only months up to the forecast date."""
    df = pax.copy()
    df["DATE"] = pd.to_datetime(dict(year=df["DEP_YEAR"], month=df["DEP_MONTH"], day=1))
    df["ROUTE"] = df["ORG_AIRPRT"] + "-" + df["DES_AIRPRT"]
    months = pd.date_range(df["DATE"].min(), df["DATE"].max() + pd.DateOffset(months=horizon if future else 0), freq="MS")
    col = {m: i for i, m in enumerate(months)}

    wide_pax = df.pivot(index="ROUTE", columns="DATE", values=TARGET).reindex(columns=months)
    wide_rev = df.pivot(index="ROUTE", columns="DATE", values="OD_REV_USD").reindex(columns=months)
    routes = wide_pax.index
    P = wide_pax.to_numpy()
    YLD = (wide_rev / wide_pax).to_numpy()
    org = routes.str.split("-").str[0].to_numpy()
    des = routes.str.split("-").str[1].to_numpy()
    org_tot = pd.DataFrame(P).groupby(org).sum(min_count=1).loc[org].to_numpy()
    des_tot = pd.DataFrame(P).groupby(des).sum(min_count=1).loc[des].to_numpy()
    network = np.nansum(P, axis=0)
    pax_2019 = P[:, [col[m] for m in pd.date_range("2019-01-01", "2019-12-01", freq="MS")]]

    a = airports.dropna(subset=["AIRPRT_CD"]).drop_duplicates("AIRPRT_CD").set_index("AIRPRT_CD")
    lat, lon = _decimal_coord(a, "LAT"), _decimal_coord(a, "LNGTD")
    lat1, lon1, lat2, lon2 = (np.radians(s.loc[k].to_numpy(dtype=float)) for s, k in
                              ((lat, org), (lon, org), (lat, des), (lon, des)))
    hav = np.sin((lat2 - lat1) / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2) ** 2
    distance_mi = 3958.8 * 2 * np.arcsin(np.sqrt(hav))
    domestic = ((a["CNTRY_CD"].loc[org].to_numpy() == "US") & (a["CNTRY_CD"].loc[des].to_numpy() == "US")).astype(int)
    airport_codes = sorted(set(org) | set(des))
    countries = sorted(a["CNTRY_CD"].dropna().unique())
    airport_attrs = {}
    for end, codes in (("org", org), ("des", des)):
        airport_attrs[f"{end}_runway_ft"] = pd.to_numeric(a["LNGST_RUNWAY_FT_QTY"].reindex(codes), errors="coerce").to_numpy()
        airport_attrs[f"{end}_world_area"] = pd.to_numeric(a["WRLD_AREA_DOT_CD"].reindex(codes), errors="coerce").to_numpy()
        airport_attrs[f"{end}_country"] = pd.Categorical(a["CNTRY_CD"].reindex(codes).to_numpy(), categories=countries)

    parts = []
    for origin in origins:
        o = col[pd.Timestamp(origin)]
        level_12 = _row_mean(P[:, o - 11:o + 1])
        level_3 = _row_mean(P[:, o - 2:o + 1])
        level_12_prev = _row_mean(P[:, o - 23:o - 11]) if o >= 23 else np.full(len(P), np.nan)
        covid_in_history = int(((COVID_MONTHS > months[o] - pd.DateOffset(months=12)) & (COVID_MONTHS <= months[o])).sum())
        for h in range(1, horizon + 1):
            t = o + h
            if t >= len(months) or (max_target is not None and months[t] > max_target):
                break
            month = months[t].month
            same_col = t - 12 * int(np.ceil(h / 12))
            last_same = P[:, same_col]
            parts.append(pd.DataFrame({
                "ROUTE": routes, "ORIGIN": months[o], "DATE": months[t], "h": h,
                "month_sin": np.sin(2 * np.pi * month / 12), "month_cos": np.cos(2 * np.pi * month / 12),
                "last_same_month": last_same,
                "level_12": level_12,
                "level_3": level_3,
                "same_vs_level12": np.log1p(last_same) - np.log1p(level_12),
                "level3_vs_level12": np.log1p(level_3) - np.log1p(level_12),
                "growth_12": np.log1p(level_12) - np.log1p(level_12_prev),
                "pax_2019_same_month": pax_2019[:, month - 1],
                "level12_vs_2019": np.log1p(level_12) - np.log1p(_row_mean(pax_2019)),
                "yield_12": _row_mean(YLD[:, o - 11:o + 1]),
                "org_level_12": _row_mean(org_tot[:, o - 11:o + 1]),
                "des_level_12": _row_mean(des_tot[:, o - 11:o + 1]),
                "months_seen_12": np.isfinite(P[:, o - 11:o + 1]).sum(axis=1),
                "distance_mi": distance_mi,
                "domestic": domestic,
                "ORG_AIRPRT": pd.Categorical(org, categories=airport_codes),
                "DES_AIRPRT": pd.Categorical(des, categories=airport_codes),
                **airport_attrs,
                "covid_months_in_history": covid_in_history,
                "network_naive": network[same_col],
                TARGET: P[:, t],
            }))
    out = pd.concat(parts, ignore_index=True)
    keep = out["months_seen_12"] > 0 if future else out[TARGET].notna()
    return out[keep].reset_index(drop=True)


def score(actual, predicted):
    actual, predicted = np.asarray(actual, dtype=float), np.asarray(predicted, dtype=float)
    err = predicted - actual
    return {
        "WAPE": np.abs(err).sum() / actual.sum(),
        "MAE": np.abs(err).mean(),
        "RMSE": np.sqrt((err ** 2).mean()),
        "bias": err.sum() / actual.sum(),
    }


def naive_forecast(df):
    return (df["last_same_month"].fillna(df["level_12"]).fillna(df["pax_2019_same_month"])
            .fillna(df["level_12"].median()))


def route_weight(df):
    """WAPE counts every passenger, so bigger routes get more say in training (square root keeps small routes in play)."""
    level = df["level_12"].fillna(df["level_12"].median())
    return np.sqrt(level).to_numpy()


def reconcile(df, pred):
    """Top-down: scale each forecast month's route forecasts so they add up to the seasonal-naive total for those routes."""
    tmp = pd.DataFrame({"ORIGIN": df["ORIGIN"].to_numpy(), "DATE": df["DATE"].to_numpy(),
                        "pred": np.asarray(pred), "naive": naive_forecast(df).to_numpy()})
    g = tmp.groupby(["ORIGIN", "DATE"])
    return (tmp["pred"] * g["naive"].transform("sum") / g["pred"].transform("sum")).to_numpy()


def _band_keys(df, edges):
    return list(zip(np.where(df["h"] <= 12, 1, 2), np.searchsorted(edges, df["level_12"].fillna(0).to_numpy())))


def interval_bands(val_df, val_pred):
    """Half-width = 80th percentile of |log(actual/forecast)| on validation, by forecast year and route size.

    Symmetric on purpose: validation (2023-24) misses lean upward because traffic was still recovering,
    and reconciliation already removes that growth from the point forecast.
    """
    edges = val_df["level_12"].quantile([1 / 3, 2 / 3]).to_numpy()
    abs_err = np.abs(np.log(val_df[TARGET].to_numpy()) - np.log(val_pred))
    keys = pd.MultiIndex.from_tuples(_band_keys(val_df, edges), names=["year", "size"])
    half = pd.Series(abs_err, index=keys).groupby(level=["year", "size"]).quantile(INTERVAL[1] - INTERVAL[0])
    return half, edges


def apply_bands(df, pred, half, edges):
    h = half.loc[_band_keys(df, edges)].to_numpy()
    return pred * np.exp(-h), pred * np.exp(h)


def fit_model(train, n_trees):
    d = xgb.DMatrix(train[FEATURES], np.log1p(train[TARGET]), weight=route_weight(train), enable_categorical=True)
    return xgb.train(PARAMS, d, num_boost_round=n_trees)


def predict(model, df):
    raw = np.expm1(model.predict(xgb.DMatrix(df[FEATURES], enable_categorical=True)))
    return raw, reconcile(df, raw)


def backtest(pax, airports, train, n_trees):
    rows = {}
    for origin in BACKTEST_ORIGINS:
        tr = train[(train["ORIGIN"] < origin) & (train["DATE"] <= origin)]
        te = build_features(pax, airports, [origin])
        _, pred = predict(fit_model(tr, n_trees), te)
        naive = naive_forecast(te).to_numpy()
        y = te[TARGET].to_numpy()
        year1 = (te["h"] <= 12).to_numpy()
        for label, m in (("year 1", year1), ("year 2", ~year1), ("all 24 months", np.ones(len(te), bool))):
            rows[(f"{origin:%Y-%m}", label)] = {
                "model WAPE": score(y[m], pred[m])["WAPE"], "naive WAPE": score(y[m], naive[m])["WAPE"],
                "model bias": score(y[m], pred[m])["bias"], "training forecast dates": tr["ORIGIN"].nunique(),
            }
    table = pd.DataFrame(rows).T
    table["model wins"] = table["model WAPE"] < table["naive WAPE"]
    print("rolling-origin backtest (each forecast date uses only data up to that date)")
    print(table.round(3).to_string())
    avg = table.groupby(level=1)[["model WAPE", "naive WAPE"]].mean()
    print("\naverage over forecast dates:")
    print(avg.round(3).to_string(), "\n")
    return table


def forecast_future(pax, airports, n_trees, half, edges, repo_root):
    last = pd.to_datetime(dict(year=pax["DEP_YEAR"], month=pax["DEP_MONTH"], day=1)).max()
    origins = pd.date_range(FIRST_TRAIN_ORIGIN, last - pd.DateOffset(months=1), freq="MS")
    train = build_features(pax, airports, origins, max_target=last)
    future = build_features(pax, airports, [last], future=True)
    future["xgb_raw"], future["forecast"] = predict(fit_model(train, n_trees), future)
    future["low"], future["high"] = apply_bands(future, future["forecast"].to_numpy(), half, edges)

    recent = pax[pd.to_datetime(dict(year=pax["DEP_YEAR"], month=pax["DEP_MONTH"], day=1)) > last - pd.DateOffset(months=12)][TARGET].sum()
    year1 = future["h"] <= 12
    print(f"two-year forecast made at {last:%Y-%m} for {future['DATE'].min():%Y-%m} to {future['DATE'].max():%Y-%m}: "
          f"{future['ROUTE'].nunique():,} routes, trained on {len(origins)} forecast dates ({len(train):,} rows)")
    print("forecast total vs the last 12 actual months:",
          {"year 1": f"{future.loc[year1, 'forecast'].sum() / recent - 1:+.1%}",
           "year 2": f"{future.loc[~year1, 'forecast'].sum() / recent - 1:+.1%}"})

    out = repo_root / "team-07-modeling" / "models" / f"xgboost_forecast_{future['DATE'].min():%Y-%m}_to_{future['DATE'].max():%Y-%m}.csv"
    future[["ROUTE", "ORG_AIRPRT", "DES_AIRPRT", "ORIGIN", "DATE", "h", "forecast", "low", "high"]].to_csv(out, index=False)
    print(f"saved to {out.relative_to(repo_root)} (git-ignored)\n")
    return future


def main():
    parser = argparse.ArgumentParser(description="Team 07 XGBoost two-year OD_PAX forecast")
    parser.add_argument("--backtest", action="store_true", help="also re-test from several forecast dates (adds ~3 min)")
    args = parser.parse_args()

    start = time.time()
    repo_root = find_repo_root()
    pax, airports = load_data(repo_root)

    train_origins = pd.date_range(FIRST_TRAIN_ORIGIN, TEST_ORIGIN - pd.DateOffset(months=1), freq="MS")
    train = build_features(pax, airports, train_origins, max_target=TEST_ORIGIN)
    test = build_features(pax, airports, [TEST_ORIGIN])
    print(f"train: {len(train):,} rows, {len(train_origins)} forecast dates, targets "
          f"{train['DATE'].min():%Y-%m} to {train['DATE'].max():%Y-%m}")
    print(f"test:  {len(test):,} rows, one forecast made at {TEST_ORIGIN:%Y-%m} for "
          f"{test['DATE'].min():%Y-%m} to {test['DATE'].max():%Y-%m}")

    fit = train["DATE"] < VAL_START
    val = (train["ORIGIN"] < VAL_START) & (train["DATE"] >= VAL_START)
    weight = route_weight(train)
    d_fit = xgb.DMatrix(train.loc[fit, FEATURES], np.log1p(train.loc[fit, TARGET]), weight=weight[fit], enable_categorical=True)
    d_val = xgb.DMatrix(train.loc[val, FEATURES], np.log1p(train.loc[val, TARGET]), weight=weight[val], enable_categorical=True)
    search = xgb.train(PARAMS, d_fit, num_boost_round=3000, evals=[(d_val, "val")], early_stopping_rounds=100, verbose_eval=False)
    n_trees = search.best_iteration + 1

    val_df = train.loc[val]
    val_pred = reconcile(val_df, np.expm1(search.predict(d_val, iteration_range=(0, n_trees))))
    bands, edges = interval_bands(val_df, val_pred)

    model = xgb.train(PARAMS, xgb.DMatrix(train[FEATURES], np.log1p(train[TARGET]), weight=weight,
                                          enable_categorical=True), num_boost_round=n_trees)
    test["xgb_raw"] = np.expm1(model.predict(xgb.DMatrix(test[FEATURES], enable_categorical=True)))
    test["forecast"] = reconcile(test, test["xgb_raw"])
    test["low"], test["high"] = apply_bands(test, test["forecast"].to_numpy(), bands, edges)
    test["naive"] = naive_forecast(test)
    print(f"trees: {n_trees}, run time so far: {time.time() - start:.0f}s\n")

    year1 = test["h"] <= 12
    rows = {}
    for label, mask in [("all 24 months", slice(None)), ("year 1 (h 1-12)", year1), ("year 2 (h 13-24)", ~year1)]:
        part = test.loc[mask]
        rows[("seasonal naive", label)] = score(part[TARGET], part["naive"])
        rows[("XGBoost raw", label)] = score(part[TARGET], part["xgb_raw"])
        rows[("XGBoost reconciled (final)", label)] = score(part[TARGET], part["forecast"])
    pd.set_option("display.width", 120)
    print("test results")
    print(pd.DataFrame(rows).T.round(3).to_string(), "\n")

    network = test.groupby("DATE")[[TARGET, "naive", "forecast"]].sum()
    miss = network[["naive", "forecast"]].sub(network[TARGET], axis=0).div(network[TARGET], axis=0).abs().mean()
    print("average monthly network-total miss:", miss.round(3).to_dict(), "\n")

    inside = (test[TARGET] >= test["low"]) & (test[TARGET] <= test["high"])
    width = ((test["high"] - test["low"]) / test["forecast"])
    print(f"{int((INTERVAL[1] - INTERVAL[0]) * 100)}% interval coverage on test:",
          {"year 1": round(float(inside[year1].mean()), 3), "year 2": round(float(inside[~year1].mean()), 3),
           "all": round(float(inside.mean()), 3)},
          "| median width as share of forecast:", round(float(width.median()), 3), "\n")

    importance = pd.Series(model.get_score(importance_type="gain")).sort_values(ascending=False)
    print("feature importance (gain):")
    print(importance.round(1).to_string(), "\n")

    out = repo_root / "team-07-modeling" / "models" / "xgboost_test_forecast.csv"
    test[["ROUTE", "ORIGIN", "DATE", "h", TARGET, "naive", "forecast", "low", "high"]].to_csv(out, index=False)
    print(f"saved test forecast to {out.relative_to(repo_root)} (git-ignored)\n")

    if args.backtest:
        backtest(pax, airports, train, n_trees)

    future_half, future_edges = interval_bands(test, test["forecast"].to_numpy())
    forecast_future(pax, airports, n_trees, future_half, future_edges, repo_root)
    print(f"total time: {time.time() - start:.0f}s")


if __name__ == "__main__":
    main()
