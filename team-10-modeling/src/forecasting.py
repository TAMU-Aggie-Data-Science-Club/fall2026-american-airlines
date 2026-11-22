"""Team 10 two-year forecasting: origins, forecast sets, candidate models, scoring, decision rule.

Design (Deliverables 4-5)
-------------------------
* A forecast is made once at an origin T for every route-month in T .. T+23 and is never
  refreshed with actuals from inside that window.
* Forecast set at T: every route seen in the 12 months before T, for each of the 24 months.
* Scoring: forecasts are outer-joined to the actual rows of the window. An actual with no forecast
  counts as forecast 0 (a route we did not plan for). A forecast with no actual counts as actual 0.
* Origins (computed from the data, never hand-picked after seeing scores):
    test       = the last 24 months of data            -> D5 result
    select     = test - 12                              -> D4 main comparison
    stability  = test - 24 (window ends just before the test window) -> D4 check
  The select window overlaps the test window by 12 months. An origin early enough to avoid that
  would put the training history inside the 2020-21 disruption, leaving fitted models nothing
  clean to learn from. To keep the overlap from steering the result, every model setting is fixed
  in this file, and the carry rule (``carried_model``) was written before any score existed.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import RidgeCV
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from features import (GROWTH_CAP, KEY, MODEL_CATEGORICAL, MODEL_NUMERIC, model_inputs,
                      shift_months, training_pairs)

H = 24
SEED = 10
MIN_TRAIN_PAIRS = 2000  # below this a growth model is "not fittable at this origin"


class InsufficientHistory(RuntimeError):
    """Too few clean year-over-year pairs before the origin to fit a growth model."""


# --------------------------------------------------------------------------- origins & sets
@dataclass(frozen=True)
class Origins:
    test: pd.Timestamp
    select: pd.Timestamp
    stability: pd.Timestamp
    forward: pd.Timestamp


def origins(pax: pd.DataFrame) -> Origins:
    last = pax["DATE"].max()
    test = last - pd.DateOffset(months=H - 1)
    return Origins(test=test, select=test - pd.DateOffset(months=12),
                   stability=test - pd.DateOffset(months=24),
                   forward=last + pd.DateOffset(months=1))


def forecast_set(pax: pd.DataFrame, origin: pd.Timestamp) -> pd.DataFrame:
    """Routes active in the 12 months before ``origin`` x the 24 months from ``origin``."""
    recent = pax[(pax["DATE"] < origin) & (pax["DATE"] >= origin - pd.DateOffset(months=12))]
    routes = recent[KEY].drop_duplicates()
    months = pd.DataFrame({"DATE": pd.date_range(origin, periods=H, freq="MS")})
    months["h"] = np.arange(1, H + 1)
    months["h_year"] = np.where(months["h"] <= 12, 1, 2)
    return routes.merge(months, how="cross")


# --------------------------------------------------------------------------- candidates
def seasonal_naive(pax: pd.DataFrame, fs: pd.DataFrame, origin: pd.Timestamp) -> np.ndarray:
    """Same route, same calendar month, in the year before ``origin``; used for both years.
    A month the route did not fly falls back to the route's median month in that year."""
    hist = pax[(pax["DATE"] < origin) & (pax["DATE"] >= origin - pd.DateOffset(months=12))]
    src = fs[[*KEY]].copy()
    src["DATE"] = shift_months(fs["DATE"], -12 * fs["h_year"])
    val = src.merge(hist[[*KEY, "DATE", "OD_PAX"]], on=[*KEY, "DATE"], how="left")["OD_PAX"]
    med = fs[KEY].merge(hist.groupby(KEY, as_index=False)["OD_PAX"].median(),
                        on=KEY, how="left")["OD_PAX"]
    return val.fillna(med).fillna(0).to_numpy()


def trend_naive(pax: pd.DataFrame, fs: pd.DataFrame, origin: pd.Timestamp) -> np.ndarray:
    """Seasonal naive x a damped growth rate: year 1 x g, year 2 x g^2.

    g blends the route's last-12 / prior-12 passenger ratio with the network ratio; the route's
    weight grows with how many months it flew in the prior year (full weight 0.5 at 12 months).
    g is capped to GROWTH_CAP.
    """
    base = seasonal_naive(pax, fs, origin)
    last = pax[(pax["DATE"] < origin) & (pax["DATE"] >= origin - pd.DateOffset(months=12))]
    prev = pax[(pax["DATE"] < origin - pd.DateOffset(months=12))
               & (pax["DATE"] >= origin - pd.DateOffset(months=24))]
    s_last, s_prev = last.groupby(KEY)["OD_PAX"].sum(), prev.groupby(KEY)["OD_PAX"].sum()
    n_prev = prev.groupby(KEY)["OD_PAX"].size()
    g_net = s_last.sum() / max(s_prev.sum(), 1.0)
    g_route = (s_last / s_prev).reindex(s_last.index)
    w = (0.5 * n_prev.reindex(s_last.index).fillna(0) / 12).clip(0, 0.5)
    g = np.exp((1 - w) * np.log(g_net) + w * np.log(g_route.clip(*GROWTH_CAP).fillna(g_net)))
    g = np.clip(fs[KEY].merge(g.rename("g").reset_index(), on=KEY, how="left")["g"].fillna(g_net),
                *GROWTH_CAP)
    return base * g.to_numpy() ** fs["h_year"].to_numpy()


def _ridge():
    pre = ColumnTransformer([
        ("num", make_pipeline(SimpleImputer(strategy="median"), StandardScaler()), MODEL_NUMERIC),
        ("cat", OneHotEncoder(handle_unknown="ignore", min_frequency=5), MODEL_CATEGORICAL),
    ])
    return make_pipeline(pre, RidgeCV(alphas=np.logspace(-2, 3, 11)))


def _gbt():
    return HistGradientBoostingRegressor(
        loss="squared_error", learning_rate=0.05, max_iter=400, max_leaf_nodes=31,
        min_samples_leaf=40, l2_regularization=1.0, early_stopping=True,
        validation_fraction=0.1, categorical_features="from_dtype", random_state=SEED)


GBT_COLS = [c for c in MODEL_NUMERIC + MODEL_CATEGORICAL if c != "month"]


def _as_categories(X: pd.DataFrame, cats: dict | None = None) -> tuple[pd.DataFrame, dict]:
    X = X.copy()
    cats = cats or {c: sorted(X[c].dropna().astype(str).unique()) for c in ("ORG_AIRPRT", "DES_AIRPRT")}
    for c, levels in cats.items():
        X[c] = pd.Categorical(X[c].astype(str), categories=levels)
    return X, cats


def fit_growth_model(name: str, pax: pd.DataFrame, routes: pd.DataFrame, origin: pd.Timestamp):
    """Fit a year-over-year log-growth model on clean pairs before ``origin``.
    Returns a function rows -> predicted log growth."""
    train = training_pairs(pax, routes, origin)
    if len(train) < MIN_TRAIN_PAIRS:
        raise InsufficientHistory(f"{name}: {len(train)} clean pairs before {origin.date()}")
    if name == "Ridge growth":
        model = _ridge().fit(train[MODEL_NUMERIC + MODEL_CATEGORICAL], train["log_growth"])
        return lambda rows: model.predict(rows[MODEL_NUMERIC + MODEL_CATEGORICAL])
    if name == "Gradient-boosted growth":
        Xtr, cats = _as_categories(train[GBT_COLS])
        model = _gbt().fit(Xtr, train["log_growth"])
        return lambda rows: model.predict(_as_categories(rows[GBT_COLS], cats)[0])
    raise ValueError(name)


def fit_predict(name: str, pax: pd.DataFrame, routes: pd.DataFrame, fs: pd.DataFrame,
                origin: pd.Timestamp) -> np.ndarray:
    """Fit candidate ``name`` on history before ``origin`` and predict the forecast set."""
    naive = seasonal_naive(pax, fs, origin)
    if name == "Seasonal naive":
        return naive
    if name == "Trend-adjusted naive":
        return trend_naive(pax, fs, origin)
    predict = fit_growth_model(name, pax[pax["DATE"] < origin], routes, origin)
    rows = model_inputs(fs.assign(anchor=naive), routes)
    growth = np.clip(predict(rows), np.log(GROWTH_CAP[0]), np.log(GROWTH_CAP[1]))
    return np.clip(naive * np.exp(growth * fs["h_year"].to_numpy()), 0, None)


CANDIDATES = ["Seasonal naive", "Trend-adjusted naive", "Ridge growth",
              "Gradient-boosted growth"]


# --------------------------------------------------------------------------- scoring
def scored_frame(pax: pd.DataFrame, fs: pd.DataFrame, pred: np.ndarray,
                 origin: pd.Timestamp) -> pd.DataFrame:
    """Outer join of forecasts and actual rows in the window, zero-filled."""
    f = fs[[*KEY, "DATE"]].assign(forecast=pred)
    end = origin + pd.DateOffset(months=H - 1)
    act = pax[(pax["DATE"] >= origin) & (pax["DATE"] <= end)][[*KEY, "DATE", "OD_PAX"]]
    out = f.merge(act, on=[*KEY, "DATE"], how="outer").fillna({"forecast": 0.0, "OD_PAX": 0.0})
    out["h"] = (out["DATE"].dt.year - origin.year) * 12 + out["DATE"].dt.month - origin.month + 1
    out["h_year"] = np.where(out["h"] <= 12, 1, 2)
    return out


def wape(a, f) -> float:
    a, f = np.asarray(a, float), np.asarray(f, float)
    return float(np.abs(a - f).sum() / max(a.sum(), 1e-9))


def summary(s: pd.DataFrame) -> dict:
    net = s.groupby("DATE")[["OD_PAX", "forecast"]].sum()
    y1, y2 = s[s["h_year"] == 1], s[s["h_year"] == 2]
    return {
        "WAPE_24m": wape(s["OD_PAX"], s["forecast"]),
        "WAPE_y1": wape(y1["OD_PAX"], y1["forecast"]),
        "WAPE_y2": wape(y2["OD_PAX"], y2["forecast"]),
        "bias_24m": float(s["forecast"].sum() / s["OD_PAX"].sum() - 1),
        "bias_y1": float(y1["forecast"].sum() / y1["OD_PAX"].sum() - 1),
        "bias_y2": float(y2["forecast"].sum() / y2["OD_PAX"].sum() - 1),
        "network_WAPE": wape(net["OD_PAX"], net["forecast"]),
        "route_MAE": float(np.abs(s["OD_PAX"] - s["forecast"]).mean()),
        "route_RMSE": float(np.sqrt(((s["OD_PAX"] - s["forecast"]) ** 2).mean())),
    }


def by_horizon(s: pd.DataFrame) -> pd.Series:
    return s.groupby("h").apply(lambda g: wape(g["OD_PAX"], g["forecast"]), include_groups=False)


def by_segment(s: pd.DataFrame, pax: pd.DataFrame, routes: pd.DataFrame,
               origin: pd.Timestamp) -> pd.DataFrame:
    """WAPE by route-volume quartile (from the year before origin) and domestic/international."""
    prior = pax[(pax["DATE"] < origin) & (pax["DATE"] >= origin - pd.DateOffset(months=12))]
    vol = prior.groupby(KEY)["OD_PAX"].sum()
    q = pd.qcut(vol, 4, labels=["Q1 smallest", "Q2", "Q3", "Q4 largest"]).rename("volume")
    t = s.merge(q.reset_index(), on=KEY, how="left").merge(
        routes[[*KEY, "is_international"]], on=KEY, how="left")
    t["volume"] = t["volume"].astype("string").fillna("new route")
    t["market"] = np.where(t["is_international"] == 1, "international", "domestic")
    rows = []
    for col in ("volume", "market"):
        for lvl, g in t.groupby(col):
            for k in (1, 2):
                gk = g[g["h_year"] == k]
                rows.append({"segment": col, "level": lvl, "year": k,
                             "share_of_pax": gk["OD_PAX"].sum() / t.loc[t["h_year"] == k, "OD_PAX"].sum(),
                             "WAPE": wape(gk["OD_PAX"], gk["forecast"])})
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- decision rule
def carried_model(select: dict, stability: dict) -> tuple[str, list[str]]:
    """Pre-registered rule (written in results/baseline-decision.md before any score was seen).

    A fitted candidate replaces the seasonal naive only if, at the SELECT origin, its route-month
    WAPE is lower in year 1 AND in year 2, and at the STABILITY origin its 24-month WAPE is no
    worse than the naive's. Among candidates that pass, the lowest SELECT 24-month WAPE wins.
    Otherwise the seasonal naive is carried. A candidate that could not be fitted at an origin
    (``None`` in the results) fails every check at that origin.
    """
    base_s, base_t = select["Seasonal naive"], stability["Seasonal naive"]
    notes, passed = [], []
    for name in CANDIDATES[1:]:
        s, t = select.get(name), stability.get(name)
        checks = {"select y1": s is not None and s["WAPE_y1"] < base_s["WAPE_y1"],
                  "select y2": s is not None and s["WAPE_y2"] < base_s["WAPE_y2"],
                  "stability 24m": t is not None and t["WAPE_24m"] <= base_t["WAPE_24m"]}
        ok = all(checks.values())
        notes.append(f"{name}: " + ", ".join(f"{k} {'pass' if v else 'fail'}" for k, v in checks.items()))
        if ok:
            passed.append(name)
    if not passed:
        return "Seasonal naive", notes
    return min(passed, key=lambda n: select[n]["WAPE_24m"]), notes
