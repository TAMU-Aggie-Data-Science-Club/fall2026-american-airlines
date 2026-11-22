"""Statistical diagnostics that shape the modeling choices."""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.tsa.seasonal import STL
from statsmodels.tsa.stattools import adfuller, kpss


def stationarity(series: pd.Series, name: str) -> list[dict]:
    """ADF (null: unit root) and KPSS (null: stationary) on a log series and its seasonal difference."""
    out = []
    s = np.log(series.clip(lower=1)).dropna()
    for label, x in ((name, s), (f"{name}, 12-month difference", s.diff(12).dropna())):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            adf_p = adfuller(x, autolag="AIC")[1]
            kpss_p = kpss(x, regression="c", nlags="auto")[1]
        out.append({"series": label, "adf_p": adf_p, "kpss_p": kpss_p,
                    "reading": "stationary" if adf_p < 0.05 and kpss_p > 0.05 else "non-stationary or mixed"})
    return out


def seasonal_strength(series: pd.Series) -> tuple[float, float]:
    """Hyndman's STL-based trend and seasonal strength, both in [0, 1]."""
    s = np.log1p(series)
    res = STL(s, period=12, robust=True).fit()
    var_r = np.var(res.resid)
    trend = max(0.0, 1 - var_r / np.var(res.trend + res.resid))
    seas = max(0.0, 1 - var_r / np.var(res.seasonal + res.resid))
    return trend, seas


def chow_test(series: pd.Series, break_date: str) -> dict:
    """Chow test for a break in a log-linear trend with monthly dummies."""
    y = np.log(series.clip(lower=1)).to_numpy()
    idx = series.index
    t = np.arange(len(y), dtype=float)
    month = pd.get_dummies(idx.month, drop_first=True).to_numpy(float)
    X = np.column_stack([np.ones_like(t), t, month])
    split = int(np.searchsorted(idx, pd.Timestamp(break_date)))

    def rss(Xp, yp):
        beta, *_ = np.linalg.lstsq(Xp, yp, rcond=None)
        return float(((yp - Xp @ beta) ** 2).sum())

    k = X.shape[1]
    r_all = rss(X, y)
    r1, r2 = rss(X[:split], y[:split]), rss(X[split:], y[split:])
    df2 = len(y) - 2 * k
    f = ((r_all - (r1 + r2)) / k) / ((r1 + r2) / df2)
    return {"break": break_date, "F": f, "p_value": float(1 - stats.f.cdf(f, k, df2))}


def market_seasonality(pax: np.ndarray, months: pd.DatetimeIndex, top: np.ndarray, start: str = "2022-01-01") -> pd.DataFrame:
    """Trend and seasonal strength for the largest markets since the recovery."""
    sel = months >= pd.Timestamp(start)
    rows = []
    for i in top:
        s = pd.Series(pax[i, sel], index=months[sel])
        if (s > 0).sum() < 36:
            continue
        tr, se = seasonal_strength(s)
        rows.append({"market_id": i, "trend_strength": tr, "seasonal_strength": se})
    return pd.DataFrame(rows)
