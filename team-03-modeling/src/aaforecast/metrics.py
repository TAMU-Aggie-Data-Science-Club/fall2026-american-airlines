"""Accuracy metrics framed for planning use."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats


def wape(y, f) -> float:
    """Weighted absolute percentage error: total absolute miss over total actual demand."""
    y, f = np.asarray(y, float), np.asarray(f, float)
    denom = np.abs(y).sum()
    return float(np.abs(y - f).sum() / denom) if denom > 0 else np.nan


def bias(y, f) -> float:
    """Signed total error as a share of actual demand (positive means over-forecast)."""
    y, f = np.asarray(y, float), np.asarray(f, float)
    denom = y.sum()
    return float((f - y).sum() / denom) if denom > 0 else np.nan


def coverage(y, lo, hi) -> float:
    y = np.asarray(y, float)
    return float(np.mean((y >= lo) & (y <= hi)))


def summarize(df: pd.DataFrame, methods: list[str], by: list[str] | None = None) -> pd.DataFrame:
    """WAPE and bias for each method, optionally grouped."""

    def one(g):
        out = {}
        if not np.isfinite(g[["y", *methods]].to_numpy(dtype=float)).all():
            raise ValueError("Metrics require finite actuals and predictions on identical cases")
        for m in methods:
            out[f"wape_{m}"] = wape(g["y"], g[m])
            out[f"bias_{m}"] = bias(g["y"], g[m])
            error = g[m].to_numpy() - g["y"].to_numpy()
            out[f"mae_{m}"] = np.abs(error).mean() if len(error) else np.nan
            out[f"rmse_{m}"] = np.sqrt(np.mean(error ** 2)) if len(error) else np.nan
            out[f"mean_error_{m}"] = error.mean() if len(error) else np.nan
        out["actual"] = g["y"].sum()
        out["rows"] = len(g)
        return pd.Series(out)

    if not by:
        return one(df).to_frame().T
    return df.groupby(by, observed=True).apply(one, include_groups=False).reset_index()


def paired_test(df: pd.DataFrame, model: str, reference: str, unit: str = "market_id", n_boot: int = 1000, seed: int = 0) -> dict:
    """Is the model's error lower than the reference's, market by market?

    Uses absolute errors summed within each market, a Wilcoxon signed-rank test,
    and a bootstrap over markets for the WAPE gap.
    """
    y = df.groupby(unit)["y"].sum()
    err_m = (df[model] - df["y"]).abs().groupby(df[unit]).sum()
    err_r = (df[reference] - df["y"]).abs().groupby(df[unit]).sum()
    diff = (err_m - err_r).to_numpy()
    stat = stats.wilcoxon(diff, alternative="less", zero_method="zsplit")
    rng = np.random.default_rng(seed)
    idx = np.arange(len(diff))
    gaps = []
    em, er, yy = err_m.to_numpy(), err_r.to_numpy(), y.to_numpy()
    for _ in range(n_boot):
        s = rng.choice(idx, size=len(idx), replace=True)
        gaps.append(em[s].sum() / yy[s].sum() - er[s].sum() / yy[s].sum())
    lo, hi = np.percentile(gaps, [2.5, 97.5])
    return {
        "model": model,
        "reference": reference,
        "wape_gap": float(em.sum() / yy.sum() - er.sum() / yy.sum()),
        "gap_ci_low": float(lo),
        "gap_ci_high": float(hi),
        "share_markets_improved": float(np.mean(diff < 0)),
        "wilcoxon_p": float(stat.pvalue),
    }
