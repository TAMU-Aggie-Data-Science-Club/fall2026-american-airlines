"""Anchor and Adjust: seasonal anchor, network outlook, and market mix.

For market m, origin month t, and horizon h (12 to 18 months):

    anchor_m   = trailing 12-month level x the market's own seasonal shape
    network    = sum of anchors x (1 + network growth) ^ (h / 12)
    mix_m      = shrink x f(features known at t)        # gradient-boosted trees
    forecast_m = anchor_m x exp(mix_m), rescaled so markets add up to the network

Three layers, each with a plain-language meaning for network planning:

1. The anchor is what the market looks like if nothing changes.
2. The network outlook is a single growth number. It defaults to flat because
   network-wide swings 12 to 18 months out are not reliably predictable from
   seven years of history, and planners usually hold a corporate view anyway.
   It can be overridden with that view.
3. The market mix is where the machine learning lives. One global model learns
   which markets will outgrow or lag the network, using momentum, catchment
   demographics, migration ties, fares, and how each market responds to fuel,
   the economy, and exchange rates. Training targets are demeaned within each
   origin and horizon, so the model learns relative movement and is not thrown
   off by the post-pandemic rebound that inflated every market at once.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

from .config import RunConfig
from .features import feature_columns


def training_mask(df: pd.DataFrame, cfg: RunConfig) -> np.ndarray:
    """Rows usable for training: target observed, outside the pandemic, enough volume."""
    return (df["y"].notna() & (df["target_pandemic"] == 0) & (df["ttm"] >= cfg.min_ttm)).to_numpy()


def log_adjustment(y, anchor) -> np.ndarray:
    return np.log1p(np.asarray(y, float)) - np.log1p(np.asarray(anchor, float))


def relative_adjustment(df: pd.DataFrame) -> np.ndarray:
    """Market log growth versus its anchor, minus the volume-weighted average for that origin and horizon."""
    z = pd.Series(log_adjustment(df["y"], df["anchor"]), index=df.index)
    w = df["anchor"].clip(lower=1.0)
    keys = [df["origin_idx"], df["horizon"]]
    mean = (z * w).groupby(keys).transform("sum") / w.groupby(keys).transform("sum")
    return (z - mean).to_numpy()


def network_scale(df: pd.DataFrame, values: np.ndarray, growth: float) -> np.ndarray:
    """Rescale ``values`` so each (origin, horizon) slice adds up to the network outlook."""
    v = pd.Series(values, index=df.index)
    keys = [df["origin_idx"], df["horizon"]]
    target = df["anchor"].groupby(keys).transform("sum") * (1 + growth) ** (df["horizon"] / 12.0)
    return (v * target / v.groupby(keys).transform("sum")).to_numpy()


class AnchorAdjustModel:
    def __init__(self, cfg: RunConfig):
        self.cfg = cfg
        self.features: list[str] = []
        self.model: HistGradientBoostingRegressor | None = None
        self.background: pd.Series | None = None

    def fit(self, df: pd.DataFrame) -> "AnchorAdjustModel":
        rows = df[training_mask(df, self.cfg)]
        if rows.empty:
            raise ValueError("No training rows")
        # Columns with no values (for example external data that was not fetched) are skipped.
        self.features = [c for c in feature_columns(rows) if rows[c].notna().any()]
        X = rows[self.features]
        z = relative_adjustment(rows)
        # Square-root volume weights: big markets matter more without drowning small ones.
        w = np.sqrt(rows["ttm"].to_numpy(float))
        self.model = HistGradientBoostingRegressor(loss="squared_error", random_state=self.cfg.seed, **self.cfg.gbm_params)
        self.model.fit(X, z, sample_weight=w / w.mean())
        self.background = X.median()
        return self

    def raw_mix(self, df: pd.DataFrame) -> np.ndarray:
        """Unshrunk relative log adjustment from the trees."""
        return self.model.predict(df[self.features])

    def predict(self, df: pd.DataFrame, shrink: float | None = None, growth: float | None = None) -> np.ndarray:
        shrink = self.cfg.shrink if shrink is None else shrink
        growth = self.cfg.network_growth if growth is None else growth
        return combine(df, self.raw_mix(df), shrink, growth)


def combine(df: pd.DataFrame, raw: np.ndarray, shrink: float, growth: float) -> np.ndarray:
    """Anchor x shrunk market mix, rescaled to the network outlook."""
    f = df["anchor"].to_numpy(float) * np.exp(shrink * np.asarray(raw, float))
    return np.clip(network_scale(df, f, growth), 0, None)


def baselines(df: pd.DataFrame) -> dict[str, np.ndarray]:
    """Planner-style reference forecasts computed from the same rows."""
    last = df["last_same_month"].to_numpy(float)
    growth = np.nan_to_num(df["mom_growth_ttm"].to_numpy(float), nan=0.0)
    return {
        "seasonal_anchor": df["anchor"].to_numpy(float),
        "same_month_last_year": last,
        "same_month_x_trend": last * np.exp(np.clip(growth, -1, 1)),
    }
