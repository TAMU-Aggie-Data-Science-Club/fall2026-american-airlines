"""Driver bridge: why a market's forecast differs from its seasonal anchor.

The learned market mix f(x) is split into the driver families in
features.DRIVER_GROUPS with exact group Shapley values. A family that is
"switched off" takes its training median, so each contribution is measured
against a typical market. The pieces add up exactly, in log points:

    log(forecast / anchor) = shrink x (typical + sum of driver contributions)
                             + network outlook and rescaling
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd

from .features import DRIVER_GROUPS, driver_group
from .model import AnchorAdjustModel


def group_shapley(model: AnchorAdjustModel, rows: pd.DataFrame, shrink: float, chunk: int = 400_000) -> pd.DataFrame:
    feats = model.features
    groups = [g for g in DRIVER_GROUPS if any(driver_group(f) == g for f in feats)]
    G = len(groups)
    cols_by_group = {g: [i for i, f in enumerate(feats) if driver_group(f) == g] for g in groups}
    X = rows[feats].to_numpy(np.float32)
    bg = model.background[feats].to_numpy(np.float32)
    n = len(X)

    # Value of every coalition (bitmask over groups).
    values = np.empty((1 << G, n), dtype=np.float64)
    for mask in range(1 << G):
        Xs = np.broadcast_to(bg, X.shape).copy()
        for gi, g in enumerate(groups):
            if mask >> gi & 1:
                Xs[:, cols_by_group[g]] = X[:, cols_by_group[g]]
        preds = [model.model.predict(pd.DataFrame(Xs[s : s + chunk], columns=feats)) for s in range(0, n, chunk)]
        values[mask] = np.concatenate(preds)

    phi = np.zeros((n, G))
    weights = [math.factorial(k) * math.factorial(G - k - 1) / math.factorial(G) for k in range(G)]
    for gi in range(G):
        bit = 1 << gi
        for mask in range(1 << G):
            if mask & bit:
                continue
            k = bin(mask).count("1")
            phi[:, gi] += weights[k] * (values[mask | bit] - values[mask])
    out = pd.DataFrame(shrink * phi, columns=[f"drv_{g}" for g in groups], index=rows.index)
    out.insert(0, "drv_typical", shrink * values[0])
    total = np.log(rows["forecast"].clip(lower=1e-9)) - np.log(rows["anchor"].clip(lower=1e-9))
    out["drv_network"] = total.to_numpy() - out.sum(axis=1).to_numpy()
    return out


def aggregate_bridge(bridge: pd.DataFrame, weights: pd.Series) -> pd.Series:
    """Volume-weighted average contribution in log points (about equal to percent)."""
    w = weights / weights.sum()
    return bridge.mul(w, axis=0).sum()


def label(col: str) -> str:
    key = col.replace("drv_", "")
    special = {"typical": "Typical market", "network": "Network outlook and rescaling"}
    return special.get(key, DRIVER_GROUPS.get(key, key))
