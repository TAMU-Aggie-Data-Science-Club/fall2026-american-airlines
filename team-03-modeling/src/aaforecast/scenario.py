"""What-if scenarios on the external drivers.

Each scenario shifts the economy, fuel, or exchange-rate features at the
forecast origin and re-scores the same markets. Because the network total is
set by the network outlook, a scenario shows how demand would shift between
markets: which hubs and segments are most exposed. Results are sensitivities
learned from 2019 onward, not causal estimates.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .config import AA_HUBS
from .model import AnchorAdjustModel

SCENARIOS = {
    "Jet fuel +40%": {
        "eco_jet_fuel_log": np.log(1.4),
        "eco_jet_fuel_chg12": np.log(1.4),
        "eco_brent_log": np.log(1.4),
        "eco_brent_chg12": np.log(1.4),
    },
    "Mild US recession": {
        "eco_unemployment": 2.0,
        "eco_unemployment_chg12": 2.0,
        "eco_sentiment": -15.0,
        "eco_sentiment_chg12": -15.0,
        "eco_real_disp_income_yoy": -0.02,
    },
    "Dollar 10% stronger": {"fx_foreign_chg12": 0.10},
    "Dollar 10% weaker": {"fx_foreign_chg12": -0.10},
}


def run_scenarios(model: AnchorAdjustModel, rows: pd.DataFrame, base_forecast: np.ndarray, shrink: float) -> pd.DataFrame:
    """Percent change versus the base forecast by segment and AA hub."""
    seg = np.where(rows["mkt_domestic"].to_numpy() == 1, "Domestic", "International")
    org, des = rows["org"].to_numpy(), rows["des"].to_numpy()
    out = []
    for name, shocks in SCENARIOS.items():
        if not any(col in model.features for col in shocks):
            continue  # No fitted shock feature: unavailable, not evidence of zero sensitivity.
        shocked = rows.copy()
        for col, delta in shocks.items():
            if col in shocked:
                shocked[col] = shocked[col] + delta
        f = model.predict(shocked, shrink=shrink)
        groups = [("Domestic", seg == "Domestic"), ("International", seg == "International")]
        groups += [(f"Hub {h}", (org == h) | (des == h)) for h in sorted(AA_HUBS)]
        for label, mask in groups:
            if not mask.any():
                continue
            b, s = base_forecast[mask].sum(), f[mask].sum()
            out.append({"scenario": name, "group": label, "base_total": b, "scenario_total": s, "change_pct": s / b - 1})
    return pd.DataFrame(out, columns=["scenario", "group", "base_total", "scenario_total", "change_pct"])
