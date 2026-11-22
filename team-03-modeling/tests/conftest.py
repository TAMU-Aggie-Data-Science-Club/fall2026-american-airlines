"""Synthetic fixtures. Tests never read the proprietary AA files."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

AIRPORTS = {
    # code: (metro, country, lat, lon)
    "DFW": ("DFW", "US", 32.90, -97.04),
    "DAL": ("DFW", "US", 32.85, -96.85),
    "LAX": ("LAX", "US", 33.94, -118.41),
    "ORD": ("CHI", "US", 41.98, -87.90),
    "MIA": ("MIA", "US", 25.79, -80.29),
    "CUN": ("CUN", "MX", 21.04, -86.87),
}


@pytest.fixture
def airports() -> pd.DataFrame:
    rows = [
        {"airport": k, "name": k, "metro": v[0], "state": None, "country": v[1], "world_area": 1,
         "lat": v[2], "lon": v[3], "runway_ft": 10000, "elevation_ft": 100}
        for k, v in AIRPORTS.items()
    ]
    return pd.DataFrame(rows).set_index("airport")


@pytest.fixture
def pax_long() -> pd.DataFrame:
    """Four years of monthly demand with seasonality and market-specific growth."""
    rng = np.random.default_rng(0)
    months = pd.date_range("2019-01-01", "2023-12-01", freq="MS")
    codes = list(AIRPORTS)
    rows = []
    for o in codes:
        for d in codes:
            if o == d:
                continue
            level = rng.uniform(200, 5000)
            growth = rng.normal(0.02, 0.05)
            for i, m in enumerate(months):
                season = 1 + 0.25 * np.sin(2 * np.pi * (m.month - 3) / 12)
                pax = level * season * np.exp(growth * i / 12) * rng.lognormal(0, 0.05)
                rows.append({"month": m, "org": o, "des": d, "od_pax": pax, "od_rev_usd": pax * rng.uniform(150, 400)})
    return pd.DataFrame(rows)


@pytest.fixture
def raw_pax(pax_long) -> pd.DataFrame:
    df = pax_long.copy()
    out = pd.DataFrame({
        "month_num": df["month"].dt.month.astype("int16"),
        "year": df["month"].dt.year.astype("int16"),
        "org": df["org"],
        "des": df["des"],
        "od_pax": df["od_pax"],
        "od_rev_usd": df["od_rev_usd"],
        "load_ts": "2026-01-01 00:00:00",
    })
    return out
