"""Macroeconomic, fuel, and exchange-rate series from FRED (no API key needed)."""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

from .http import download

log = logging.getLogger(__name__)
FRED_CSV = "https://fred.stlouisfed.org/graph/fredgraph.csv?id={sid}&cosd=2015-01-01"

MACRO_SERIES = {
    "MJFUELUSGULF": "jet_fuel",  # US Gulf Coast kerosene-type jet fuel, $/gal
    "MCOILBRENTEU": "brent",  # Brent crude, $/bbl
    "UNRATE": "unemployment",
    "UMCSENT": "consumer_sentiment",
    "CUSR0000SETG01": "cpi_airfare",
    "CPIAUCSL": "cpi",
    "DSPIC96": "real_disp_income",
}

# FRED id -> (currency, quoted as USD per unit). All are stored as units per USD.
FX_SERIES = {
    "DEXMXUS": ("MXN", False),
    "DEXCAUS": ("CAD", False),
    "DEXUSEU": ("EUR", True),
    "DEXUSUK": ("GBP", True),
    "DEXJPUS": ("JPY", False),
    "DEXKOUS": ("KRW", False),
    "DEXCHUS": ("CNY", False),
    "DEXHKUS": ("HKD", False),
    "DEXTAUS": ("TWD", False),
    "DEXINUS": ("INR", False),
    "DEXBZUS": ("BRL", False),
    "DEXSZUS": ("CHF", False),
}

COUNTRY_CURRENCY = {
    "MX": "MXN", "CA": "CAD", "GB": "GBP", "JP": "JPY", "KR": "KRW", "CN": "CNY",
    "HK": "HKD", "TW": "TWD", "IN": "INR", "BR": "BRL", "CH": "CHF",
    **{c: "EUR" for c in ("IT", "DE", "ES", "IE", "PT", "GR", "NL", "FR", "AT", "BE", "FI")},
}


def _read(path: Path, sid: str) -> pd.Series:
    df = pd.read_csv(path)
    date_col = df.columns[0]
    s = pd.to_numeric(df[sid], errors="coerce")
    s.index = pd.to_datetime(df[date_col])
    return s.dropna()


def fetch(dest_dir: Path, force: bool = False) -> pd.DataFrame:
    """Download all series and return a monthly table indexed by month start."""
    cols = {}
    for sid, name in MACRO_SERIES.items():
        try:
            path = download(FRED_CSV.format(sid=sid), dest_dir / f"{sid}.csv", force=force)
            cols[name] = _read(path, sid).resample("MS").mean()
        except Exception as exc:  # keep going; the model tolerates missing features
            log.warning("FRED %s unavailable: %s", sid, exc)
    for sid, (ccy, invert) in FX_SERIES.items():
        try:
            path = download(FRED_CSV.format(sid=sid), dest_dir / f"{sid}.csv", force=force)
            s = _read(path, sid).resample("MS").mean()
            cols[f"fx_{ccy}"] = 1.0 / s if invert else s
        except Exception as exc:
            log.warning("FRED %s unavailable: %s", sid, exc)
    out = pd.DataFrame(cols).sort_index()
    out.index.name = "month"
    return out.replace([np.inf, -np.inf], np.nan)
