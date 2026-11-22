"""Load, clean, and reshape the AA passenger and airport datasets."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from . import config

log = logging.getLogger(__name__)

PAX_COLUMNS = {
    "DEP_MONTH": "month_num",
    "DEP_YEAR": "year",
    "ORG_AIRPRT": "org",
    "DES_AIRPRT": "des",
    "OD_PAX": "od_pax",
    "OD_REV_USD": "od_rev_usd",
    "SF_LOAD_TMS": "load_ts",
}

AIRPORT_COLUMNS = {
    "AIRPRT_CD": "airport",
    "AIRPRT_NM": "name",
    "CITY_METRO_IATA_CD": "metro",
    "STATE_PROVNC_CD": "state",
    "CNTRY_CD": "country",
    "WRLD_AREA_DOT_CD": "world_area",
    "LNGST_RUNWAY_FT_QTY": "runway_ft",
    "ELEVATN_FT_QTY": "elevation_ft",
}


def read_pax_raw(path: Path | str = config.PAX_FILE) -> pd.DataFrame:
    """Read the raw passenger file with explicit types."""
    dtypes = {
        "DEP_MONTH": "int16",
        "DEP_YEAR": "int16",
        "ORG_AIRPRT": "string",
        "DES_AIRPRT": "string",
        "OD_PAX": "float64",
        "OD_REV_USD": "float64",
        "SF_LOAD_TMS": "string",
    }
    df = pd.read_csv(path, dtype=dtypes, keep_default_na=False, na_values=[""])
    df = df.drop(columns=[c for c in df.columns if c.startswith("Unnamed")])
    missing = set(PAX_COLUMNS) - set(df.columns)
    if missing:
        raise ValueError(f"Passenger file is missing columns: {sorted(missing)}")
    return df.rename(columns=PAX_COLUMNS)


def clean_pax(raw: pd.DataFrame) -> pd.DataFrame:
    """Standardize keys, build a month column, and resolve duplicates.

    Rows with a missing key or a negative value are dropped. Duplicate
    (month, org, des) keys are summed so each market-month appears once.
    """
    df = raw.copy()
    df["org"] = df["org"].str.strip().str.upper()
    df["des"] = df["des"].str.strip().str.upper()
    df = df.dropna(subset=["year", "month_num", "org", "des"])
    bad_month = ~df["month_num"].between(1, 12)
    if bad_month.any():
        log.warning("Dropping %d rows with an invalid month", int(bad_month.sum()))
        df = df[~bad_month]
    df["month"] = pd.to_datetime(
        {"year": df["year"].astype(int), "month": df["month_num"].astype(int), "day": 1}
    )
    for col in ("od_pax", "od_rev_usd"):
        neg = df[col] < 0
        if neg.any():
            log.warning("Dropping %d rows with negative %s", int(neg.sum()), col)
            df = df[~neg]
    df = df[df["org"] != df["des"]]
    df = (
        df.groupby(["month", "org", "des"], as_index=False, observed=True)[["od_pax", "od_rev_usd"]]
        .sum()
        .sort_values(["org", "des", "month"], ignore_index=True)
    )
    df["org"] = df["org"].astype(str)
    df["des"] = df["des"].astype(str)
    return df


def load_pax(path: Path | str = config.PAX_FILE, use_cache: bool = True) -> pd.DataFrame:
    """Read and clean the selected CSV without executable cache deserialization.

    ``use_cache`` remains accepted for compatibility with existing team scripts.
    Legacy pickle caches are deliberately neither loaded nor refreshed.
    """
    return clean_pax(read_pax_raw(Path(path)))


def dms_to_decimal(deg, minute, sec, hemi) -> np.ndarray:
    """Convert degree/minute/second columns to signed decimal degrees."""
    value = np.asarray(deg, float) + np.asarray(minute, float) / 60 + np.asarray(sec, float) / 3600
    sign = np.where(np.isin(np.asarray(hemi, str), ["S", "W"]), -1.0, 1.0)
    return value * sign


def load_airports(path: Path | str = config.AIRPORT_FILE) -> pd.DataFrame:
    """Read the airport reference table with decimal coordinates."""
    # keep_default_na=False stops codes such as "NA" (Namibia) becoming NaN.
    raw = pd.read_csv(path, dtype=str, keep_default_na=False)
    raw = raw.drop(columns=[c for c in raw.columns if c.startswith("Unnamed")])
    df = raw.rename(columns=AIRPORT_COLUMNS)
    df["lat"] = dms_to_decimal(
        raw["LAT_DEGREE_QTY"], raw["LAT_MINUTE_QTY"], raw["LAT_SECOND_QTY"], raw["LAT_HEMSPHR_CD"]
    )
    df["lon"] = dms_to_decimal(
        raw["LNGTD_DEGREE_QTY"], raw["LNGTD_MINUTE_QTY"], raw["LNGTD_SECOND_QTY"], raw["LNGTD_HEMSPHR_CD"]
    )
    for col in ("world_area", "runway_ft", "elevation_ft"):
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df["state"] = df["state"].replace("", pd.NA)
    keep = ["airport", "name", "metro", "state", "country", "world_area", "lat", "lon", "runway_ft", "elevation_ft"]
    return df[keep].drop_duplicates("airport").set_index("airport")


@dataclass
class Panel:
    """Dense market-by-month matrices built from the long table."""

    months: pd.DatetimeIndex
    markets: pd.DataFrame  # columns org, des; row i matches matrix row i
    pax: np.ndarray  # (n_markets, n_months), zero where unreported
    rev: np.ndarray
    observed: np.ndarray  # bool, True where a row existed

    def values(self, target_col: str) -> np.ndarray:
        return self.pax if target_col == "od_pax" else self.rev

    def month_index(self, month) -> int:
        return int(self.months.get_loc(pd.Timestamp(month)))


def build_panel(df: pd.DataFrame) -> Panel:
    """Pivot the long table into dense matrices over a contiguous month range."""
    months = pd.date_range(df["month"].min(), df["month"].max(), freq="MS")
    markets = df[["org", "des"]].drop_duplicates().sort_values(["org", "des"], ignore_index=True)
    mkt_idx = pd.MultiIndex.from_frame(markets).get_indexer(pd.MultiIndex.from_frame(df[["org", "des"]]))
    mon_idx = months.get_indexer(df["month"])
    shape = (len(markets), len(months))
    pax = np.zeros(shape, dtype=np.float64)
    rev = np.zeros(shape, dtype=np.float64)
    observed = np.zeros(shape, dtype=bool)
    pax[mkt_idx, mon_idx] = df["od_pax"].to_numpy()
    rev[mkt_idx, mon_idx] = df["od_rev_usd"].to_numpy()
    observed[mkt_idx, mon_idx] = True
    return Panel(months=months, markets=markets, pax=pax, rev=rev, observed=observed)
