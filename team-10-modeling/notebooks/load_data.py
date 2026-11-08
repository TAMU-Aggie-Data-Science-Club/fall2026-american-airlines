"""Shared data loader for Team 10 (Deliverable 3).

Every team-10 notebook should get the data through ``load_data()`` so we all
work from the same rows with the same column types.

Usage (from a notebook in this folder)::

    from load_data import load_data
    df = load_data()                      # the single CSV in data/raw/
    df = load_data("some_other_file.csv") # or a specific file in data/raw/

The dataset lives in ``<repo>/data/raw/``, which is git-ignored. NEVER copy it
anywhere else in the repo, and never commit a sample of it (NDA).

This loader only *reads and types* the data. It does NOT drop duplicates,
outliers, or leakage columns -- those are EDA decisions and are made (and
explained) in the EDA notebooks.
"""

from __future__ import annotations

import warnings
from pathlib import Path

import pandas as pd

# notebooks/ -> team-10-modeling/ -> repo root
REPO_ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = REPO_ROOT / "data" / "raw"

# ---------------------------------------------------------------------------
# Column type map (from the dataset's .md data dictionary).
# Columns that are missing from the file are skipped; columns in the file that
# are not listed here are left as pandas inferred them and reported as a warning,
# so a schema change in the real file shows up as a warning, not as silent errors.
# ---------------------------------------------------------------------------
DATE_COLS = ["search_date", "departure_date"]

CATEGORY_COLS = [
    "origin",
    "destination",
    "carrier",
    "cabin",
    "fare_booking_class",
]

# Ordered categoricals: the order means something (sorting, plots, ordinal encoding).
ORDERED_CATEGORIES = {
    "day_of_week": ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"],
}

# 0/1 flags -> pandas nullable boolean (keeps NaN if the real file has gaps).
BOOL_COLS = ["is_weekend_departure", "is_holiday_period", "nonstop"]

# Whole numbers -> nullable Int64 (survives missing values, unlike int64).
INT_COLS = [
    "search_id",
    "days_to_departure",
    "distance_miles",
    "departure_hour",
    "month",
    "num_stops",
    "aircraft_capacity",
]

FLOAT_COLS = [
    "seats_remaining",
    "load_factor",
    "competitor_min_price",
    "base_fare",
    "taxes_fees",
    "total_price",
]

TARGET = "total_price"

KNOWN_COLS = set(
    DATE_COLS + CATEGORY_COLS + list(ORDERED_CATEGORIES) + BOOL_COLS + INT_COLS + FLOAT_COLS
)


def _find_raw_file(filename: str | None) -> Path:
    """Return the path to the raw data file inside data/raw/."""
    if not RAW_DIR.is_dir():
        raise FileNotFoundError(
            f"{RAW_DIR} does not exist. Download the dataset from the team's "
            "Google Drive link into data/raw/ (see DATA.md)."
        )
    if filename is not None:
        path = RAW_DIR / filename
        if not path.is_file():
            raise FileNotFoundError(f"{path} not found.")
        return path

    candidates = sorted(RAW_DIR.glob("*.csv"))
    if len(candidates) == 1:
        return candidates[0]
    if not candidates:
        raise FileNotFoundError(f"No .csv file in {RAW_DIR}. See DATA.md for how to get it.")
    names = ", ".join(p.name for p in candidates)
    raise ValueError(
        f"Several CSVs in {RAW_DIR} ({names}). Pass one explicitly: load_data('<name>.csv')."
    )


def _to_bool(s: pd.Series) -> pd.Series:
    """0/1 (or True/False, 'yes'/'no') -> pandas nullable boolean."""
    mapping = {1: True, 0: False, "1": True, "0": False, "true": True, "false": False,
               "yes": True, "no": False, "y": True, "n": False, True: True, False: False}
    lowered = s.map(lambda v: v.strip().lower() if isinstance(v, str) else v)
    out = lowered.map(mapping)
    unmapped = s.notna() & out.isna()
    if unmapped.any():
        warnings.warn(f"{s.name}: {int(unmapped.sum())} values not 0/1 -> set to <NA>.")
    return out.astype("boolean")


def fix_types(df: pd.DataFrame) -> pd.DataFrame:
    """Apply the team's column types. Returns a new DataFrame."""
    df = df.copy()
    # Strip stray whitespace from headers and string cells.
    df.columns = df.columns.str.strip()
    for col in df.select_dtypes(include=["object", "string"]).columns:
        df[col] = df[col].str.strip()

    for col in DATE_COLS:
        if col in df:
            df[col] = pd.to_datetime(df[col], errors="coerce")

    for col in CATEGORY_COLS:
        if col in df:
            df[col] = df[col].astype("category")

    for col, order in ORDERED_CATEGORIES.items():
        if col in df:
            df[col] = pd.Categorical(df[col], categories=order, ordered=True)

    for col in BOOL_COLS:
        if col in df:
            df[col] = _to_bool(df[col])

    for col in INT_COLS:
        if col in df:
            num = pd.to_numeric(df[col], errors="coerce")
            if (num.dropna() % 1 != 0).any():
                warnings.warn(f"{col}: has non-whole values, kept as float.")
                df[col] = num.astype("float64")
            else:
                df[col] = num.astype("Int64")

    for col in FLOAT_COLS:
        if col in df:
            df[col] = pd.to_numeric(df[col], errors="coerce").astype("float64")

    unknown = [c for c in df.columns if c not in KNOWN_COLS]
    if unknown:
        warnings.warn(
            f"Columns not in the team type map (left as inferred): {unknown}. "
            "Add them to load_data.py once their meaning is confirmed."
        )
    missing = sorted(KNOWN_COLS - set(df.columns))
    if missing:
        warnings.warn(f"Expected columns not found in file: {missing}")
    return df


def load_data(filename: str | None = None, *, raw: bool = False) -> pd.DataFrame:
    """Load the AA fare dataset from data/raw/ with team-standard column types.

    Parameters
    ----------
    filename : str, optional
        File name inside data/raw/. If omitted, the single CSV there is used.
    raw : bool, default False
        If True, return exactly what pandas reads (all columns as strings),
        with no type fixing -- useful for checking what the type fixes changed.
    """
    path = _find_raw_file(filename)
    if raw:
        return pd.read_csv(path, dtype=str, keep_default_na=False)
    df = pd.read_csv(path, low_memory=False)
    return fix_types(df)


if __name__ == "__main__":
    data = load_data()
    print(f"{data.shape[0]:,} rows x {data.shape[1]} columns")
    print(data.dtypes.to_string())
