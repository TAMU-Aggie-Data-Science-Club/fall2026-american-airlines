"""Shared data loader for Team 10.

Every team-10 notebook gets the data through ``load_data()`` so we all work from
the same rows with the same column types.

Usage (from a notebook in this folder)::

    from load_data import load_data
    df = load_data()                       # the team's fare file in data/raw/
    df = load_data("some_other_file.csv")  # or a specific file in data/raw/

The dataset lives in ``<repo>/data/raw/``, which is git-ignored. Never copy it
anywhere else in the repo, and never commit a sample of it.

This loader only *reads and types* the data. It does NOT drop outliers or
engineer features -- those are EDA decisions, made and explained in
``features.py`` and the EDA notebook.
"""

from __future__ import annotations

import warnings
from pathlib import Path

import pandas as pd

# notebooks/ -> fare-side-study/ -> team-10-modeling/ -> repo root
REPO_ROOT = Path(__file__).resolve().parents[3]
RAW_DIR = REPO_ROOT / "data" / "raw"
DEFAULT_FILE = "aa_fares_compact_SYNTHETIC.csv"

TARGET = "total_price"

# Column type map for the compact fare file.
CABIN_ORDER = ["Basic Economy", "Main Cabin", "Premium Economy", "Business/First"]
CATEGORY_COLS = ["origin", "destination"]
BOOL_COLS = ["is_weekend_departure", "is_holiday_period", "nonstop"]
INT_COLS = ["days_to_departure", "distance_miles", "departure_hour", "month"]
FLOAT_COLS = ["load_factor", "competitor_min_price", "total_price"]

EXPECTED_COLS = set(CATEGORY_COLS + ["cabin"] + BOOL_COLS + INT_COLS + FLOAT_COLS)


def _raw_path(filename: str | None) -> Path:
    path = RAW_DIR / (filename or DEFAULT_FILE)
    if not path.is_file():
        raise FileNotFoundError(
            f"{path} not found. Put the team's data file in data/raw/ (see DATA.md)."
        )
    return path


def fix_types(df: pd.DataFrame) -> pd.DataFrame:
    """Apply the team's column types. Returns a new DataFrame."""
    df = df.copy()
    df.columns = df.columns.str.strip()
    for col in df.select_dtypes(include=["object", "string"]).columns:
        df[col] = df[col].str.strip()

    for col in CATEGORY_COLS:
        if col in df:
            df[col] = df[col].astype("category")
    if "cabin" in df:
        unknown = set(df["cabin"].dropna()) - set(CABIN_ORDER)
        if unknown:
            warnings.warn(f"cabin: unexpected labels {sorted(unknown)} -> set to NaN.")
        df["cabin"] = pd.Categorical(df["cabin"], categories=CABIN_ORDER, ordered=True)
    for col in BOOL_COLS:
        if col in df:
            df[col] = pd.to_numeric(df[col], errors="coerce").astype("Int64")
    for col in INT_COLS:
        if col in df:
            df[col] = pd.to_numeric(df[col], errors="coerce").astype("Int64")
    for col in FLOAT_COLS:
        if col in df:
            df[col] = pd.to_numeric(df[col], errors="coerce").astype("float64")

    extra = sorted(set(df.columns) - EXPECTED_COLS)
    missing = sorted(EXPECTED_COLS - set(df.columns))
    if extra:
        warnings.warn(f"Columns not in the team type map (left as inferred): {extra}")
    if missing:
        warnings.warn(f"Expected columns not found in file: {missing}")
    return df


def load_data(filename: str | None = None, *, raw: bool = False) -> pd.DataFrame:
    """Load the fare file from data/raw/ with team-standard column types.

    Parameters
    ----------
    filename : str, optional
        File name inside data/raw/. Defaults to the compact fare file.
    raw : bool, default False
        If True, return exactly what pandas reads (all columns as strings).
    """
    path = _raw_path(filename)
    if raw:
        return pd.read_csv(path, dtype=str, keep_default_na=False)
    return fix_types(pd.read_csv(path))


if __name__ == "__main__":
    data = load_data()
    print(f"{data.shape[0]:,} rows x {data.shape[1]} columns")
    print(data.dtypes.to_string())
