"""Paths and settings. The real column names stay in the git-ignored data/real/columns.json, never here."""
import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
REAL_DIR = ROOT / "data" / "real"
FAKE_DIR = ROOT / "data" / "fake"
# Real data takes over as soon as data/real/columns.json exists. Until then everything runs on fake data.
DATA_DIR = REAL_DIR if (REAL_DIR / "columns.json").exists() else FAKE_DIR
PROCESSED_DIR = ROOT / "data" / "processed"
RESULTS_DIR = ROOT / "team-04-modeling" / "results"

# Names the code uses. columns.json maps each one to the matching column in the data files.
TARGET = "PAX"
DATE = "DATE"  # built from YEAR and MONTH in load()
ROUTE_KEYS = ["ORIGIN", "DEST"]
AIRPORT_KEY = "AIRPORT"
AIRPORT_NUMERIC = ["LAT", "LON", "RUNWAY_FT", "ELEVATION_FT"]
AIRPORT_CATEGORICAL = ["REGION"]
LEAKY = ["REVENUE"]  # only known after passengers have flown

PERIOD = pd.DateOffset(months=1)
HORIZON = 1  # periods ahead we forecast. Confirm with a PM: this decides which lags are legal.
HOLDOUT_PERIODS = 12
BACKTEST_FOLDS = 3
MIN_GAIN = 0.001  # a feature group stays only if removing it worsens WAPE by at least this much


def read_table(columns, name, keys):
    # Only blanks are missing: otherwise pandas reads a country code like "NA" as a gap.
    return (
        pd.read_csv(DATA_DIR / columns[f"{name}_file"], keep_default_na=False, na_values=[""], low_memory=False,
                    dtype={columns[name][k]: str for k in keys})
        .rename(columns={real: ours for ours, real in columns[name].items()})
        .drop(columns=columns["drop"], errors="ignore")
    )


def load():
    """Return (pax, airports) using our column names, and say which data this is."""
    print(f"USING {'REAL' if DATA_DIR == REAL_DIR else 'FAKE'} DATA from {DATA_DIR.relative_to(ROOT)}")
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    columns = json.loads((DATA_DIR / "columns.json").read_text())
    pax = read_table(columns, "pax", ROUTE_KEYS)
    pax[DATE] = pd.to_datetime({"year": pax["YEAR"], "month": pax["MONTH"], "day": 1})
    return pax, read_table(columns, "airports", [AIRPORT_KEY])
