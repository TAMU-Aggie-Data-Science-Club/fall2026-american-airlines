"""Team 10 data access: find the files, load them with fixed types, decide where outputs go.

Data mode
---------
* REAL: both ``data/raw/*PAX_FORECASTING*.csv`` and ``data/raw/*AIRPORT_INFO*.csv`` exist
  (the NDA files, downloaded by a team member from the team Drive link).
* FAKE: otherwise, or whenever the environment variable ``TEAM10_FORCE_FAKE=1`` is set (used to
  test code changes without loading the NDA files). Uses ``data/fake/FAKE_*.csv`` from
  ``scripts/make_fake_data.py`` (generated on first use if missing).

Outputs follow the mode so real numbers never land in a tracked folder by accident:
* FAKE: tables -> ``team-10-modeling/results/generated/``, figures -> ``team-10-modeling/visuals/``
* REAL: tables and figures -> ``data/interim/team-10/`` (git-ignored). Copy *rounded* numbers into
  the narrative reports by hand, and check with a PM before committing any real-data chart.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pandas as pd

MODELING = Path(__file__).resolve().parents[1]
REPO_ROOT = MODELING.parent
RAW, FAKE = REPO_ROOT / "data" / "raw", REPO_ROOT / "data" / "fake"


def _one(folder: Path, pattern: str) -> Path | None:
    hits = sorted(folder.glob(f"*{pattern}*.csv")) if folder.is_dir() else []
    if len(hits) > 1:
        raise ValueError(f"Several {pattern} files in {folder}: {[h.name for h in hits]}")
    return hits[0] if hits else None


def _paths() -> tuple[str, Path, Path]:
    if os.environ.get("TEAM10_FORCE_FAKE") != "1":
        pax, apt = _one(RAW, "PAX_FORECASTING"), _one(RAW, "AIRPORT_INFO")
        if pax and apt:
            return "REAL", pax, apt
    fpax, fapt = FAKE / "FAKE_PAX_FORECASTING.csv", FAKE / "FAKE_AIRPORT_INFO.csv"
    if not (fpax.exists() and fapt.exists()):
        subprocess.run([sys.executable, str(MODELING / "scripts" / "make_fake_data.py")], check=True)
    return "FAKE", fpax, fapt


MODE, PAX_PATH, AIRPORT_PATH = _paths()

if MODE == "FAKE":
    TABLES, FIGS = MODELING / "results" / "generated", MODELING / "visuals"
else:
    TABLES = FIGS = REPO_ROOT / "data" / "interim" / "team-10"
TABLES.mkdir(parents=True, exist_ok=True)
FIGS.mkdir(parents=True, exist_ok=True)

BANNER = (
    "USING FAKE DATA: pipeline check only. These numbers say nothing about American Airlines."
    if MODE == "FAKE" else
    "USING REAL DATA (NDA): outputs stay in data/interim/team-10/. Do not commit them."
)


def tag(title: str) -> str:
    """Stamp FAKE charts so they can never pass for real ones; real charts keep a clean title."""
    return f"[FAKE DATA] {title}" if MODE == "FAKE" else title


def load_pax() -> pd.DataFrame:
    """Route-month passenger file with fixed types and a DATE column (first of the month)."""
    df = pd.read_csv(PAX_PATH, dtype={"ORG_AIRPRT": "string", "DES_AIRPRT": "string"})
    df["DEP_YEAR"] = pd.to_numeric(df["DEP_YEAR"], errors="coerce").astype("Int64")
    df["DEP_MONTH"] = pd.to_numeric(df["DEP_MONTH"], errors="coerce").astype("Int64")
    for c in ("OD_PAX", "OD_REV_USD"):
        if c in df:
            df[c] = pd.to_numeric(df[c], errors="coerce").astype("float64")
    df["DATE"] = pd.to_datetime(
        {"year": df["DEP_YEAR"], "month": df["DEP_MONTH"], "day": 1}, errors="coerce")
    return df


def load_airports() -> pd.DataFrame:
    """Airport lookup. keep_default_na=False so a literal 'NA' (Namibia) is not read as missing."""
    df = pd.read_csv(AIRPORT_PATH, keep_default_na=False, na_values=[""])
    for c in ("AIRPRT_CD", "CITY_METRO_IATA_CD", "STATE_PROVNC_CD", "CNTRY_CD"):
        if c in df:
            df[c] = df[c].astype("string").str.strip()
    return df


def write_table(name: str, markdown: str) -> Path:
    path = TABLES / name
    path.write_text(f"> {BANNER}\n\n{markdown}", encoding="utf-8")
    return path
