"""Census county population, county centroids, and county-to-county migration."""

from __future__ import annotations

import logging
import re
from pathlib import Path

import pandas as pd

from .http import download

log = logging.getLogger(__name__)

COUNTY_POP_URLS = [
    "https://www2.census.gov/programs-surveys/popest/datasets/2020-2025/counties/totals/co-est2025-alldata.csv",
    "https://www2.census.gov/programs-surveys/popest/datasets/2020-2024/counties/totals/co-est2024-alldata.csv",
]
METRO_POP_URLS = [
    "https://www2.census.gov/programs-surveys/popest/datasets/2020-2025/metro/totals/cbsa-est2025-alldata.csv",
    "https://www2.census.gov/programs-surveys/popest/datasets/2020-2024/metro/totals/cbsa-est2024-alldata.csv",
]
COUNTY_POINTS_URL = (
    "https://public.opendatasoft.com/api/explore/v2.1/catalog/datasets/us-county-boundaries/"
    "exports/csv?select=geoid,name,stusab,intptlat,intptlon"
)
MIGRATION_URL = (
    "https://www2.census.gov/programs-surveys/demo/tables/geographic-mobility/2020/"
    "county-to-county-migration-2016-2020/county-to-county-migration-flows/CtyxCty_US.txt"
)


def _first_available(urls: list[str], dest_dir: Path, force: bool) -> Path:
    last_exc = None
    for url in urls:
        try:
            return download(url, dest_dir / url.rsplit("/", 1)[-1], force=force)
        except Exception as exc:
            last_exc = exc
    raise RuntimeError(f"No source available: {last_exc}")


def county_population(dest_dir: Path, force: bool = False) -> pd.DataFrame:
    """County population estimates, long format: fips, year, population."""
    path = _first_available(COUNTY_POP_URLS, dest_dir, force)
    df = pd.read_csv(path, dtype={"STATE": str, "COUNTY": str}, encoding="latin-1")
    df = df[df["SUMLEV"].astype(int) == 50]
    df["fips"] = df["STATE"].str.zfill(2) + df["COUNTY"].str.zfill(3)
    pop_cols = [c for c in df.columns if re.fullmatch(r"POPESTIMATE\d{4}", c)]
    long = df.melt(id_vars="fips", value_vars=pop_cols, var_name="year", value_name="population")
    long["year"] = long["year"].str[-4:].astype(int)
    return long


def metro_population(dest_dir: Path, force: bool = False) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Metro and micro statistical area (CBSA) totals plus the county to CBSA lookup.

    Returns (cbsa_pop long table: cbsa, name, year, population; county_cbsa: fips, cbsa).
    """
    path = _first_available(METRO_POP_URLS, dest_dir, force)
    df = pd.read_csv(path, dtype={"CBSA": str, "STCOU": str}, encoding="latin-1")
    areas = df[df["LSAD"].isin(["Metropolitan Statistical Area", "Micropolitan Statistical Area"])]
    pop_cols = [c for c in df.columns if re.fullmatch(r"POPESTIMATE\d{4}", c)]
    long = areas.melt(id_vars=["CBSA", "NAME"], value_vars=pop_cols, var_name="year", value_name="population")
    long["year"] = long["year"].str[-4:].astype(int)
    counties = df[df["STCOU"].notna() & (df["LSAD"] == "County or equivalent")]
    lookup = pd.DataFrame({"fips": counties["STCOU"].str.zfill(5), "cbsa": counties["CBSA"]}).drop_duplicates("fips")
    return long.rename(columns={"CBSA": "cbsa", "NAME": "name"}), lookup


def county_points(dest_dir: Path, force: bool = False) -> pd.DataFrame:
    """Internal point (lat, lon) for each US county from the county boundaries dataset."""
    path = download(COUNTY_POINTS_URL, dest_dir / "us_county_points.csv", force=force)
    df = pd.read_csv(path, sep=";", dtype={"geoid": str})
    df = df.rename(columns={"geoid": "fips", "intptlat": "lat", "intptlon": "lon"})
    df["fips"] = df["fips"].str.zfill(5)
    df["lat"] = pd.to_numeric(df["lat"], errors="coerce")
    df["lon"] = pd.to_numeric(df["lon"], errors="coerce")
    return df[["fips", "name", "stusab", "lat", "lon"]].dropna()


def county_migration(dest_dir: Path, force: bool = False) -> pd.DataFrame:
    """County-to-county mover flows (ACS 2016-2020): dest_fips, src_fips, movers.

    Each record gives movers now living in county A who lived in county B one year
    earlier. The two FIPS blocks sit at the start of every line and the flow
    estimate and its margin of error are the last two fields.
    """
    path = download(MIGRATION_URL, dest_dir / "CtyxCty_US_2016_2020.txt", force=force)
    rows = []
    with open(path, encoding="latin-1") as fh:
        for line in fh:
            if len(line) < 12:
                continue
            a, b = line[0:6], line[6:12]
            if not (a.isdigit() and b.isdigit()):
                continue  # flows from abroad or other non-county origins
            toks = line.split()
            try:
                movers = float(toks[-2])
            except (ValueError, IndexError):
                continue
            rows.append((a[1:3] + a[3:6], b[1:3] + b[3:6], movers))
    return pd.DataFrame(rows, columns=["dest_fips", "src_fips", "movers"])
