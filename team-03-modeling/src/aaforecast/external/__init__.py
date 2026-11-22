"""Download public datasets and turn them into tidy tables keyed by airport or month.

Processed tables are written to data/external/processed:

* macro_monthly.csv       fuel, oil, labor, sentiment, CPI, income, FX by month
* airport_population.csv  catchment population by airport and year (Census counties)
* airport_cbsa_pop.csv    home metro area population by airport and year (Census CBSA)
* airport_enplanements.csv FAA enplanements by airport and calendar year
* market_migration.csv    county-to-county movers between airport catchments
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import pandas as pd

from .. import config
from . import catchment, census, faa, fred

log = logging.getLogger(__name__)
PROCESSED = config.EXTERNAL_DIR / "processed"


@dataclass
class External:
    macro: pd.DataFrame | None = None
    population: pd.DataFrame | None = None
    cbsa_pop: pd.DataFrame | None = None
    enplanements: pd.DataFrame | None = None
    migration: pd.DataFrame | None = None


def fetch_all(airports: pd.DataFrame, force: bool = False) -> dict[str, str]:
    """Fetch every source. Failures are logged and reported, never fatal."""
    PROCESSED.mkdir(parents=True, exist_ok=True)
    status: dict[str, str] = {}

    def run(name, fn):
        try:
            fn()
            status[name] = "ok"
        except Exception as exc:
            log.warning("External source %s failed: %s", name, exc)
            status[name] = f"failed: {exc}"

    run("fred", lambda: fred.fetch(config.EXTERNAL_DIR / "fred", force).to_csv(PROCESSED / "macro_monthly.csv"))

    cdir = config.EXTERNAL_DIR / "census"
    points = {}

    def counties():
        pts = census.county_points(cdir, force)
        points["pts"] = pts
        assign = catchment.assign_counties(pts, airports)
        points["assign"] = assign
        assign.to_csv(PROCESSED / "county_catchment.csv", index=False)
        pop = catchment.airport_population(assign, census.county_population(cdir, force), airports)
        pop.to_csv(PROCESSED / "airport_population.csv", index=False)

    run("census_county_population", counties)

    def cbsa():
        cbsa_pop, lookup = census.metro_population(cdir, force)
        pts = points.get("pts")
        if pts is None:
            pts = census.county_points(cdir, force)
        catchment.airport_home_cbsa(pts, lookup, cbsa_pop, airports).to_csv(PROCESSED / "airport_cbsa_pop.csv", index=False)

    run("census_metro_population", cbsa)

    def migration():
        assign = points.get("assign")
        if assign is None:
            raise RuntimeError("county catchments unavailable")
        flows = census.county_migration(cdir, force)
        catchment.market_migration(assign, flows, airports).to_csv(PROCESSED / "market_migration.csv", index=False)

    run("census_migration", migration)
    run(
        "faa_enplanements",
        lambda: faa.enplanements(config.EXTERNAL_DIR / "faa", force).to_csv(PROCESSED / "airport_enplanements.csv", index=False),
    )
    return status


def load() -> External:
    """Load whichever processed tables exist."""

    def read(name, **kw):
        path = PROCESSED / name
        return pd.read_csv(path, **kw) if path.exists() else None

    macro = read("macro_monthly.csv", index_col="month", parse_dates=["month"])
    return External(
        macro=macro,
        population=read("airport_population.csv"),
        cbsa_pop=read("airport_cbsa_pop.csv", dtype={"cbsa": str}),
        enplanements=read("airport_enplanements.csv"),
        migration=read("market_migration.csv"),
    )
