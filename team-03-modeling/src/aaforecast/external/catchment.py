"""Map county-level public data onto airports.

Every US county is assigned to its nearest airport in the AA network (within a
distance cap). Airports that share a metro code pool their counties, so the
three New York airports share one catchment. County statistics are then summed
by catchment.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..geo import US_COUNTRIES, nearest

MAX_CATCHMENT_KM = 160.0


def assign_counties(points: pd.DataFrame, airports: pd.DataFrame, max_km: float = MAX_CATCHMENT_KM) -> pd.DataFrame:
    """Return fips, airport, metro, distance_km for each county inside a catchment."""
    us = airports[airports["country"].isin(US_COUNTRIES)]
    idx, dist = nearest(points["lat"], points["lon"], us["lat"], us["lon"], max_km=max_km)
    out = pd.DataFrame({"fips": points["fips"].to_numpy(), "airport_idx": idx, "distance_km": dist})
    out = out[out["airport_idx"] >= 0]
    out["airport"] = us.index.to_numpy()[out["airport_idx"]]
    out["metro"] = us["metro"].to_numpy()[out["airport_idx"]]
    return out.drop(columns="airport_idx").reset_index(drop=True)


def airport_population(assign: pd.DataFrame, county_pop: pd.DataFrame, airports: pd.DataFrame) -> pd.DataFrame:
    """Catchment population per airport and year (shared across a metro)."""
    pop = county_pop.merge(assign[["fips", "metro"]], on="fips")
    metro_pop = pop.groupby(["metro", "year"], as_index=False)["population"].sum()
    ap = airports.reset_index()[["airport", "metro"]]
    return ap.merge(metro_pop, on="metro")[["airport", "year", "population"]].rename(
        columns={"population": "catchment_pop"}
    )


def airport_home_cbsa(points: pd.DataFrame, county_cbsa: pd.DataFrame, cbsa_pop: pd.DataFrame, airports: pd.DataFrame) -> pd.DataFrame:
    """Population of the CBSA containing each airport (via the nearest county point)."""
    us = airports[airports["country"].isin(US_COUNTRIES)]
    pts = points.merge(county_cbsa, on="fips")
    idx, _ = nearest(us["lat"], us["lon"], pts["lat"], pts["lon"], max_km=60)
    home = pd.DataFrame({"airport": us.index, "cbsa": np.where(idx >= 0, pts["cbsa"].to_numpy()[np.maximum(idx, 0)], None)})
    out = home.dropna().merge(cbsa_pop[["cbsa", "year", "population"]], on="cbsa")
    return out.rename(columns={"population": "cbsa_pop"})[["airport", "year", "cbsa_pop"]]


def market_migration(assign: pd.DataFrame, flows: pd.DataFrame, airports: pd.DataFrame) -> pd.DataFrame:
    """Annual movers between two airports' catchments, summed over both directions."""
    m = assign.set_index("fips")["metro"]
    f = flows.assign(dest_metro=flows["dest_fips"].map(m), src_metro=flows["src_fips"].map(m)).dropna()
    f = f[f["dest_metro"] != f["src_metro"]]
    pair = f.groupby(["src_metro", "dest_metro"])["movers"].sum()
    both = pair.add(pair.swaplevel().rename_axis(pair.index.names), fill_value=0)
    both = both.rename("movers").reset_index().rename(columns={"src_metro": "org_metro", "dest_metro": "des_metro"})
    return both
