"""Geographic helpers: distances, nearest-airport assignment, market attributes."""

from __future__ import annotations

import numpy as np
import pandas as pd

from .config import AA_HUBS

EARTH_RADIUS_KM = 6371.0088
US_COUNTRIES = {"US", "PR", "VI", "GU"}


def haversine_km(lat1, lon1, lat2, lon2) -> np.ndarray:
    """Great-circle distance in kilometers. Inputs are decimal degrees."""
    lat1, lon1, lat2, lon2 = (np.radians(np.asarray(x, float)) for x in (lat1, lon1, lat2, lon2))
    a = np.sin((lat2 - lat1) / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2) ** 2
    return 2 * EARTH_RADIUS_KM * np.arcsin(np.sqrt(np.clip(a, 0, 1)))


def nearest(points_lat, points_lon, ref_lat, ref_lon, max_km: float | None = None):
    """For each point, return the index of the nearest reference location and its distance.

    Points farther than ``max_km`` from every reference get index -1.
    """
    plat = np.asarray(points_lat, float)[:, None]
    plon = np.asarray(points_lon, float)[:, None]
    d = haversine_km(plat, plon, np.asarray(ref_lat, float)[None, :], np.asarray(ref_lon, float)[None, :])
    idx = d.argmin(axis=1)
    dist = d[np.arange(len(idx)), idx]
    if max_km is not None:
        idx = np.where(dist <= max_km, idx, -1)
    return idx, dist


def market_attributes(markets: pd.DataFrame, airports: pd.DataFrame) -> pd.DataFrame:
    """Static per-market features from the airport reference table."""
    o = airports.reindex(markets["org"]).reset_index(drop=True)
    d = airports.reindex(markets["des"]).reset_index(drop=True)
    out = pd.DataFrame(index=markets.index)
    out["distance_km"] = haversine_km(o["lat"], o["lon"], d["lat"], d["lon"])
    o_us = o["country"].isin(US_COUNTRIES).to_numpy()
    d_us = d["country"].isin(US_COUNTRIES).to_numpy()
    out["domestic"] = (o_us & d_us).astype(int)
    unknown = o["country"].fillna("").eq("") | d["country"].fillna("").eq("")
    out.loc[unknown, "domestic"] = -1
    out["org_us"] = o_us.astype(int)
    out["org_hub"] = markets["org"].isin(AA_HUBS).astype(int).to_numpy()
    out["des_hub"] = markets["des"].isin(AA_HUBS).astype(int).to_numpy()
    out["org_lat"], out["org_lon"] = o["lat"].to_numpy(), o["lon"].to_numpy()
    out["des_lat"], out["des_lon"] = d["lat"].to_numpy(), d["lon"].to_numpy()
    out["org_world_area"] = o["world_area"].to_numpy()
    out["des_world_area"] = d["world_area"].to_numpy()
    out["org_runway_ft"] = o["runway_ft"].to_numpy()
    out["des_runway_ft"] = d["runway_ft"].to_numpy()
    # The foreign end of an international market, used to attach FX rates.
    foreign = np.where(~d_us, d["country"], np.where(~o_us, o["country"], ""))
    out["foreign_country"] = foreign
    return out
