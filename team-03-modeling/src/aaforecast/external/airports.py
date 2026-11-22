"""Public airport reference from OurAirports; no sponsor identifiers are sent."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd
from .http import download

BASE = "https://davidmegginson.github.io/ourairports-data/"


def convert_reference(airports, runways):
    table = airports[airports["iata_code"].fillna("").str.fullmatch(r"[A-Z]{3}")].copy()
    table["priority"] = table["type"].map({"large_airport": 0, "medium_airport": 1, "small_airport": 2, "closed": 9}).fillna(5)
    table = table.sort_values(["priority", "id"]).drop_duplicates("iata_code")
    longest = runways.groupby("airport_ref")["length_ft"].max()
    out = pd.DataFrame(index=table.index)
    out["AIRPRT_CD"] = table["iata_code"]
    out["AIRPRT_NM"] = table["name"]
    # A municipality is not an authoritative metro code: keep airports separate.
    out["CITY_METRO_IATA_CD"] = table["iata_code"]
    out["STATE_PROVNC_CD"] = table["iso_region"].fillna("").str.split("-").str[-1]
    out["CNTRY_CD"] = table["iso_country"]
    out["WRLD_AREA_DOT_CD"] = np.nan
    out["LNGST_RUNWAY_FT_QTY"] = table["id"].map(longest)
    out["ELEVATN_FT_QTY"] = table["elevation_ft"]
    for field, prefix, hemis in (("latitude_deg", "LAT", ("N", "S")), ("longitude_deg", "LNGTD", ("E", "W"))):
        values = pd.to_numeric(table[field], errors="coerce")
        absolute = values.abs()
        degrees = np.floor(absolute)
        minutes = np.floor((absolute - degrees) * 60)
        seconds = ((absolute - degrees) * 60 - minutes) * 60
        out[f"{prefix}_DEGREE_QTY"] = degrees
        out[f"{prefix}_MINUTE_QTY"] = minutes
        out[f"{prefix}_SECOND_QTY"] = seconds
        out[f"{prefix}_HEMSPHR_CD"] = np.where(values < 0, hemis[1], hemis[0])
    valid = table.latitude_deg.between(-90, 90) & table.longitude_deg.between(-180, 180)
    out = out.loc[valid].reset_index(drop=True)
    # Alias reference attributes only; never merge historical passenger keys.
    if "PBI" not in set(out.AIRPRT_CD) and "DJT" in set(out.AIRPRT_CD):
        alias = out[out.AIRPRT_CD == "DJT"].copy()
        alias["AIRPRT_CD"] = "PBI"
        alias["CITY_METRO_IATA_CD"] = "PBI"
        out = pd.concat([out, alias], ignore_index=True)
    return out


def fetch_reference(directory, force=False):
    directory = Path(directory)
    paths = {name: download(BASE + name + ".csv", directory / (name + ".csv"), force=force)
             for name in ("airports", "runways")}
    airports = pd.read_csv(paths["airports"], keep_default_na=False, na_values=[""])
    runways = pd.read_csv(paths["runways"])
    reference = convert_reference(airports, runways)
    target = directory / "AA_AIRPORT_INFO.csv"
    reference.to_csv(target, index=False)
    metadata = {"source": BASE, "license": "Unlicense / public-domain dedication", "retrieved": datetime.now(timezone.utc).isoformat(),
                "sha256": {name: hashlib.sha256(path.read_bytes()).hexdigest() for name, path in paths.items()},
                "historical_reference_aliases": {"PBI": "DJT"},
                "alias_source": "https://www.pbia.org/articles/post/president-donald-j-trump-international-airport-name-change-takes-effect/",
                "limitations": "Current public snapshot; DOT area unknown; metro codes use airport codes without pooling; other historical code changes need review."}
    (directory / "provenance.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return target
