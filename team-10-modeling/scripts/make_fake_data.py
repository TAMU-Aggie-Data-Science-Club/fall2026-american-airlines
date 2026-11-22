"""Generate a FAKE route-month passenger panel with the same columns as the AA files.

Nothing here is derived from the real data. Values come from a random process tuned only to
have the *shape* the project brief describes (monthly route panel, 2019-01 to 2026-07, strong
yearly seasonality, a 2020 collapse and recovery, right-skewed route sizes, missing
route-months, non-US airports without a state code). Airport codes are invented (XAA, XAB, ...).

Output (git-ignored):
    data/fake/FAKE_PAX_FORECASTING.csv
    data/fake/FAKE_AIRPORT_INFO.csv

Run from anywhere:  python team-10-modeling/scripts/make_fake_data.py
"""

from __future__ import annotations

import string
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
OUT = REPO_ROOT / "data" / "fake"
SEED = 10
N_AIRPORTS = 60
N_HUBS = 8
START, END = "2019-01-01", "2026-07-01"
LOAD_TMS = "2026-08-15 03:12:00.000"

US_STATES = ["TX", "CA", "NY", "IL", "FL", "NC", "PA", "AZ", "CO", "WA", "GA", "MA", "NV", "MN"]
INTL = [("CA", 2, "ON"), ("MX", 3, None), ("GB", 4, None), ("DE", 4, None),
        ("JP", 6, None), ("BR", 5, None)]


def _dms(value: float) -> tuple[int, int, int]:
    value = abs(value)
    d = int(value)
    m = int((value - d) * 60)
    s = int(round(((value - d) * 60 - m) * 60)) % 60
    return d, m, s


def make_airports(rng: np.random.Generator) -> pd.DataFrame:
    letters = string.ascii_uppercase
    rows = []
    for i in range(N_AIRPORTS):
        code = "X" + letters[i // 26] + letters[i % 26]
        us = i < N_HUBS or rng.random() < 0.72
        if us:
            cntry, area, state = "US", 1, rng.choice(US_STATES)
            lat, lng = rng.uniform(25, 48), -rng.uniform(70, 122)
        else:
            cntry, area, state = INTL[rng.integers(len(INTL))]
            lat, lng = rng.uniform(-30, 55), rng.uniform(-120, 140)
        la, lo = _dms(lat), _dms(lng)
        rows.append({
            "AIRPRT_CD": code,
            "AIRPRT_NM": f"Fake Airport {i:02d}",
            "CITY_METRO_IATA_CD": code if rng.random() > 0.1 else "XMT",
            "STATE_PROVNC_CD": state,
            "CNTRY_CD": cntry if rng.random() > 0.02 else None,
            "WRLD_AREA_DOT_CD": area,
            "LAT_DEGREE_QTY": la[0], "LAT_MINUTE_QTY": la[1], "LAT_SECOND_QTY": la[2],
            "LAT_HEMSPHR_CD": "N" if lat >= 0 else "S",
            "LNGTD_DEGREE_QTY": lo[0], "LNGTD_MINUTE_QTY": lo[1], "LNGTD_SECOND_QTY": lo[2],
            "LNGTD_HEMSPHR_CD": "E" if lng >= 0 else "W",
            "LNGST_RUNWAY_FT_QTY": int(rng.integers(5000, 13500)),
            "ELEVATN_FT_QTY": int(rng.integers(0, 5500)),
            "SF_LOAD_TMS": LOAD_TMS,
            "_lat": lat, "_lng": lng, "_hub": i < N_HUBS,
        })
    return pd.DataFrame(rows)


def covid_factor(dates: pd.DatetimeIndex, intl: bool) -> np.ndarray:
    """Demand multiplier: collapse in spring 2020, recovery through 2022 (slower abroad)."""
    t = ((dates.year - 2020) * 12 + dates.month - 3).to_numpy()  # months since 2020-03
    f = np.ones(len(dates))
    hit = (t >= 0)
    rec_months = 22 if intl else 18
    low = 0.04 if intl else 0.08
    f[hit] = np.clip(low + (1 - low) * (t[hit] / rec_months) ** 1.3, low, 1.0)
    return f


def make_pax(rng: np.random.Generator, apt: pd.DataFrame) -> pd.DataFrame:
    dates = pd.date_range(START, END, freq="MS")
    codes = apt["AIRPRT_CD"].to_numpy()
    hub = apt["_hub"].to_numpy()
    cntry = apt["CNTRY_CD"].fillna("??").to_numpy()
    lat, lng = np.radians(apt["_lat"].to_numpy()), np.radians(apt["_lng"].to_numpy())

    rows = []
    for i in range(N_AIRPORTS):
        for j in range(N_AIRPORTS):
            if i == j:
                continue
            p = 0.9 if (hub[i] or hub[j]) else 0.18
            if rng.random() > p:
                continue
            intl = cntry[i] != "US" or cntry[j] != "US"
            dist = 3959 * 2 * np.arcsin(np.sqrt(
                np.sin((lat[j] - lat[i]) / 2) ** 2
                + np.cos(lat[i]) * np.cos(lat[j]) * np.sin((lng[j] - lng[i]) / 2) ** 2))
            size = np.exp(rng.normal(6.0, 1.4)) * (3 if hub[i] and hub[j] else 1.5 if hub[i] or hub[j] else 1)
            amp = rng.uniform(0.10, 0.35) * (1.4 if intl else 1.0)
            peak = 7 + rng.normal(0, 0.7)
            growth = rng.normal(0.025, 0.04)
            m = dates.month.to_numpy()
            season = 1 + amp * np.cos(2 * np.pi * (m - peak) / 12) + 0.08 * (m == 12)
            years = (dates.year - 2019 + (m - 1) / 12).to_numpy()
            level = size * season * (1 + growth) ** years * covid_factor(dates, intl)
            pax = level * np.exp(rng.normal(0, 0.12, len(dates)))
            start = dates[0] if rng.random() > 0.1 else dates[rng.integers(6, 70)]
            end = dates[-1] if rng.random() > 0.05 else dates[rng.integers(30, len(dates) - 3)]
            keep = (dates >= start) & (dates <= end) & (pax >= 0.5) & (rng.random(len(dates)) > 0.03)
            if intl:
                covid_gap = (dates >= "2020-04-01") & (dates <= "2020-09-01") & (rng.random(len(dates)) < 0.7)
                keep &= ~covid_gap
            fare = (60 + 0.12 * dist) * (1.3 if intl else 1.0)
            for d, v in zip(dates[keep], pax[keep]):
                rows.append((d.month, d.year, codes[i], codes[j], round(float(v), 2),
                             round(float(v * fare * np.exp(rng.normal(0, 0.15))), 2)))

    df = pd.DataFrame(rows, columns=["DEP_MONTH", "DEP_YEAR", "ORG_AIRPRT", "DES_AIRPRT",
                                     "OD_PAX", "OD_REV_USD"])
    df = df.sample(frac=1, random_state=SEED).reset_index(drop=True)  # export order is arbitrary
    df["SF_LOAD_TMS"] = LOAD_TMS
    df.insert(0, "Unnamed: 0", np.arange(len(df)))
    return df


def main() -> None:
    rng = np.random.default_rng(SEED)
    OUT.mkdir(parents=True, exist_ok=True)
    apt = make_airports(rng)
    pax = make_pax(rng, apt)
    apt_out = apt.drop(columns=["_lat", "_lng", "_hub"])
    apt_out.insert(0, "Unnamed: 0", np.arange(len(apt_out)))
    pax.to_csv(OUT / "FAKE_PAX_FORECASTING.csv", index=False)
    apt_out.to_csv(OUT / "FAKE_AIRPORT_INFO.csv", index=False)
    print(f"FAKE data written to {OUT}: {len(pax):,} route-months, {len(apt_out)} airports")


if __name__ == "__main__":
    main()
