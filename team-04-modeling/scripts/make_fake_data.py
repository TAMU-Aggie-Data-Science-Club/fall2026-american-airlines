"""Write made-up tables to the git-ignored data/fake/ folder, so the pipeline runs without AA data.

Usage: python team-04-modeling/scripts/make_fake_data.py

Nothing here comes from AA: every value is random. The quirks are planted, so a "finding" on this
data only proves that the code can detect that quirk. It says nothing about the real data.
"""
import json

import numpy as np
import pandas as pd

from config import FAKE_DIR

N_AIRPORTS, N_ROUTES, N_MONTHS = 40, 300, 84


def degrees(rng, n, top, positive, negative):
    value = rng.uniform(0, top, n)
    return {
        "DEG": value.astype(int),
        "MIN": ((value % 1) * 60).astype(int),
        "SEC": rng.integers(0, 60, n),
        "HEMI": rng.choice([positive, negative], n),
    }


def make_airports(rng):
    df = pd.DataFrame({
        "AIRPORT": [f"Z{i:02d}" for i in range(N_AIRPORTS)],
        "NAME": [f"Fake Airport {i}" for i in range(N_AIRPORTS)],
        "COUNTRY": rng.choice(["AA", "BB", "NA"], N_AIRPORTS, p=[0.7, 0.2, 0.1]),  # planted: "NA" is a code, not a gap
        "REGION": rng.choice(["R1", "R2", "R3", "R4"], N_AIRPORTS),
        "RUNWAY_FT": rng.integers(5000, 14000, N_AIRPORTS),
        "ELEVATION_FT": rng.integers(0, 6000, N_AIRPORTS),
        "ROW_ID": range(N_AIRPORTS),
        "LOADED_AT": "2000-01-01 00:00:00",
    })
    for axis, top, positive, negative in [("LAT", 70, "N", "S"), ("LON", 170, "E", "W")]:
        for part, values in degrees(rng, N_AIRPORTS, top, positive, negative).items():
            df[f"{axis}_{part}"] = values
    df["STATE"] = np.where(df["COUNTRY"] == "AA", "S1", None)  # planted: missing outside one country
    return df


def make_pax(rng, airports):
    months = pd.date_range("2010-01-01", periods=N_MONTHS, freq="MS")
    codes = airports["AIRPORT"].to_numpy()
    pairs = set()
    while len(pairs) < N_ROUTES:
        origin, dest = rng.choice(codes, 2, replace=False)
        pairs.add((origin, dest))

    frames = []
    t = np.arange(N_MONTHS)
    for origin, dest in sorted(pairs):
        base = rng.lognormal(7, 1.2)
        season = 1 + 0.25 * np.sin(2 * np.pi * (t % 12) / 12 + rng.uniform(-0.3, 0.3))
        shock = np.where((t >= 20) & (t < 32), 0.35, 1.0)  # planted: a demand shock
        pax = base * season * shock * (1 + 0.003 * t) * rng.lognormal(0, 0.08, N_MONTHS)
        skip = 0.25 if base < 500 else 0.01  # planted: thin routes skip more months
        frames.append(pd.DataFrame({"ORIGIN": origin, "DEST": dest, "YEAR": months.year, "MONTH": months.month, "PAX": pax.round()})[rng.random(N_MONTHS) > skip])
    df = pd.concat(frames, ignore_index=True)

    df["REVENUE"] = (df["PAX"] * rng.lognormal(5, 0.1, len(df))).round(2)  # planted leak: computed from the target
    outliers = rng.choice(df.index, 25, replace=False)
    df.loc[outliers, "PAX"] *= 40  # planted: outliers
    df = pd.concat([df, df.sample(30, random_state=0)], ignore_index=True)  # planted: exact duplicate rows
    df["ROW_ID"] = range(len(df))  # planted: an export row number that hides the duplicates
    df["LOADED_AT"] = "2000-01-01 00:00:00"
    return df


def main():
    rng = np.random.default_rng(4)
    FAKE_DIR.mkdir(parents=True, exist_ok=True)
    airports = make_airports(rng)
    pax = make_pax(rng, airports)
    airports.iloc[:-1].to_csv(FAKE_DIR / "airports.csv", index=False)  # planted: one airport missing from the table
    pax.to_csv(FAKE_DIR / "pax.csv", index=False)
    same = lambda cols: {col: col for col in cols}
    (FAKE_DIR / "columns.json").write_text(json.dumps({
        "pax_file": "pax.csv",
        "airports_file": "airports.csv",
        "pax": same(["YEAR", "MONTH", "ORIGIN", "DEST", "PAX", "REVENUE"]),
        "airports": same(["AIRPORT", "COUNTRY", "REGION", "RUNWAY_FT", "ELEVATION_FT"]
                         + [f"{axis}_{part}" for axis in ("LAT", "LON") for part in ("DEG", "MIN", "SEC", "HEMI")]),
        "drop": ["ROW_ID", "LOADED_AT"],
    }, indent=2))
    print(f"FAKE data written to {FAKE_DIR}: {len(pax):,} rows, {len(airports)} airports")


if __name__ == "__main__":
    main()
