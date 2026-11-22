"""What do the business indicators in the AA passenger data look like?

One self-contained script. It audits data quality, enriches the passenger file
with airport attributes, and writes the tables and charts a forecasting team
needs before training: volume / revenue / fare trends, seasonality,
concentration, segments, route life cycle, directional balance, fare vs
distance, persistence, and a seasonal-naive baseline to beat.

Usage
-----
Real data (run locally; outputs must stay out of Git):
    python eda_business_indicators.py --pax-csv data/AA_PAX_FORECASTING.csv \
        --airport-csv data/AA_AIRPORT_INFO.csv --out outputs/eda

Synthetic demo + self-check (fake data with planted patterns, no network):
    python eda_business_indicators.py --synthetic --out outputs/eda_demo

Method notes
------------
* Grain is one row per month x origin x destination (directional). Rows that
  are exact copies are dropped; rows that repeat a market-month with different
  values are summed. Both counts are reported in summary.md.
* Average fare = revenue / passengers, computed on totals (never an average of
  route-level fares), so it is passenger-weighted.
* Year-over-year changes are 12-month log changes. Target months from
  Mar 2020 to Dec 2022 are masked by default (COVID, plus the year after it
  whose base is a COVID month). Use --keep-covid to keep them.
* "Trailing 12 months" (TTM) = the latest 12 months in the file. Annual
  tables flag partial years instead of comparing them with full ones.
* A route with a missing month is treated as zero traffic in that month for
  the baseline. Whether a gap means "no service" or "no data" is a question
  for the data owner; the lifecycle section tells you how much it matters.
* The seasonal-naive baseline (same month last year) is the number any model
  has to beat.
* SF_LOAD_TMS looks like a load timestamp, not a load factor, and is ignored.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from aaforecast.geo import BUSINESS_US_CA as US_CA  # descriptive grouping; scored domestic excludes Canada
COVID_START, COVID_END = "2020-03-01", "2022-12-01"
KEY = ["ORG_AIRPRT", "DES_AIRPRT", "DEP_YEAR", "DEP_MONTH"]
REQUIRED = KEY + ["OD_PAX", "OD_REV_USD"]
GEO_COLS = ["LAT_DEGREE_QTY", "LAT_MINUTE_QTY", "LAT_SECOND_QTY", "LAT_HEMSPHR_CD",
            "LNGTD_DEGREE_QTY", "LNGTD_MINUTE_QTY", "LNGTD_SECOND_QTY", "LNGTD_HEMSPHR_CD"]
BAND_EDGES = [0, 500, 1000, 2000, 3000, 5000, np.inf]
BAND_LABELS = ["<500 mi", "500-1000", "1000-2000", "2000-3000", "3000-5000", "5000+ mi"]
MONTH_NAMES = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
               "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]


# ---------------------------------------------------------------------------
# Loading, cleaning, enrichment
# ---------------------------------------------------------------------------
def load(pax_csv: str, airport_csv: str):
    df = pd.read_csv(pax_csv, keep_default_na=False, na_values=[""])
    ap = pd.read_csv(airport_csv, keep_default_na=False, na_values=[""])
    df.columns = [c.strip().upper() for c in df.columns]
    ap.columns = [c.strip().upper() for c in ap.columns]
    return df, ap


def clean(df: pd.DataFrame):
    missing = [c for c in REQUIRED if c not in df.columns]
    if missing:
        sys.exit(f"Passenger CSV is missing columns: {missing}")
    dq = {"rows_loaded": len(df)}
    dq["exact_duplicate_rows"] = int(df.duplicated().sum())
    df = df.drop_duplicates()
    dq["key_duplicates_summed"] = int(df.duplicated(KEY).sum())
    na = df[["OD_PAX", "OD_REV_USD"]].isna().any(axis=1)
    dq["missing_pax_or_revenue"] = int(na.sum())
    df = df[~na]
    dq["nonpositive_pax"] = int((df["OD_PAX"] <= 0).sum())
    dq["negative_revenue"] = int((df["OD_REV_USD"] < 0).sum())
    dq["zero_revenue_with_pax"] = int(((df["OD_PAX"] > 0) & (df["OD_REV_USD"] == 0)).sum())
    df = df[(df["OD_PAX"] > 0) & (df["OD_REV_USD"] >= 0)]
    df = df.groupby(KEY, as_index=False)[["OD_PAX", "OD_REV_USD"]].sum()
    df["period"] = pd.to_datetime(pd.DataFrame(
        {"year": df["DEP_YEAR"].astype(int), "month": df["DEP_MONTH"].astype(int), "day": 1}))
    df["route"] = df["ORG_AIRPRT"].astype(str) + "-" + df["DES_AIRPRT"].astype(str)
    dq["rows_used"] = len(df)
    full = pd.date_range(df["period"].min(), df["period"].max(), freq="MS")
    dq["missing_months"] = [d.strftime("%Y-%m")
                            for d in full.difference(pd.DatetimeIndex(df["period"].unique()))]
    return df, dq


def dms(deg, minute, sec, hemi):
    dd = deg.astype(float) + minute.astype(float) / 60 + sec.astype(float) / 3600
    return dd.where(~hemi.astype(str).str.upper().isin(["S", "W"]), -dd)


def haversine(lat1, lon1, lat2, lon2):
    r = 3958.8
    p1, p2 = np.radians(lat1), np.radians(lat2)
    a = np.sin((p2 - p1) / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(np.radians(lon2 - lon1) / 2) ** 2
    return 2 * r * np.arcsin(np.sqrt(a))


def enrich(df: pd.DataFrame, ap: pd.DataFrame, dq: dict) -> pd.DataFrame:
    dq["airport_lookup_duplicates"] = int(ap.duplicated("AIRPRT_CD").sum())
    ap = ap.drop_duplicates("AIRPRT_CD").set_index("AIRPRT_CD")
    geo = all(c in ap.columns for c in GEO_COLS)
    if geo:
        ap["LAT"] = dms(ap["LAT_DEGREE_QTY"], ap["LAT_MINUTE_QTY"], ap["LAT_SECOND_QTY"], ap["LAT_HEMSPHR_CD"])
        ap["LON"] = dms(ap["LNGTD_DEGREE_QTY"], ap["LNGTD_MINUTE_QTY"], ap["LNGTD_SECOND_QTY"], ap["LNGTD_HEMSPHR_CD"])
    for side, col in (("ORG", "ORG_AIRPRT"), ("DES", "DES_AIRPRT")):
        df[f"{side}_CNTRY"] = df[col].map(ap["CNTRY_CD"])
        if geo:
            df[f"{side}_LAT"] = df[col].map(ap["LAT"])
            df[f"{side}_LON"] = df[col].map(ap["LON"])
    codes = set(df["ORG_AIRPRT"]) | set(df["DES_AIRPRT"])
    unmapped = sorted(codes - set(ap.index))
    dq["airports_in_data"] = len(codes)
    dq["airports_missing_from_lookup"] = unmapped
    unknown = df["ORG_CNTRY"].isna() | df["DES_CNTRY"].isna()
    dom = df["ORG_CNTRY"].isin(US_CA) & df["DES_CNTRY"].isin(US_CA)
    df["market_type"] = np.where(unknown, "Unknown", np.where(dom, "US/CA", "International"))
    if geo:
        df["distance_mi"] = haversine(df["ORG_LAT"], df["ORG_LON"], df["DES_LAT"], df["DES_LON"])
        df["distance_band"] = pd.cut(df["distance_mi"], BAND_EDGES, labels=BAND_LABELS,
                                     right=False).astype(object).fillna("Unknown")
    else:
        df["distance_mi"] = np.nan
        df["distance_band"] = "Unknown"
        print("  ! airport file has no usable lat/long columns; distance analyses skipped")
    return df


# ---------------------------------------------------------------------------
# Analyses
# ---------------------------------------------------------------------------
def mask_covid(s: pd.Series, keep_covid: bool) -> pd.Series:
    if keep_covid:
        return s
    bad = (s.index >= COVID_START) & (s.index <= COVID_END)
    return s.mask(bad)


def monthly_kpis(df: pd.DataFrame) -> pd.DataFrame:
    m = df.groupby("period")[["OD_PAX", "OD_REV_USD"]].sum().asfreq("MS")
    m["fare"] = m["OD_REV_USD"] / m["OD_PAX"]
    for c in ["OD_PAX", "OD_REV_USD", "fare"]:
        m[f"{c}_yoy"] = np.log(m[c]).diff(12)
    m["active_routes"] = df.groupby("period")["route"].nunique().reindex(m.index)
    return m


def annual_table(df: pd.DataFrame) -> pd.DataFrame:
    a = df.groupby("DEP_YEAR").agg(pax=("OD_PAX", "sum"), revenue=("OD_REV_USD", "sum"),
                                   months=("period", "nunique"))
    a["fare"] = a["revenue"] / a["pax"]
    a["complete"] = a["months"] == 12
    prev_ok = a["complete"] & a["complete"].shift(1, fill_value=False)
    for c in ["pax", "revenue", "fare"]:
        a[f"{c}_growth"] = np.log(a[c] / a[c].shift(1)).where(prev_ok)
    return a


def seasonal_index(m: pd.DataFrame, keep_covid: bool) -> pd.Series:
    s = m["OD_PAX"]
    ma = s.rolling(12).mean().rolling(2).mean().shift(-6)  # centred 2x12 moving average
    ratio = s / ma
    if not keep_covid:
        lo = pd.Timestamp(COVID_START) - pd.DateOffset(months=6)
        hi = pd.Timestamp(COVID_END) + pd.DateOffset(months=6)
        ratio = ratio[(ratio.index < lo) | (ratio.index > hi)]
    idx = ratio.groupby(ratio.index.month).median().reindex(range(1, 13))
    return idx / idx.mean()


def ttm_frame(df: pd.DataFrame) -> pd.DataFrame:
    cut = df["period"].max() - pd.DateOffset(months=11)
    return df[df["period"] >= cut]


def concentration(ttm: pd.DataFrame):
    r = ttm.groupby("route").agg(pax=("OD_PAX", "sum"), revenue=("OD_REV_USD", "sum"))
    r["fare"] = r["revenue"] / r["pax"]
    r = r.sort_values("pax", ascending=False)
    r["pax_share"] = r["pax"] / r["pax"].sum()
    r["cum_pax_share"] = r["pax_share"].cumsum()
    a = (ttm.groupby("ORG_AIRPRT")["OD_PAX"].sum()
         .add(ttm.groupby("DES_AIRPRT")["OD_PAX"].sum(), fill_value=0)
         .sort_values(ascending=False).rename("pax_touching"))
    a = a.to_frame()
    a["share_of_pax"] = a["pax_touching"] / ttm["OD_PAX"].sum()  # each pax touches 2 airports
    return r, a


def segment_table(df: pd.DataFrame) -> pd.DataFrame:
    months = df.groupby("DEP_YEAR")["period"].nunique().rename("months_in_year")
    frames = []
    for d in ["market_type", "distance_band"]:
        g = (df.groupby([d, "DEP_YEAR"])[["OD_PAX", "OD_REV_USD"]].sum().reset_index()
             .rename(columns={d: "value"}))
        g.insert(0, "dimension", d)
        frames.append(g)
    t = pd.concat(frames, ignore_index=True)
    t["fare"] = t["OD_REV_USD"] / t["OD_PAX"]
    t["pax_share"] = t["OD_PAX"] / t.groupby(["dimension", "DEP_YEAR"])["OD_PAX"].transform("sum")
    return t.merge(months, left_on="DEP_YEAR", right_index=True)


def lifecycle(df: pd.DataFrame):
    g = df.groupby("route")["period"].agg(first="min", last="max", n_obs="nunique")
    g["span"] = ((g["last"].dt.year - g["first"].dt.year) * 12
                 + g["last"].dt.month - g["first"].dt.month + 1)
    g["gaps"] = g["span"] - g["n_obs"]
    g["pax"] = df.groupby("route")["OD_PAX"].sum()
    start, end = df["period"].min(), df["period"].max()
    births = g[g["first"] > start].groupby(g["first"].dt.year).size().rename("births")
    deaths = g[g["last"] < end].groupby(g["last"].dt.year).size().rename("deaths")
    yearly = pd.concat([births, deaths], axis=1).fillna(0).astype(int).sort_index()
    return g, yearly


def symmetry(ttm: pd.DataFrame):
    r = ttm.groupby(["ORG_AIRPRT", "DES_AIRPRT"])["OD_PAX"].sum().reset_index()
    fwd = r["ORG_AIRPRT"] < r["DES_AIRPRT"]
    r["pair"] = np.where(fwd, r["ORG_AIRPRT"] + "-" + r["DES_AIRPRT"],
                         r["DES_AIRPRT"] + "-" + r["ORG_AIRPRT"])
    r["dir"] = np.where(fwd, "fwd", "rev")
    piv = r.pivot_table(index="pair", columns="dir", values="OD_PAX", aggfunc="sum")
    for c in ("fwd", "rev"):
        if c not in piv:
            piv[c] = np.nan
    both = piv.dropna()
    imb = (both["fwd"] - both["rev"]).abs() / (both["fwd"] + both["rev"])
    return {"pairs": len(piv), "one_direction_only": int(len(piv) - len(both)),
            "median_imbalance": float(imb.median()) if len(imb) else np.nan,
            "share_over_20pct": float((imb > 0.2).mean()) if len(imb) else np.nan}


def yield_by_band(ttm: pd.DataFrame) -> pd.DataFrame:
    t = ttm[ttm["distance_mi"].notna()].copy()
    t["pax_miles"] = t["OD_PAX"] * t["distance_mi"]
    g = t.groupby("distance_band").agg(pax=("OD_PAX", "sum"), revenue=("OD_REV_USD", "sum"),
                                       pax_miles=("pax_miles", "sum"))
    g["fare"] = g["revenue"] / g["pax"]
    g["yield_per_mile"] = g["revenue"] / g["pax_miles"]
    return g.reindex([b for b in BAND_LABELS if b in g.index])


def acf(x: np.ndarray, nlags: int) -> np.ndarray:
    x = x - x.mean()
    denom = (x ** 2).sum()
    return np.array([(x[k:] * x[:len(x) - k]).sum() / denom for k in range(nlags + 1)])


def clean_run_acf(m: pd.DataFrame, keep_covid: bool):
    """ACF of YoY log passenger change on the longest run free of NaN / COVID."""
    s = mask_covid(m["OD_PAX_yoy"], keep_covid)
    ok = s.notna().values
    best, i = (0, 0), 0
    while i < len(ok):
        if ok[i]:
            j = i
            while j < len(ok) and ok[j]:
                j += 1
            if j - i > best[1] - best[0]:
                best = (i, j)
            i = j
        else:
            i += 1
    seg = s.values[best[0]:best[1]]
    if len(seg) < 36:
        return None
    nlags = min(24, len(seg) // 3)
    return acf(seg, nlags), len(seg)


def seasonal_naive(df: pd.DataFrame, months: int = 12):
    last = df["period"].max()
    h0 = last - pd.DateOffset(months=months - 1)
    if df["period"].min() > h0 - pd.DateOffset(years=1):
        return None
    hold = df[df["period"] >= h0][["route", "period", "OD_PAX"]]
    prev = df[(df["period"] >= h0 - pd.DateOffset(years=1)) &
              (df["period"] <= last - pd.DateOffset(years=1))][["route", "period", "OD_PAX"]].copy()
    prev["period"] = prev["period"] + pd.DateOffset(years=1)
    m = hold.merge(prev, on=["route", "period"], how="outer", suffixes=("", "_fc"))
    m["actual"] = m["OD_PAX"].fillna(0)
    m["fc"] = m["OD_PAX_fc"].fillna(0)
    mt = df.drop_duplicates("route").set_index("route")["market_type"]
    m["market_type"] = m["route"].map(mt)
    cs, ce = pd.Timestamp(COVID_START), pd.Timestamp(COVID_END)
    overlap = (h0 - pd.DateOffset(years=1) <= ce) and (last >= cs)

    def wape(x):
        return float((x["actual"] - x["fc"]).abs().sum() / x["actual"].sum())

    tot = m.groupby("period")[["actual", "fc"]].sum()
    agg_mape = float(((tot["actual"] - tot["fc"]).abs() / tot["actual"]).mean())
    out = {"window": f"{h0:%Y-%m} to {last:%Y-%m}", "route_wape": wape(m),
           "aggregate_mape": agg_mape, "covid_overlap": overlap,
           "covered_share": float(m.loc[m["OD_PAX_fc"].notna(), "actual"].sum() / m["actual"].sum())}
    for k, g in m.groupby("market_type"):
        out[f"route_wape_{k}"] = wape(g)
    return out


# ---------------------------------------------------------------------------
# Charts
# ---------------------------------------------------------------------------
def shade(ax, keep_covid: bool):
    if not keep_covid:
        ax.axvspan(pd.Timestamp(COVID_START), pd.Timestamp(COVID_END), color="grey", alpha=0.15, lw=0)


def plot_totals(m, keep_covid, path):
    fig, axes = plt.subplots(3, 1, figsize=(10, 7), sharex=True)
    axes[0].plot(m.index, m["OD_PAX"] / 1e6, color="tab:blue")
    axes[0].set_ylabel("Passengers (M)")
    axes[1].plot(m.index, m["OD_REV_USD"] / 1e6, color="tab:green")
    axes[1].set_ylabel("Revenue ($M)")
    axes[2].plot(m.index, m["fare"], color="tab:red")
    axes[2].set_ylabel("Avg fare ($)")
    for ax in axes:
        shade(ax, keep_covid)
        ax.grid(alpha=0.3)
    axes[0].set_title("Monthly totals (grey = excluded from YoY)" if not keep_covid else "Monthly totals")
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def plot_yoy(m, keep_covid, path):
    fig, ax = plt.subplots(figsize=(10, 4))
    for c, lab, col in (("OD_PAX_yoy", "Passengers", "tab:blue"), ("fare_yoy", "Avg fare", "tab:red"),
                        ("OD_REV_USD_yoy", "Revenue", "tab:green")):
        ax.plot(m.index, mask_covid(m[c], keep_covid) * 100, label=lab, color=col)
    shade(ax, keep_covid)
    ax.axhline(0, color="black", lw=0.8)
    ax.set_ylabel("YoY change (log %, ~ %)")
    ax.set_title("Year-over-year change")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def plot_seasonality(idx, path):
    fig, ax = plt.subplots(figsize=(8, 3.5))
    ax.bar(MONTH_NAMES, idx.values, color="tab:blue")
    ax.axhline(1, color="black", lw=0.8)
    ax.set_ylim(min(0.7, idx.min() * 0.95), max(1.3, idx.max() * 1.05))
    ax.set_title("Seasonal index of passengers (1.0 = average month)")
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def plot_concentration(routes, airports, path):
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8))
    axes[0].plot(np.arange(1, len(routes) + 1), routes["cum_pax_share"].values * 100)
    axes[0].set_xlabel("Routes, ranked by passengers")
    axes[0].set_ylabel("Cumulative share of passengers (%)")
    axes[0].set_title("Route concentration (TTM)")
    top = airports.head(15)
    axes[1].barh(top.index[::-1], top["share_of_pax"].values[::-1] * 100, color="tab:blue")
    axes[1].set_xlabel("% of passengers touching airport (origin + destination)")
    axes[1].set_title("Top airports (TTM)")
    for ax in axes:
        ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def plot_segments(df, yb, keep_covid, path):
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.8))
    pv = df.groupby(["period", "market_type"])["OD_PAX"].sum().unstack() / 1e6
    for c in pv.columns:
        axes[0].plot(pv.index, pv[c], label=c)
    shade(axes[0], keep_covid)
    axes[0].set_ylabel("Passengers (M)")
    axes[0].set_title("Passengers by market type")
    axes[0].legend()
    axes[0].grid(alpha=0.3)
    if len(yb):
        axes[1].bar(yb.index, yb["yield_per_mile"].values * 100, color="tab:orange")
        axes[1].set_ylabel("Revenue per passenger-mile (cents)")
        axes[1].set_title("Yield by distance band (TTM)")
        axes[1].tick_params(axis="x", rotation=30)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def plot_lifecycle(m, yearly, path):
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.8))
    axes[0].plot(m.index, m["active_routes"])
    axes[0].set_title("Routes with traffic, by month")
    axes[0].grid(alpha=0.3)
    if len(yearly):
        x = np.arange(len(yearly))
        axes[1].bar(x - 0.2, yearly.get("births", 0), 0.4, label="first seen")
        axes[1].bar(x + 0.2, yearly.get("deaths", 0), 0.4, label="last seen")
        axes[1].set_xticks(x, yearly.index.astype(str))
        axes[1].legend()
    axes[1].set_title("Route entries / exits by year")
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def plot_growth(annual, path):
    a = annual.dropna(subset=["pax_growth"])
    if a.empty:
        return
    fig, ax = plt.subplots(figsize=(9, 3.8))
    x = np.arange(len(a))
    ax.bar(x - 0.2, a["pax_growth"] * 100, 0.4, label="Passengers")
    ax.bar(x + 0.2, a["fare_growth"] * 100, 0.4, label="Avg fare")
    ax.plot(x, a["revenue_growth"] * 100, color="black", marker="o", label="Revenue")
    ax.set_xticks(x, a.index.astype(str))
    ax.axhline(0, color="black", lw=0.8)
    ax.set_ylabel("Log change (~ %)")
    ax.set_title("Revenue growth = passenger growth + fare growth (complete years only)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def plot_acf(res, path):
    if res is None:
        return
    a, n = res
    fig, ax = plt.subplots(figsize=(8, 3.5))
    ax.bar(range(len(a)), a, color="tab:blue")
    ci = 1.96 / np.sqrt(n)
    ax.axhline(ci, color="red", ls="--", lw=0.8)
    ax.axhline(-ci, color="red", ls="--", lw=0.8)
    ax.set_xlabel("lag (months)")
    ax.set_title(f"Autocorrelation of YoY passenger change ({n} clean months)")
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
def pct(x, d=1):
    return "n/a" if x is None or pd.isna(x) else f"{x * 100:.{d}f}%"


def write_summary(R: dict, path: Path, synthetic: bool):
    dq, m = R["dq"], R["monthly"]
    L = ["# EDA: business indicators in the AA passenger data", ""]
    if synthetic:
        L += ["**SYNTHETIC DATA.** Everything below comes from generated data and says "
              "nothing about American Airlines.", ""]
    L += ["## Key findings", ""]
    t = R["ttm"]
    L.append(f"- **Scale (TTM {R['ttm_start']:%Y-%m} to {R['last']:%Y-%m}):** "
             f"{t['OD_PAX'].sum() / 1e6:,.1f}M passengers, ${t['OD_REV_USD'].sum() / 1e9:,.2f}B revenue, "
             f"average fare ${t['OD_REV_USD'].sum() / t['OD_PAX'].sum():,.0f}.")
    ann = R["annual"]
    cy = ann[ann["pax_growth"].notna()]
    if len(cy):
        y = cy.index[-1]
        r = cy.loc[y]
        L.append(f"- **Latest complete-year growth ({y} vs {y - 1}):** passengers {r['pax_growth'] * 100:+.1f}%, "
                 f"fare {r['fare_growth'] * 100:+.1f}%, revenue {r['revenue_growth'] * 100:+.1f}% (log changes).")
    partial = ann[~ann["complete"]]
    if len(partial):
        L.append("- **Partial years (excluded from year-over-year comparisons):** "
                 + ", ".join(f"{y} ({int(v)} months)" for y, v in partial["months"].items()) + ".")
    si = R["seasonal"]
    L.append(f"- **Seasonality:** peak {MONTH_NAMES[int(si.idxmax()) - 1]} (index {si.max():.2f}), "
             f"trough {MONTH_NAMES[int(si.idxmin()) - 1]} (index {si.min():.2f}); "
             "the peak-to-trough swing is " + f"{(si.max() / si.min() - 1) * 100:.0f}%.")
    rt = R["routes"]
    n80 = int((rt["cum_pax_share"] < 0.8).sum() + 1)
    L.append(f"- **Concentration:** the top 10 routes carry {pct(rt['pax_share'].head(10).sum())} of TTM passengers "
             f"and the top 50 carry {pct(rt['pax_share'].head(50).sum())}; "
             f"{n80:,} of {len(rt):,} routes ({pct(n80 / len(rt))}) make up 80%.")
    seg = R["segments"]
    mt = seg[(seg["dimension"] == "market_type") & (seg["DEP_YEAR"] == R["last"].year)]
    if len(mt):
        L.append("- **Market mix (latest year in file):** " + ", ".join(
            f"{r.value} {pct(r.pax_share)} of pax at ${r.fare:,.0f} avg fare" for r in mt.itertuples()) + ".")
    lc = R["lifecycle"]
    gap = lc[lc["gaps"] > 0]
    L.append(f"- **Route stability:** {len(gap):,} of {len(lc):,} routes have gaps in their history, "
             f"covering {pct(gap['pax'].sum() / lc['pax'].sum())} of all passengers; "
             f"{int((lc['last'] < R['last']).sum()):,} routes stopped before the end of the file and "
             f"{int((lc['first'] > R['first']).sum()):,} started after the beginning.")
    sy = R["symmetry"]
    L.append(f"- **Directional balance:** {sy['one_direction_only']:,} of {sy['pairs']:,} city pairs fly in one direction only; "
             f"among two-way pairs the median imbalance is {pct(sy['median_imbalance'])} "
             f"and {pct(sy['share_over_20pct'])} differ by more than 20%.")
    yb = R["yield"]
    if len(yb):
        L.append(f"- **Yield vs distance:** revenue per passenger-mile runs from {yb['yield_per_mile'].iloc[0] * 100:.1f}c "
                 f"({yb.index[0]}) to {yb['yield_per_mile'].iloc[-1] * 100:.1f}c ({yb.index[-1]}).")
    b = R["baseline"]
    if b:
        L.append(f"- **Baseline to beat (seasonal naive, {b['window']}):** route-level WAPE {pct(b['route_wape'])}, "
                 f"aggregate monthly MAPE {pct(b['aggregate_mape'])}"
                 + (" (window touches COVID, so treat as pessimistic)" if b["covid_overlap"] else "") + ".")
    L += ["", "## Data quality", ""]
    L += [f"- Rows loaded: {dq['rows_loaded']:,}; rows used after cleaning and aggregation: {dq['rows_used']:,}",
          f"- Exact duplicate rows dropped: {dq['exact_duplicate_rows']:,}",
          f"- Repeated market-months with different values (summed): {dq['key_duplicates_summed']:,}",
          f"- Missing pax or revenue (dropped): {dq['missing_pax_or_revenue']:,}",
          f"- Non-positive pax (dropped): {dq['nonpositive_pax']:,}; negative revenue (dropped): {dq['negative_revenue']:,}",
          f"- Zero revenue with passengers (kept): {dq['zero_revenue_with_pax']:,}",
          f"- Months missing from the date range: {', '.join(dq['missing_months']) or 'none'}",
          f"- Duplicate airport codes in lookup: {dq['airport_lookup_duplicates']:,}",
          f"- Airports in data / missing from lookup: {dq['airports_in_data']:,} / {len(dq['airports_missing_from_lookup']):,}"
          + (f" ({', '.join(dq['airports_missing_from_lookup'][:10])})" if dq["airports_missing_from_lookup"] else "")]
    L += ["", "## Annual totals", "", "| year | months | passengers (M) | revenue ($M) | avg fare | complete |",
          "|---|---|---|---|---|---|"]
    L += [f"| {y} | {int(r.months)} | {r.pax / 1e6:,.2f} | {r.revenue / 1e6:,.1f} | {r.fare:,.0f} | {'yes' if r.complete else 'no'} |"
          for y, r in ann.iterrows()]
    L += ["", "## Top 10 routes (TTM)", "", "| route | passengers | share | avg fare |", "|---|---|---|---|"]
    L += [f"| {i} | {r.pax:,.0f} | {pct(r.pax_share)} | {r.fare:,.0f} |" for i, r in rt.head(10).iterrows()]
    L += ["", "## Reading this", "",
          "- This is one airline's O&D data: no market share, and no capacity or load factor.",
          "- Fare is revenue / passengers; mix shifts (more long-haul, more international) move it without any price change.",
          "- Gaps are treated as zero traffic in the baseline; confirm with the data owner whether they mean no service or no data.",
          "- The US/CA grouping is a business definition (DOT counts US-Canada as international). Adjust US_CA at the top of the script.",
          "- Figures and CSV tables are written next to this file. Keep outputs out of Git.", ""]
    path.write_text("\n".join(L), encoding="utf-8")


# ---------------------------------------------------------------------------
# Synthetic data (doubles as the test)
# ---------------------------------------------------------------------------
AIRPORTS = {  # code: (country, lat deg, lat min, N/S, lon deg, lon min, E/W)
    "DFW": ("US", 32, 53, "N", 97, 2, "W"), "ORD": ("US", 41, 58, "N", 87, 54, "W"),
    "LAX": ("US", 33, 56, "N", 118, 24, "W"), "MIA": ("US", 25, 47, "N", 80, 17, "W"),
    "PHX": ("US", 33, 26, "N", 112, 0, "W"), "CLT": ("US", 35, 13, "N", 80, 56, "W"),
    "LHR": ("GB", 51, 28, "N", 0, 27, "W"), "MEX": ("MX", 19, 26, "N", 99, 4, "W"),
    "NRT": ("JP", 35, 46, "N", 140, 23, "E"), "CDG": ("FR", 49, 0, "N", 2, 33, "E"),
}


def make_synthetic(seed: int = 7):
    rng = np.random.default_rng(seed)
    months = pd.date_range("2014-01-01", "2025-06-01", freq="MS")
    ap = pd.DataFrame([{
        "AIRPRT_CD": k, "CNTRY_CD": v[0], "STATE_PROVNC_CD": None, "WRLD_AREA_DOT_CD": 1,
        "LAT_DEGREE_QTY": v[1], "LAT_MINUTE_QTY": v[2], "LAT_SECOND_QTY": 0, "LAT_HEMSPHR_CD": v[3],
        "LNGTD_DEGREE_QTY": v[4], "LNGTD_MINUTE_QTY": v[5], "LNGTD_SECOND_QTY": 0, "LNGTD_HEMSPHR_CD": v[6],
    } for k, v in AIRPORTS.items()])
    ap["LAT"] = dms(ap["LAT_DEGREE_QTY"], ap["LAT_MINUTE_QTY"], ap["LAT_SECOND_QTY"], ap["LAT_HEMSPHR_CD"])
    ap["LON"] = dms(ap["LNGTD_DEGREE_QTY"], ap["LNGTD_MINUTE_QTY"], ap["LNGTD_SECOND_QTY"], ap["LNGTD_HEMSPHR_CD"])
    coords = ap.set_index("AIRPRT_CD")[["LAT", "LON"]]
    us = ["DFW", "ORD", "LAX", "MIA", "PHX", "CLT"]
    routes = [("DFW", "ORD"), ("ORD", "DFW")]
    routes += [(o, d) for o in us for d in us if o != d and (o, d) not in routes]
    for a, b in [("DFW", "LHR"), ("MIA", "MEX"), ("DFW", "NRT"), ("CLT", "CDG"), ("ORD", "LHR")]:
        routes += [(a, b), (b, a)]
    n_us = 30
    order = [0, 1] + list(rng.permutation(np.arange(2, len(routes))))
    base_pax = {routes[i]: 40000 / (rank + 1) ** 0.8 for rank, i in enumerate(order)}
    base_pax[("ORD", "DFW")] *= 0.9
    late = {routes[n_us + 0]: "2018-01-01", routes[n_us + 2]: "2018-01-01", routes[n_us + 4]: "2022-01-01"}
    early_end = {routes[n_us + 1]: "2019-06-01", routes[n_us + 3]: "2019-06-01"}
    gap_routes = {routes[5]: ("2021-01-01", "2021-06-01"), routes[9]: ("2021-01-01", "2021-06-01"),
                  routes[13]: ("2021-01-01", "2021-06-01")}

    def covid(ts):
        t0, t1, t2 = pd.Timestamp("2020-03-01"), pd.Timestamp("2020-05-01"), pd.Timestamp("2022-06-01")
        if ts < t0 or ts >= t2:
            return 1.0
        if ts < t1:
            return 1 - 0.85 * (ts - t0).days / (t1 - t0).days
        return 0.15 + 0.85 * (ts - t1).days / (t2 - t1).days

    yrs = np.array([(t - months[0]).days / 365.25 for t in months])
    season = 1 + 0.2 * np.cos(2 * np.pi * (months.month.values - 7) / 12)
    cv = np.array([covid(t) for t in months])
    rows = []
    for o, d in routes:
        dist = float(haversine(coords.loc[o, "LAT"], coords.loc[o, "LON"], coords.loc[d, "LAT"], coords.loc[d, "LON"]))
        pax = base_pax[(o, d)] * season * 1.03 ** yrs * cv * np.exp(rng.normal(0, 0.05, len(months)))
        fare = (60 + 0.11 * dist) * 1.02 ** yrs * np.exp(rng.normal(0, 0.03, len(months)))
        keep = np.ones(len(months), bool)
        if (o, d) in late:
            keep &= months >= pd.Timestamp(late[(o, d)])
        if (o, d) in early_end:
            keep &= months <= pd.Timestamp(early_end[(o, d)])
        if (o, d) in gap_routes:
            g0, g1 = gap_routes[(o, d)]
            keep &= ~((months >= pd.Timestamp(g0)) & (months <= pd.Timestamp(g1)))
        for i in np.where(keep)[0]:
            p = max(1, int(round(pax[i])))
            rows.append((o, d, months[i].year, months[i].month, p, p * fare[i]))
    df = pd.DataFrame(rows, columns=["ORG_AIRPRT", "DES_AIRPRT", "DEP_YEAR", "DEP_MONTH", "OD_PAX", "OD_REV_USD"])
    pick = rng.choice(len(df), 25, replace=False)
    exact = df.iloc[pick[:20]]
    keyd = df.iloc[pick[20:]].copy()
    keyd["OD_PAX"] += 1
    df = pd.concat([df, exact, keyd]).sample(frac=1, random_state=seed).reset_index(drop=True)
    return df, ap.drop(columns=["LAT", "LON"])


def self_check(R: dict) -> bool:
    checks = []
    dq = R["dq"]
    checks.append(("exact duplicates found (20 planted)", dq["exact_duplicate_rows"] == 20))
    checks.append(("repeated market-months found (5 planted)", dq["key_duplicates_summed"] == 5))
    checks.append(("seasonal peak is July, trough is January",
                   int(R["seasonal"].idxmax()) == 7 and int(R["seasonal"].idxmin()) == 1))
    checks.append(("largest route is DFW-ORD", R["routes"].index[0] == "DFW-ORD"))
    lc = R["lifecycle"]
    checks.append(("3 routes with gaps (planted)", int((lc["gaps"] > 0).sum()) == 3))
    checks.append(("3 late starts and 2 early exits found",
                   int((lc["first"] > R["first"]).sum()) == 3 and int((lc["last"] < R["last"]).sum()) == 2))
    ann = R["annual"]
    checks.append(("2025 flagged as partial year", not bool(ann.loc[2025, "complete"])))
    d = R["dfw_ord_distance"]
    checks.append((f"DFW-ORD distance ~802 mi (got {d:.0f})", 770 < d < 840))
    yb = R["yield"]
    checks.append(("yield per mile falls with distance",
                   len(yb) > 2 and yb["yield_per_mile"].iloc[0] > yb["yield_per_mile"].iloc[-1]))
    m = R["monthly"]
    checks.append(("COVID drop visible (YoY passengers < -30%)", m["OD_PAX_yoy"].min() < -0.3))
    b = R["baseline"]
    checks.append(("seasonal-naive baseline computed and sensible (WAPE < 25%)",
                   b is not None and b["route_wape"] < 0.25))
    print("\nSelf-check:")
    for name, ok in checks:
        print(f"  [{'PASS' if ok else 'FAIL'}] {name}")
    return all(ok for _, ok in checks)


# ---------------------------------------------------------------------------
def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--pax-csv")
    p.add_argument("--airport-csv")
    p.add_argument("--out", default="outputs/eda")
    p.add_argument("--keep-covid", action="store_true", help="keep Mar 2020 - Dec 2022 in YoY and seasonality")
    p.add_argument("--synthetic", action="store_true", help="use generated data with planted patterns")
    args = p.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    if args.synthetic:
        df, ap = make_synthetic()
    else:
        if not (args.pax_csv and args.airport_csv):
            p.error("--pax-csv and --airport-csv are required unless --synthetic")
        df, ap = load(args.pax_csv, args.airport_csv)

    df, dq = clean(df)
    df = enrich(df, ap, dq)
    print(f"Rows used: {len(df):,} | {df['period'].min():%Y-%m} to {df['period'].max():%Y-%m}")

    m = monthly_kpis(df)
    ttm = ttm_frame(df)
    routes, airports = concentration(ttm)
    lc, yearly = lifecycle(df)
    seg = segment_table(df)
    yb = yield_by_band(ttm)
    ann = annual_table(df)
    si = seasonal_index(m, args.keep_covid)
    acf_res = clean_run_acf(m, args.keep_covid)
    dfw = df[df["route"] == "DFW-ORD"]["distance_mi"]
    R = {"dq": dq, "monthly": m, "annual": ann, "seasonal": si, "ttm": ttm, "routes": routes,
         "segments": seg, "lifecycle": lc, "symmetry": symmetry(ttm), "yield": yb,
         "baseline": seasonal_naive(df), "first": df["period"].min(), "last": df["period"].max(),
         "ttm_start": ttm["period"].min(),
         "dfw_ord_distance": float(dfw.iloc[0]) if len(dfw) else float("nan")}

    m.to_csv(out / "monthly_kpis.csv")
    ann.to_csv(out / "annual_kpis.csv")
    si.rename("seasonal_index").to_csv(out / "seasonal_index.csv")
    routes.to_csv(out / "route_concentration_ttm.csv")
    airports.to_csv(out / "airport_concentration_ttm.csv")
    seg.to_csv(out / "segments_by_year.csv", index=False)
    lc.to_csv(out / "route_lifecycle.csv")
    yb.to_csv(out / "yield_by_distance_band.csv")
    plot_totals(m, args.keep_covid, out / "totals.png")
    plot_yoy(m, args.keep_covid, out / "yoy.png")
    plot_seasonality(si, out / "seasonality.png")
    plot_concentration(routes, airports, out / "concentration.png")
    plot_segments(df, yb, args.keep_covid, out / "segments.png")
    plot_lifecycle(m, yearly, out / "route_lifecycle.png")
    plot_growth(ann, out / "growth_decomposition.png")
    plot_acf(acf_res, out / "acf_yoy_pax.png")
    write_summary(R, out / "summary.md", args.synthetic)
    print(f"Outputs in {out.resolve()}")

    if args.synthetic:
        sys.exit(0 if self_check(R) else 1)


if __name__ == "__main__":
    main()
