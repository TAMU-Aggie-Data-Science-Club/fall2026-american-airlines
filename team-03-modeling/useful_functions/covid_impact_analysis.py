"""How did COVID-19 hit AA's passengers, fares and revenue, and what should we do about it?

One self-contained script. It reads the same two files as the earlier scripts
(AA_PAX_FORECASTING_DATA.csv and AIRPORT_INFO_AA_DATA.csv), builds monthly
network / domestic / international series, and measures the COVID shock, the
recovery, and what changed permanently.

Usage
-----
Real data (run locally; outputs must stay out of Git):
    python covid_impact_analysis.py --pax-csv data/AA_PAX_FORECASTING_DATA.csv \
        --airport-csv data/AIRPORT_INFO_AA_DATA.csv --out outputs/covid

Synthetic demo + self-check (fake AA data with a planted COVID shock, no network):
    python covid_impact_analysis.py --synthetic --out outputs/covid_demo

What it answers
---------------
1. How deep was the drop, and when?            (trough, % of same month in 2019)
2. How long did recovery take?                 (months until >= 100% for 3 months running)
3. How much was lost in total?                 (cumulative passenger and revenue shortfall)
4. Was it volume or price?                     (revenue bridge: volume effect vs price effect)
5. What happened to fares for the SAME routes? (fixed-weight / Laspeyres fare index)
6. Who recovered and who did not?              (domestic vs international, markets, origin airports)
7. Did the seasonal shape of demand change?    (monthly share of annual passengers)
8. Did cheaper routes bounce back faster?      (route-level pax recovery vs fare change)

Method notes
------------
* Baseline = calendar year 2019. Every month is compared with the SAME calendar
  month of 2019, so normal seasonality does not look like recovery.
* COVID window = Mar 2020 - Dec 2022 (same window as macro_price_analysis.py).
* Average fare (revenue / passengers) is distorted by mix: when long-haul or
  premium routes vanish, the average moves even if no ticket price changed.
  The fare index therefore holds route weights fixed at 2019 levels.
* Missing market-months are treated as zero passengers (the route was not
  flown / sold). A genuine hole in the extract would look the same, so check
  the data-quality printout at the top of the run.
* The data is origin-destination (O&D) ticketing data. It describes demand for
  itineraries, not seats flown, so load factor and capacity are NOT observable
  here. Pair this with capacity (ASM) data before making capacity decisions.
* Correlation is not causation. The route-level fare/demand relationship below
  is descriptive, not a price-elasticity estimate.

Only local files are read. Nothing leaves the machine.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # no display needed; we only save PNGs
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy import stats  # noqa: E402

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
# Split by COUNTRY, not by state/province (AU, BR, IN, AR airports also have
# states). This descriptive grouping includes Canada. The scored forecast does not.
from aaforecast.geo import BUSINESS_US_CA as US_CA
COVID_START = pd.Timestamp("2020-03-01")
COVID_END = pd.Timestamp("2022-12-01")
BASE_YEAR = 2019              # pre-COVID benchmark year
SUSTAIN_MONTHS = 3            # months in a row at >=100% to call it "recovered"
MIN_BASE_PAX = 1000           # ignore tiny markets (annual 2019 pax) in market stats
SEGMENTS = ["ALL", "DOMESTIC", "INTERNATIONAL"]
NEED_COLS = ["ORG_AIRPRT", "DES_AIRPRT", "DEP_YEAR", "DEP_MONTH", "OD_PAX", "OD_REV_USD"]


# ---------------------------------------------------------------------------
# 1. Load + clean: AA CSV -> one row per (market, month)
# ---------------------------------------------------------------------------
def load_data(pax_csv: str, airport_csv: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    return (pd.read_csv(pax_csv, keep_default_na=False, na_values=[""]),
            pd.read_csv(airport_csv, keep_default_na=False, na_values=[""]))


def build_market_months(df: pd.DataFrame, airports: pd.DataFrame) -> pd.DataFrame:
    """Return one row per (market, month) with pax, rev and a segment label."""
    missing_cols = [c for c in NEED_COLS if c not in df.columns]
    if missing_cols:
        sys.exit(f"AA CSV is missing columns: {missing_cols}")
    df = df[NEED_COLS].copy()

    # Data-quality printout: read this before trusting anything below.
    n_missing = int(df[["OD_PAX", "OD_REV_USD"]].isna().any(axis=1).sum())
    dupes = int(df.duplicated(NEED_COLS[:4]).sum())
    n_nonpos = int((df["OD_PAX"] <= 0).sum())
    print(f"AA rows: {len(df):,} | missing pax/revenue: {n_missing:,} (dropped) | "
          f"pax <= 0: {n_nonpos:,} (dropped) | repeated market-months: {dupes:,} (summed)")
    df = df.dropna(subset=["OD_PAX", "OD_REV_USD"])
    df = df[df["OD_PAX"] > 0]

    df["month"] = pd.to_datetime(pd.DataFrame(
        {"year": df["DEP_YEAR"].astype(int), "month": df["DEP_MONTH"].astype(int), "day": 1}))

    # Segment each row by the countries of its two airports.
    country = airports.drop_duplicates("AIRPRT_CD").set_index("AIRPRT_CD")["CNTRY_CD"]
    org_c = df["ORG_AIRPRT"].map(country)
    des_c = df["DES_AIRPRT"].map(country)
    domestic = org_c.isin(US_CA) & des_c.isin(US_CA)
    international = org_c.notna() & des_c.notna() & ~domestic
    df["segment"] = np.select([domestic, international], ["DOMESTIC", "INTERNATIONAL"],
                              default="UNKNOWN")
    n_unknown = int((df["segment"] == "UNKNOWN").sum())
    if n_unknown:
        print(f"  {n_unknown:,} rows have an airport missing from the airport file; "
              "they count in ALL only")

    # Sum duplicates (same idea as the `+=` in the routes dictionary in commit 1).
    mm = (df.groupby(["ORG_AIRPRT", "DES_AIRPRT", "month", "segment"], as_index=False)
            [["OD_PAX", "OD_REV_USD"]].sum()
            .rename(columns={"ORG_AIRPRT": "org", "DES_AIRPRT": "des",
                             "OD_PAX": "pax", "OD_REV_USD": "rev"}))
    mm["market"] = mm["org"] + "->" + mm["des"]      # same key style as routes.json
    mm["year"] = mm["month"].dt.year
    mm["moy"] = mm["month"].dt.month                 # month of year, for seasonal baselines
    print(f"  data spans {mm['month'].min():%b %Y} to {mm['month'].max():%b %Y}, "
          f"{mm['market'].nunique():,} markets")
    if BASE_YEAR not in set(mm["year"]) or mm.loc[mm["year"] == BASE_YEAR, "moy"].nunique() < 12:
        sys.exit(f"Need all 12 months of {BASE_YEAR} as the pre-COVID baseline.")
    return mm


# ---------------------------------------------------------------------------
# 2. Monthly series and recovery metrics
# ---------------------------------------------------------------------------
def monthly(mm: pd.DataFrame, seg: str) -> pd.DataFrame:
    """Monthly pax, revenue and average fare for one segment."""
    d = mm if seg == "ALL" else mm[mm["segment"] == seg]
    g = d.groupby("month")[["pax", "rev"]].sum().asfreq("MS").fillna(0.0)
    g["fare"] = g["rev"] / g["pax"].replace(0, np.nan)
    return g


def same_month_index(s: pd.Series) -> pd.Series:
    """Series divided by the same calendar month of BASE_YEAR (1.0 = back to 2019)."""
    base = s[s.index.year == BASE_YEAR]
    base_by_moy = dict(zip(base.index.month, base.values))
    baseline = pd.Series([base_by_moy.get(d.month, np.nan) for d in s.index], index=s.index)
    return s / baseline


def latest_full_year(m: pd.DataFrame) -> int | None:
    """Most recent year after BASE_YEAR with all 12 months present."""
    counts = m.groupby(m.index.year).size()
    years = [y for y, n in counts.items() if n == 12 and y > BASE_YEAR]
    return max(years) if years else None


def recovery_metrics(m: pd.DataFrame, metric: str, latest: int | None) -> dict:
    """Trough depth, time to recover and cumulative loss for one metric."""
    s = m[metric]
    idx = same_month_index(s)
    baseline = s / idx                                   # 2019 same-month level
    win = idx[(idx.index >= COVID_START) & (idx.index <= COVID_END)]
    trough_month = win.idxmin()

    # "Recovered" = first month that starts a run of SUSTAIN_MONTHS months >= 100%.
    after = idx[idx.index > trough_month]
    run_ok = after.rolling(SUSTAIN_MONTHS).min() >= 1.0
    rec_month = None
    if run_ok.any():
        rec_month = run_ok[run_ok].index[0] - pd.DateOffset(months=SUSTAIN_MONTHS - 1)

    months_to_recover = (np.nan if rec_month is None else
                         (rec_month.year - COVID_START.year) * 12
                         + rec_month.month - COVID_START.month)
    shortfall = (baseline - s)[(s.index >= COVID_START) & (s.index <= COVID_END)].sum()
    latest_pct = (s[s.index.year == latest].sum() / s[s.index.year == BASE_YEAR].sum()
                  if latest else np.nan)
    return dict(metric=metric, trough_month=trough_month, trough_pct=win.min(),
                recovered_month=rec_month, months_to_recover=months_to_recover,
                cumulative_shortfall=shortfall, latest_year=latest, latest_pct=latest_pct)


def fare_index(mm: pd.DataFrame, seg: str) -> pd.DataFrame:
    """Fixed-weight (Laspeyres) fare index vs the same month of BASE_YEAR.

    For each month, every market's current fare is weighted by how many
    passengers it carried in the same month of 2019. Markets that were not
    flown in 2019 are ignored, so a changing route mix cannot move the index.
    `coverage` is the share of 2019 passengers the matched markets represent;
    when it is low (deep COVID months) the index rests on fewer routes.
    """
    d = mm if seg == "ALL" else mm[mm["segment"] == seg]
    base = d[d["year"] == BASE_YEAR][["market", "moy", "pax", "rev"]].rename(
        columns={"pax": "bpax", "rev": "brev"})
    base["bfare"] = base["brev"] / base["bpax"]
    j = d.merge(base, on=["market", "moy"], how="inner")
    j["num"] = j["bpax"] * (j["rev"] / j["pax"])
    j["den"] = j["bpax"] * j["bfare"]
    g = j.groupby("month")[["num", "den", "bpax"]].sum()
    base_tot = base.groupby("moy")["bpax"].sum()
    cov = g["bpax"] / pd.Series(g.index.month, index=g.index).map(base_tot)
    return pd.DataFrame({"fare_index": g["num"] / g["den"], "coverage": cov})


def revenue_bridge(m: pd.DataFrame) -> pd.DataFrame:
    """Split the change in annual revenue vs BASE_YEAR into volume and price.

    revenue = pax * fare, so
        change = (pax - pax0) * fare0      <- volume effect (fewer/more passengers)
               + pax * (fare - fare0)      <- price effect  (higher/lower average fare)
    This is an exact identity. The price effect also contains route-mix changes
    (that is what the fixed-weight fare index above separates out).
    Only full 12-month years are used.
    """
    yr = m.groupby(m.index.year).agg(pax=("pax", "sum"), rev=("rev", "sum"),
                                     n=("pax", "size"))
    yr = yr[yr["n"] == 12]
    yr["fare"] = yr["rev"] / yr["pax"]
    b = yr.loc[BASE_YEAR]
    out = yr.drop(index=BASE_YEAR).copy()
    out["volume_effect"] = (out["pax"] - b["pax"]) * b["fare"]
    out["price_effect"] = out["pax"] * (out["fare"] - b["fare"])
    out["total_change"] = out["rev"] - b["rev"]
    out["pct_of_base_rev"] = out["total_change"] / b["rev"]
    return out.drop(columns="n")


# ---------------------------------------------------------------------------
# 3. Who recovered? Markets and origin airports
# ---------------------------------------------------------------------------
def market_recovery(mm: pd.DataFrame, latest: int) -> pd.DataFrame:
    """Per-market passengers in 2019, 2020 and the latest full year."""
    ann = mm.groupby(["market", "segment", "year"])[["pax", "rev"]].sum().reset_index()
    pax = ann.pivot_table(index=["market", "segment"], columns="year", values="pax",
                          fill_value=0)
    rev = ann.pivot_table(index=["market", "segment"], columns="year", values="rev",
                          fill_value=0)
    t = pd.DataFrame({"pax_base": pax.get(BASE_YEAR), "pax_2020": pax.get(2020, 0),
                      "pax_latest": pax.get(latest),
                      "rev_base": rev.get(BASE_YEAR), "rev_latest": rev.get(latest)})
    t = t[t["pax_base"] >= MIN_BASE_PAX].copy()
    t["pct_latest"] = t["pax_latest"] / t["pax_base"]
    t["pct_2020"] = t["pax_2020"] / t["pax_base"]
    t["fare_base"] = t["rev_base"] / t["pax_base"]
    t["fare_latest"] = t["rev_latest"] / t["pax_latest"].replace(0, np.nan)
    t["fare_change"] = t["fare_latest"] / t["fare_base"] - 1

    # Suspended = carried passengers in Apr-Jun 2019 but none in Apr-Jun 2020.
    def window(year):
        d = mm[(mm["year"] == year) & mm["moy"].isin([4, 5, 6])]
        return d.groupby(["market", "segment"])["pax"].sum()
    suspended = (window(2020).reindex(t.index, fill_value=0) == 0) & \
                (window(BASE_YEAR).reindex(t.index, fill_value=0) > 0)
    t["suspended_spring_2020"] = suspended

    t["bucket"] = np.select(
        [t["pax_latest"] == 0, t["pct_latest"] < 0.7, t["pct_latest"] < 1.0],
        ["Exited", "Lagging (<70%)", "Partial (70-100%)"], default="Recovered (>=100%)")
    return t.reset_index().sort_values("pax_base", ascending=False)


def origin_recovery(mm: pd.DataFrame, latest: int, top_n: int = 15) -> pd.DataFrame:
    """Recovery by origin airport (a proxy for hub exposure)."""
    p = mm.groupby(["org", "year"])["pax"].sum().unstack(fill_value=0)
    t = pd.DataFrame({"pax_base": p[BASE_YEAR], "pax_2020": p.get(2020, 0),
                      "pax_latest": p[latest]})
    t["pct_2020"] = t["pax_2020"] / t["pax_base"]
    t["pct_latest"] = t["pax_latest"] / t["pax_base"]
    return t.sort_values("pax_base", ascending=False).head(top_n)


def seasonality(m: pd.DataFrame, latest: int) -> pd.DataFrame:
    """Share of the year's passengers that flies in each month: base year vs latest year."""
    def share(year):
        s = m.loc[m.index.year == year, "pax"]
        return pd.Series((s / s.sum()).values, index=s.index.month)
    return pd.DataFrame({f"{BASE_YEAR}": share(BASE_YEAR), f"{latest}": share(latest)})


# ---------------------------------------------------------------------------
# 4. Plots
# ---------------------------------------------------------------------------
def shade_covid(ax):
    ax.axvspan(COVID_START, COVID_END, color="grey", alpha=0.12, lw=0)
    ax.axhline(1.0, color="black", lw=0.8, ls=":")


def plot_recovery_curves(series: dict, path: Path):
    fig, ax = plt.subplots(figsize=(10, 4.5))
    colors = {"ALL": "black", "DOMESTIC": "tab:blue", "INTERNATIONAL": "tab:red"}
    for seg, m in series.items():
        ax.plot(m.index, same_month_index(m["pax"]), label=f"{seg} passengers",
                color=colors[seg])
    ax.plot(series["ALL"].index, same_month_index(series["ALL"]["rev"]),
            label="ALL revenue", color="tab:green", ls="--")
    shade_covid(ax)
    ax.set_ylabel(f"fraction of same month in {BASE_YEAR}")
    ax.set_title("Recovery vs same month of 2019 (grey = COVID window)")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def plot_price_volume(all_m: pd.DataFrame, fidx: pd.DataFrame, path: Path):
    fig, axes = plt.subplots(3, 1, figsize=(10, 7), sharex=True)
    panels = [("Passengers", same_month_index(all_m["pax"])),
              ("Fixed-weight fare index (same routes)", fidx["fare_index"]),
              ("Revenue", same_month_index(all_m["rev"]))]
    for ax, (title, s) in zip(axes, panels):
        ax.plot(s.index, s, color="black")
        shade_covid(ax)
        ax.set_title(title + f" (1.0 = same month {BASE_YEAR})", fontsize=9)
    fig.suptitle("Volume collapsed; did price compensate?")
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def plot_bridge(bridge: pd.DataFrame, path: Path):
    fig, ax = plt.subplots(figsize=(9, 4.5))
    x = np.arange(len(bridge))
    ax.bar(x, bridge["volume_effect"] / 1e9, label="volume effect", color="tab:blue")
    ax.bar(x, bridge["price_effect"] / 1e9, bottom=bridge["volume_effect"] / 1e9,
           label="price effect (incl. mix)", color="tab:orange")
    ax.plot(x, bridge["total_change"] / 1e9, "ko", label="total change")
    ax.axhline(0, color="black", lw=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(bridge.index)
    ax.set_ylabel(f"$ billions vs {BASE_YEAR}")
    ax.set_title("Revenue change vs 2019: volume vs price")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def plot_markets(mkt: pd.DataFrame, latest: int, path: Path):
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(12, 4.5))
    a1.hist(mkt["pct_latest"].clip(upper=2), bins=30, color="tab:blue")
    a1.axvline(1.0, color="black", ls=":")
    a1.set_xlabel(f"{latest} passengers / {BASE_YEAR} passengers (clipped at 2)")
    a1.set_ylabel("markets")
    a1.set_title("How many markets are back?")
    ok = mkt[(mkt["pax_latest"] > 0) & mkt["fare_change"].notna()]
    a2.scatter(ok["fare_change"], ok["pct_latest"].clip(upper=2), s=12, alpha=0.5)
    a2.axhline(1.0, color="black", ls=":")
    a2.axvline(0.0, color="black", ls=":")
    a2.set_xlabel(f"fare change {BASE_YEAR} -> {latest}")
    a2.set_ylabel("passenger recovery")
    a2.set_title("Market recovery vs fare change (descriptive only)")
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


# ---------------------------------------------------------------------------
# 5. Written findings: numbers come from the data, wording is conditional
# ---------------------------------------------------------------------------
def write_summary(rec: pd.DataFrame, bridge: pd.DataFrame, fidx: pd.DataFrame,
                  mkt: pd.DataFrame, orig: pd.DataFrame, seas: pd.DataFrame,
                  latest: int, rho: float, path: Path, synthetic: bool):
    def row(seg, metric):
        return rec[(rec["segment"] == seg) & (rec["metric"] == metric)].iloc[0]

    def rec_text(r):
        if pd.isna(r.months_to_recover):
            return "had not sustained 100% of 2019 by the end of the data"
        return f"sustained 100% of 2019 from {r.recovered_month:%b %Y} ({int(r.months_to_recover)} months after Mar 2020)"

    a, d, i = row("ALL", "pax"), row("DOMESTIC", "pax"), row("INTERNATIONAL", "pax")
    ar = row("ALL", "rev")
    L = ["# COVID-19 impact on AA passengers, fares and revenue", ""]
    if synthetic:
        L += ["**SYNTHETIC DATA.** Generated numbers; they say nothing about American Airlines.", ""]

    L += ["## What happened", "",
          f"- **Collapse.** Network passengers bottomed in {a.trough_month:%b %Y} at "
          f"{a.trough_pct:.0%} of the same month in {BASE_YEAR}; revenue bottomed at {ar.trough_pct:.0%}.",
          f"- **Recovery time.** Network passengers {rec_text(a)}. "
          f"Domestic {rec_text(d)}; international {rec_text(i)}.",
          f"- **Cumulative loss (Mar 2020 - Dec 2022 vs 2019 baseline).** "
          f"{a.cumulative_shortfall / 1e6:,.1f} million passengers and "
          f"${ar.cumulative_shortfall / 1e9:,.1f} billion of revenue.",
          f"- **Where we are.** {latest} network passengers = {a.latest_pct:.0%} of {BASE_YEAR}; "
          f"revenue = {ar.latest_pct:.0%}. Domestic {d.latest_pct:.0%}, international {i.latest_pct:.0%}."]

    b20 = bridge.loc[2020] if 2020 in bridge.index else None
    bl = bridge.loc[latest]
    if b20 is not None:
        L.append(f"- **Volume vs price.** In 2020 the volume effect was "
                 f"${b20.volume_effect / 1e9:,.1f}B and the price/mix effect ${b20.price_effect / 1e9:+,.1f}B. "
                 f"In {latest}: volume ${bl.volume_effect / 1e9:+,.1f}B, price/mix ${bl.price_effect / 1e9:+,.1f}B.")
    fl = fidx.loc[fidx.index.year == latest, "fare_index"].mean()
    L.append(f"- **Same-route fares.** The fixed-weight fare index averaged {fl:.2f} in {latest} "
             f"(1.00 = same month of {BASE_YEAR}), i.e. {fl - 1:+.0%} on identical routes.")

    vc = mkt["bucket"].value_counts()
    w = mkt.groupby("bucket")["pax_base"].sum() / mkt["pax_base"].sum()
    L += ["", "## Who did not come back", ""]
    for name in ["Recovered (>=100%)", "Partial (70-100%)", "Lagging (<70%)", "Exited"]:
        L.append(f"- {name}: {int(vc.get(name, 0)):,} markets, {w.get(name, 0):.0%} of {BASE_YEAR} passengers")
    n_susp = int(mkt["suspended_spring_2020"].sum())
    L.append(f"- {n_susp:,} of {len(mkt):,} markets carried no one in Apr-Jun 2020 (suspended).")
    worst = orig.sort_values("pct_latest").head(3)
    L.append("- Weakest top origin airports in " + str(latest) + ": " +
             ", ".join(f"{k} ({v:.0%})" for k, v in worst["pct_latest"].items()) + ".")
    pk = seas.idxmax()
    L.append(f"- Peak month by passenger share: month {int(pk.iloc[0])} in {BASE_YEAR}, "
             f"month {int(pk.iloc[1])} in {latest} "
             f"({seas.max().iloc[0]:.1%} -> {seas.max().iloc[1]:.1%} of the year).")
    L.append(f"- Across markets, Spearman correlation between fare change and passenger recovery: "
             f"{rho:+.2f} (descriptive, not an elasticity).")

    # ---- Recommendations: some are conditional on what the data showed ----
    L += ["", "## What to do about it", "",
          "Data-driven (triggered by the numbers above):", ""]
    if i.latest_pct < d.latest_pct - 0.05 or (pd.isna(i.months_to_recover) and not pd.isna(d.months_to_recover)):
        L.append("- **International is the laggard.** Rebuild it market by market rather than network-wide: "
                 "prioritise markets with the highest 2019 passengers that are still below 100%, "
                 "and stage re-entry behind demand signals (bookings, corporate contracts) instead of calendar dates.")
    else:
        L.append("- International has kept pace with domestic; keep monitoring market by market rather than adding blanket capacity.")
    if bl.price_effect > 0 and bl.volume_effect < 0:
        L.append("- **Price is carrying revenue while volume is still down.** That is fragile: it relies on "
                 "demand staying inelastic. Test fare moves in lagging markets before assuming it will hold.")
    if w.get("Lagging (<70%)", 0) + w.get("Exited", 0) > 0.10:
        L.append("- **A real slice of 2019 demand is still missing.** Decide per market whether it is "
                 "recoverable (restart, partner/codeshare, seasonal schedule) or structural (business "
                 "travel that moved to video calls); do not plan to a full return to 2019.")
    if abs(rho) < 0.2:
        L.append("- Fare change and recovery are only weakly related across markets, so a "
                 "network-wide price lever is blunt. Segment by market type.")
    L += ["", "Standing playbook (general, not derived from this data):", "",
          "- **Forecasting:** stop training on 2020-2022 as if it were normal. Mark the COVID window with "
          "indicator variables or drop it, anchor models on 2019 and 2023+, and keep recovery-ratio "
          "(% of 2019) as a feature. This is why macro_price_analysis.py also excludes those months.",
          "- **Network and capacity:** shift to flexible, demand-triggered scheduling (seasonal and "
          "short-notice adds/cuts); protect hub-feeding routes that recovered; use partners for thin markets.",
          "- **Revenue management:** segment by leisure vs business. Leisure and VFR demand returned first; "
          "price discovery in business-heavy markets should be re-tested because pre-COVID fare ladders may no longer fit.",
          "- **Resilience:** keep a liquidity buffer and variable-cost levers (fleet flexibility, "
          "contract terms) so the next shock is absorbed rather than financed.",
          "- **Next data to add:** capacity (seats/ASMs) and load factor, cabin class, and booking-curve "
          "data. Without them we can see that demand was lost but not whether it was a pricing, "
          "capacity or product problem.",
          "", "## Reading this", "",
          f"- All comparisons use the same calendar month of {BASE_YEAR}; 'recovered' means "
          f"{SUSTAIN_MONTHS} straight months at or above 100%.",
          "- O&D data shows itinerary demand, not seats flown.",
          "- The price effect in the revenue bridge includes route mix; the fare index is the clean same-route view.",
          "- Per-market and per-airport tables are in market_recovery.csv and origin_recovery.csv."]
    path.write_text("\n".join(L) + "\n", encoding="utf-8")


# ---------------------------------------------------------------------------
# 6. Synthetic data (doubles as the test)
# ---------------------------------------------------------------------------
def make_synthetic(seed: int = 11):
    """Fake AA data with a planted COVID shock:
    domestic recovers by ~Dec 2022, international by ~Dec 2023, fares +12-15% afterwards,
    a few markets suspended for spring 2020, one market that never returns."""
    rng = np.random.default_rng(seed)
    months = pd.date_range("2017-01-01", "2025-12-01", freq="MS")
    k = (months.year - 2020) * 12 + (months.month - 3)   # k=0 is Mar 2020
    dom = np.interp(k, [-100, -1, 0, 1, 3, 9, 21, 33, 100], [1, 1, .70, .12, .35, .50, .85, 1.10, 1.15])  # >1 so the segment clears 100% despite the exited market
    intl = np.interp(k, [-100, -1, 0, 1, 3, 9, 21, 33, 45, 100], [1, 1, .75, .06, .15, .25, .55, .88, 1.0, 1.08])
    fare_m = np.interp(k, [-100, -1, 1, 9, 21, 33, 100], [1, 1, .80, .95, 1.12, 1.15, 1.12])
    season = 1 + 0.15 * np.sin(2 * np.pi * (months.month - 3) / 12)

    airports = pd.DataFrame({
        "AIRPRT_CD": ["DFW", "ORD", "LAX", "MIA", "PHX", "CLT", "LHR", "MEX", "NRT", "CDG"],
        "CNTRY_CD": ["US", "US", "US", "US", "US", "US", "GB", "MX", "JP", "FR"],
        "STATE_PROVNC_CD": ["TX", "IL", "CA", "FL", "AZ", "NC", None, None, None, None],
    })
    us = ["DFW", "ORD", "LAX", "MIA", "PHX", "CLT"]
    markets = [(o, d) for o in us for d in us if o != d][:20]
    markets += [("DFW", "LHR"), ("MIA", "MEX"), ("DFW", "NRT"), ("CLT", "CDG")]
    intl_set = {("DFW", "LHR"), ("MIA", "MEX"), ("DFW", "NRT"), ("CLT", "CDG")}
    suspended = {markets[0], markets[1], ("DFW", "NRT")}
    exited = markets[2]

    rows = []
    for o, d in markets:
        curve = intl if (o, d) in intl_set else dom
        depth = rng.uniform(0.8, 1.1)                       # market-specific recovery strength
        post = np.where(k >= 0, depth, 1.0)
        pax = rng.uniform(2000, 20000) * season * curve * post * np.exp(rng.normal(0, .03, len(k)))
        fare = rng.uniform(150, 900) * fare_m * np.exp(rng.normal(0, .03, len(k)))
        for i, m in enumerate(months):
            if (o, d) in suspended and 1 <= k[i] <= 3:
                continue
            if (o, d) == exited and k[i] >= 1:
                continue
            rows.append((o, d, m.year, m.month, round(pax[i]), pax[i] * fare[i]))
    df = pd.DataFrame(rows, columns=NEED_COLS)
    return df, airports


def self_check(rec, bridge, fidx, mkt) -> bool:
    """Did we recover the planted COVID pattern?"""
    def r(seg, metric="pax"):
        return rec[(rec["segment"] == seg) & (rec["metric"] == metric)].iloc[0]
    a, d, i = r("ALL"), r("DOMESTIC"), r("INTERNATIONAL")
    checks = [
        ("trough falls in Mar-May 2020",
         pd.Timestamp("2020-03-01") <= a.trough_month <= pd.Timestamp("2020-05-01")),
        ("trough is deep (< 25% of 2019)", a.trough_pct < 0.25),
        ("international recovers later than domestic",
         (not pd.isna(d.months_to_recover)) and
         (pd.isna(i.months_to_recover) or i.months_to_recover > d.months_to_recover)),
        ("same-route fares ~12-15% above 2019 in 2023",
         1.05 < fidx.loc[fidx.index.year == 2023, "fare_index"].mean() < 1.25),
        ("revenue bridge adds up exactly",
         bool(np.allclose(bridge["volume_effect"] + bridge["price_effect"], bridge["total_change"]))),
        ("2020 revenue fell > 40% vs 2019, driven by volume",
         bridge.loc[2020, "pct_of_base_rev"] < -0.4
         and bridge.loc[2020, "volume_effect"] < bridge.loc[2020, "price_effect"]),
        ("never-returning market flagged as Exited", (mkt["bucket"] == "Exited").sum() >= 1),
        ("suspended markets detected", int(mkt["suspended_spring_2020"].sum()) >= 2),
    ]
    print("\nSelf-check:")
    for name, ok in checks:
        print(f"  [{'PASS' if ok else 'FAIL'}] {name}")
    return all(ok for _, ok in checks)


# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--pax-csv", help="AA_PAX_FORECASTING_DATA.csv")
    ap.add_argument("--airport-csv", help="AIRPORT_INFO_AA_DATA.csv")
    ap.add_argument("--out", default="outputs/covid")
    ap.add_argument("--synthetic", action="store_true",
                    help="use generated AA data with a planted COVID shock")
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    if args.synthetic:
        df, airports = make_synthetic()
    else:
        if not (args.pax_csv and args.airport_csv):
            ap.error("--pax-csv and --airport-csv are required unless --synthetic")
        df, airports = load_data(args.pax_csv, args.airport_csv)

    mm = build_market_months(df, airports)
    series = {seg: monthly(mm, seg) for seg in SEGMENTS}
    latest = latest_full_year(series["ALL"])
    if latest is None:
        sys.exit("Need at least one full year after 2019 to measure recovery.")
    print(f"Latest full year used for 'where are we now': {latest}")

    # Recovery metrics for passengers and revenue, every segment.
    rec_rows = []
    for seg, m in series.items():
        for metric in ("pax", "rev"):
            rec_rows.append({"segment": seg, **recovery_metrics(m, metric, latest)})
    rec = pd.DataFrame(rec_rows)

    fidx = fare_index(mm, "ALL")
    bridge = revenue_bridge(series["ALL"])
    mkt = market_recovery(mm, latest)
    orig = origin_recovery(mm, latest)
    seas = seasonality(series["ALL"], latest)

    ok = mkt[(mkt["pax_latest"] > 0) & mkt["fare_change"].notna()]
    rho = stats.spearmanr(ok["fare_change"], ok["pct_latest"]).statistic if len(ok) > 3 else np.nan

    # Save tables, plots and the written summary.
    pd.concat({s: m for s, m in series.items()}, names=["segment"]).to_csv(out / "monthly_series.csv")
    rec.to_csv(out / "recovery_summary.csv", index=False)
    fidx.to_csv(out / "fare_index.csv")
    bridge.to_csv(out / "revenue_bridge.csv")
    mkt.to_csv(out / "market_recovery.csv", index=False)
    orig.to_csv(out / "origin_recovery.csv")
    plot_recovery_curves(series, out / "recovery_curves.png")
    plot_price_volume(series["ALL"], fidx, out / "price_vs_volume.png")
    plot_bridge(bridge, out / "revenue_bridge.png")
    plot_markets(mkt, latest, out / "market_recovery.png")
    write_summary(rec, bridge, fidx, mkt, orig, seas, latest, rho, out / "summary.md", args.synthetic)

    show = rec.copy()
    show["trough_month"] = show["trough_month"].dt.strftime("%b %Y")
    show["recovered_month"] = show["recovered_month"].apply(
        lambda x: "-" if pd.isna(x) else x.strftime("%b %Y"))
    print("\n" + show[["segment", "metric", "trough_month", "trough_pct", "recovered_month",
                       "months_to_recover", "latest_pct"]].to_string(index=False))
    print(f"\nOutputs in {out.resolve()}")

    if args.synthetic:
        sys.exit(0 if self_check(rec, bridge, fidx, mkt) else 1)


if __name__ == "__main__":
    main()
