"""Which macro indicators move with AA fares and passenger demand?

One self-contained script. It builds monthly fare (revenue / passengers) and
passenger series from the AA forecasting CSV, downloads public FRED series,
and scans lead/lag correlations on year-over-year changes.

Usage
-----
Real data (run locally; outputs must stay out of Git):
    python macro_price_analysis.py --pax-csv data/AA_PAX_FORECASTING_DATA.csv \
        --airport-csv data/AIRPORT_INFO_AA_DATA.csv --out outputs/macro

Synthetic demo + self-check (fake AA data and fake macro data, no network):
    python macro_price_analysis.py --synthetic --out outputs/macro_demo

Synthetic AA data with real FRED downloads (checks that fetching works):
    python macro_price_analysis.py --synthetic --live-fred --out outputs/macro_live

Method notes
------------
* Every series is compared as a year-over-year change, never as a raw level.
  Levels share trends, so almost anything looks correlated in levels.
* "lag" = how many months before the target month the indicator value comes
  from. Lags shorter than an indicator's publication delay are skipped,
  because that data would not have been out yet.
* Target months from Mar 2020 to Dec 2022 are dropped by default: COVID itself
  plus the year after it, whose year-over-year base is a COVID month.
* p-values use an effective sample size that accounts for autocorrelation
  (overlapping 12-month changes are far from independent), and q-values apply
  a Benjamini-Hochberg correction across every test the script runs.
* Correlation is not causation, and FRED serves today's revised numbers, not
  the values that were published at the time.

Only public FRED series IDs are sent over the network. No AA data leaves the
machine.
"""

from __future__ import annotations

import argparse
import io
import sys
import urllib.request
import zlib
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy import stats  # noqa: E402

# ---------------------------------------------------------------------------
# Configuration: add or remove indicators here.
#   id: (label, publication delay in months, transform)
#   transform "log" = 12-month log change (prices, levels)
#   transform "diff" = 12-month change in points (rates, indexes in %)
# ---------------------------------------------------------------------------
NATIONAL_SERIES = {
    "UNRATE": ("Unemployment rate", 1, "diff"),
    "CPIAUCSL": ("CPI, all items", 1, "log"),
    "CUSR0000SETG01": ("CPI, airline fares", 1, "log"),
    "FEDFUNDS": ("Fed funds rate", 1, "diff"),
    "DGS10": ("10-year Treasury yield", 0, "diff"),
    "WJFUELUSGULF": ("Jet fuel price, Gulf Coast", 0, "log"),
    "DCOILBRENTEU": ("Brent crude oil", 0, "log"),
    "UMCSENT": ("Consumer sentiment (UMich)", 1, "diff"),
    "DSPIC96": ("Real disposable income", 1, "log"),
    "DTWEXBGS": ("Broad US dollar index", 0, "log"),
}

STATE_UNEMPLOYMENT_LAG = 1  # FRED id is "<state>UR", e.g. TXUR

# Exchange rates, keyed by country code: (FRED id, invert).
# invert=True turns "USD per foreign unit" into "foreign units per USD",
# so for every series "up" means a stronger dollar.
FX_SERIES = {
    "MX": ("DEXMXUS", False), "CA": ("DEXCAUS", False), "JP": ("DEXJPUS", False),
    "BR": ("DEXBZUS", False), "CN": ("DEXCHUS", False), "IN": ("DEXINUS", False),
    "KR": ("DEXKOUS", False), "CH": ("DEXSZUS", False),
    "GB": ("DEXUSUK", True), "AU": ("DEXUSAL", True),
    **{c: ("DEXUSEU", True) for c in
       ["DE", "FR", "ES", "IT", "NL", "IE", "PT", "BE", "AT", "GR", "FI"]},
}
FX_LAG = 0

US_CA = {"US", "CA", "PR", "VI", "GU", "AS", "MP"}  # territories count as US
COVID_START, COVID_END = "2020-03-01", "2022-12-01"
MIN_OBS = 24
Q_THRESHOLD = 0.05
FRED_URL = "https://fred.stlouisfed.org/graph/fredgraph.csv?id={}"


# ---------------------------------------------------------------------------
# AA data -> monthly segment series
# ---------------------------------------------------------------------------
def load_aa(pax_csv: str, airport_csv: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    df = pd.read_csv(pax_csv, keep_default_na=False, na_values=[""])
    airports = pd.read_csv(airport_csv, keep_default_na=False, na_values=[""])
    return df, airports


def build_targets(df: pd.DataFrame, airports: pd.DataFrame, top_n: int):
    """Return (targets, market_info).

    targets: dict segment -> DataFrame indexed by month with columns fare, pax.
    market_info: dict market -> {"state": ..., "foreign": ...} for local indicators.
    """
    need = ["ORG_AIRPRT", "DES_AIRPRT", "DEP_YEAR", "DEP_MONTH", "OD_PAX", "OD_REV_USD"]
    missing_cols = [c for c in need if c not in df.columns]
    if missing_cols:
        sys.exit(f"AA CSV is missing columns: {missing_cols}")
    df = df[need].copy()

    n_missing = int(df[["OD_PAX", "OD_REV_USD"]].isna().any(axis=1).sum())
    dupes = int(df.duplicated(need[:4]).sum())
    print(f"AA rows: {len(df):,} | rows with missing pax/revenue: {n_missing:,} "
          f"(dropped) | repeated market-months: {dupes:,} (summed)")
    df = df.dropna(subset=["OD_PAX", "OD_REV_USD"])
    df = df[df["OD_PAX"] > 0]

    df["month"] = pd.to_datetime(pd.DataFrame(
        {"year": df["DEP_YEAR"].astype(int), "month": df["DEP_MONTH"].astype(int), "day": 1}))
    df["market"] = df["ORG_AIRPRT"].astype(str) + "-" + df["DES_AIRPRT"].astype(str)

    ap = airports.drop_duplicates("AIRPRT_CD").set_index("AIRPRT_CD")
    country = ap["CNTRY_CD"].to_dict()
    state = ap["STATE_PROVNC_CD"].to_dict() if "STATE_PROVNC_CD" in ap else {}
    org_c = df["ORG_AIRPRT"].map(country)
    des_c = df["DES_AIRPRT"].map(country)
    unknown = int((org_c.isna() | des_c.isna()).sum())
    if unknown:
        print(f"  {unknown:,} rows have an airport missing from the airport file; "
              "they count in ALL only")
    df["domestic"] = org_c.isin(US_CA) & des_c.isin(US_CA)
    df["international"] = org_c.notna() & des_c.notna() & ~df["domestic"]

    def monthly(sub: pd.DataFrame) -> pd.DataFrame:
        g = sub.groupby("month")[["OD_PAX", "OD_REV_USD"]].sum()
        out = pd.DataFrame({"pax": g["OD_PAX"], "fare": g["OD_REV_USD"] / g["OD_PAX"]})
        return out.asfreq("MS")

    # Fixed market set: only markets present in (nearly) every month, so a
    # change in route mix does not show up as a change in average fare.
    n_months = df["month"].nunique()
    coverage = df.groupby("market")["month"].nunique()
    stable = coverage[coverage >= 0.9 * n_months].index
    print(f"  stable markets (in >=90% of {n_months} months): {len(stable):,} "
          f"of {coverage.size:,}")
    sdf = df[df["market"].isin(stable)]

    targets = {
        "ALL": monthly(sdf),
        "DOMESTIC": monthly(sdf[sdf["domestic"]]),
        "INTERNATIONAL": monthly(sdf[sdf["international"]]),
    }
    market_info = {}
    top = sdf.groupby("market")["OD_PAX"].sum().nlargest(top_n).index
    for m in top:
        sub = sdf[sdf["market"] == m]
        org, des = m.split("-", 1)
        foreign = None
        if country.get(des) not in US_CA and country.get(des) is not None:
            foreign = country.get(des)
        elif country.get(org) not in US_CA and country.get(org) is not None:
            foreign = country.get(org)
        st = state.get(org) if country.get(org) == "US" else None
        targets[m] = monthly(sub)
        market_info[m] = {"state": st if isinstance(st, str) else None,
                          "foreign": foreign}
    return {k: v for k, v in targets.items() if v["pax"].notna().sum() >= MIN_OBS}, market_info


# ---------------------------------------------------------------------------
# FRED
# ---------------------------------------------------------------------------
def fetch_fred(series_id: str, cache_dir: Path) -> pd.Series | None:
    """Monthly average of a FRED series, cached as CSV. None if unavailable."""
    cache = cache_dir / f"{series_id}.csv"
    if cache.exists():
        text = cache.read_text()
    else:
        try:
            # Keep Python's default User-Agent: FRED stalls on custom ones.
            with urllib.request.urlopen(FRED_URL.format(series_id), timeout=30) as r:
                text = r.read().decode()
        except Exception as e:  # network or HTTP error
            print(f"  ! could not download {series_id}: {e}")
            return None
        cache_dir.mkdir(parents=True, exist_ok=True)
        cache.write_text(text)
    raw = pd.read_csv(io.StringIO(text))
    if raw.shape[1] < 2:
        print(f"  ! unexpected response for {series_id}")
        return None
    s = pd.Series(pd.to_numeric(raw.iloc[:, 1], errors="coerce").values,
                  index=pd.to_datetime(raw.iloc[:, 0]), name=series_id).dropna()
    return s.resample("MS").mean()


def yoy(s: pd.Series, how: str) -> pd.Series:
    if how == "log":
        return np.log(s.where(s > 0)).diff(12)
    return s.diff(12)


def build_indicators(market_info, fetch, synthetic_extra=None):
    """Return dict id -> (label, delay, yoy series, scope).

    scope is "national" or a market name (local indicator for that market).
    """
    out = {}
    for sid, (label, delay, how) in NATIONAL_SERIES.items():
        s = fetch(sid)
        if s is not None:
            out[sid] = (label, delay, yoy(s, how), "national")
    for sid, (label, delay, how, s) in (synthetic_extra or {}).items():
        out[sid] = (label, delay, yoy(s, how), "national")

    for m, info in market_info.items():
        if info["state"]:
            sid = f"{info['state']}UR"
            s = fetch(sid)
            if s is not None:
                out[f"{m}|{sid}"] = (f"{info['state']} unemployment rate",
                                     STATE_UNEMPLOYMENT_LAG, yoy(s, "diff"), m)
        if info["foreign"] in FX_SERIES:
            sid, invert = FX_SERIES[info["foreign"]]
            s = fetch(sid)
            if s is not None:
                s = 1 / s if invert else s
                out[f"{m}|{sid}"] = (f"{info['foreign']} currency per USD",
                                     FX_LAG, yoy(s, "log"), m)
    return out


# ---------------------------------------------------------------------------
# Correlation scan
# ---------------------------------------------------------------------------
def corr_test(x: np.ndarray, y: np.ndarray):
    """Pearson r with an autocorrelation-adjusted p-value, plus Spearman r."""
    n = len(x)
    r = np.corrcoef(x, y)[0, 1]
    r1 = np.corrcoef(x[1:], x[:-1])[0, 1]
    r2 = np.corrcoef(y[1:], y[:-1])[0, 1]
    n_eff = n * (1 - r1 * r2) / (1 + r1 * r2) if abs(r1 * r2) < 1 else 3
    n_eff = float(np.clip(n_eff, 3, n))
    t = r * np.sqrt((n_eff - 2) / max(1e-12, 1 - r * r))
    p = 2 * stats.t.sf(abs(t), n_eff - 2)
    rho = stats.spearmanr(x, y).statistic
    return r, rho, p, n_eff


def bh_qvalues(p: np.ndarray) -> np.ndarray:
    n = len(p)
    order = np.argsort(p)
    ranked = p[order] * n / np.arange(1, n + 1)
    q = np.minimum.accumulate(ranked[::-1])[::-1]
    out = np.empty(n)
    out[order] = np.minimum(q, 1)
    return out


def scan(targets, indicators, max_lag: int, keep_covid: bool) -> pd.DataFrame:
    rows = []
    for seg, tdf in targets.items():
        for target in ("fare", "pax"):
            y_all = yoy(tdf[target], "log")
            if not keep_covid:
                y_all = y_all[(y_all.index < COVID_START) | (y_all.index > COVID_END)]
            for iid, (label, delay, xs, scope) in indicators.items():
                if scope not in ("national", seg):
                    continue
                for lag in range(delay, max_lag + 1):
                    x_shift = xs.shift(lag, freq="MS")
                    pair = pd.concat([y_all, x_shift], axis=1, join="inner").dropna()
                    if len(pair) < MIN_OBS:
                        continue
                    r, rho, p, n_eff = corr_test(pair.iloc[:, 1].values,
                                                 pair.iloc[:, 0].values)
                    rows.append(dict(segment=seg, target=target,
                                     indicator=iid.split("|")[-1], label=label,
                                     lag=lag, pearson_r=r, spearman_r=rho,
                                     n=len(pair), n_eff=n_eff, p=p))
    res = pd.DataFrame(rows)
    if res.empty:
        sys.exit("No indicator/target pairs had enough overlapping months.")
    res["q"] = bh_qvalues(res["p"].values)
    res["significant"] = res["q"] < Q_THRESHOLD
    return res


def best_lags(res: pd.DataFrame) -> pd.DataFrame:
    """Strongest lag per (segment, target, indicator)."""
    idx = res.assign(a=res["pearson_r"].abs()).groupby(
        ["segment", "target", "indicator"])["a"].idxmax()
    best = res.loc[idx].copy()
    best["abs_r"] = best["pearson_r"].abs()
    return best.sort_values(["segment", "target", "abs_r"], ascending=[True, True, False])


# ---------------------------------------------------------------------------
# Outputs
# ---------------------------------------------------------------------------
def plot_heatmap(res: pd.DataFrame, target: str, path: Path):
    sub = res[(res["segment"] == "ALL") & (res["target"] == target)]
    if sub.empty:
        return
    grid = sub.pivot_table(index="label", columns="lag", values="pearson_r")
    sig = sub.pivot_table(index="label", columns="lag", values="significant",
                          aggfunc="max").reindex_like(grid)
    grid = grid.loc[grid.abs().max(axis=1).sort_values(ascending=False).index]
    sig = sig.loc[grid.index]
    fig, ax = plt.subplots(figsize=(1.0 + 0.55 * grid.shape[1], 1.2 + 0.42 * grid.shape[0]))
    im = ax.imshow(grid.values, cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto")
    ax.set_xticks(range(grid.shape[1]), grid.columns)
    ax.set_yticks(range(grid.shape[0]), grid.index)
    for i in range(grid.shape[0]):
        for j in range(grid.shape[1]):
            v = grid.values[i, j]
            if np.isnan(v):
                continue
            star = "*" if sig.values[i, j] == True else ""  # noqa: E712
            ax.text(j, i, f"{v:.2f}{star}", ha="center", va="center", fontsize=7,
                    color="white" if abs(v) > 0.6 else "black")
    ax.set_xlabel("lag (months the indicator leads)")
    ax.set_title(f"{target}: year-over-year correlation, all markets (* = q < {Q_THRESHOLD})")
    fig.colorbar(im, ax=ax, shrink=0.8)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def plot_overlay(top: pd.DataFrame, targets, indicators, keep_covid: bool, path: Path):
    top = top[top["segment"] == "ALL"].head(5)
    if top.empty:
        return
    fig, axes = plt.subplots(len(top), 1, figsize=(10, 2.3 * len(top)), sharex=True,
                             squeeze=False)
    for ax, (_, row) in zip(axes[:, 0], top.iterrows()):
        y = yoy(targets["ALL"][row["target"]], "log")
        x = indicators[row["indicator"]][2].shift(int(row["lag"]), freq="MS")
        pair = pd.concat([y.rename("t"), x.rename("i")], axis=1, sort=True).dropna()
        z = (pair - pair.mean()) / pair.std()
        ax.plot(z.index, z["t"], label=f"{row['target']} (YoY)", color="black")
        ax.plot(z.index, z["i"], label=f"{row['label']} lagged {int(row['lag'])}m",
                color="tab:orange")
        if not keep_covid:
            ax.axvspan(pd.Timestamp(COVID_START), pd.Timestamp(COVID_END),
                       color="grey", alpha=0.15, lw=0)
        ax.set_title(f"r = {row['pearson_r']:.2f}, q = {row['q']:.3g}", fontsize=9)
        ax.legend(fontsize=8, loc="upper left")
    fig.suptitle("Top indicators vs all-market target (standardized; grey = excluded)")
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def write_summary(top: pd.DataFrame, res: pd.DataFrame, path: Path, synthetic: bool):
    lines = ["# Macro indicators vs AA fare and passengers", ""]
    if synthetic:
        lines += ["**SYNTHETIC DATA.** These numbers come from generated data and say "
                  "nothing about American Airlines.", ""]
    lines += [f"{len(res):,} correlations tested; {int(res['significant'].sum()):,} pass "
              f"q < {Q_THRESHOLD} after the Benjamini-Hochberg correction.", ""]
    for target in ("fare", "pax"):
        t = top[(top["segment"] == "ALL") & (top["target"] == target)].head(8)
        lines += [f"## {target}: all markets, strongest indicators", "",
                  "| indicator | lag (months) | r | q |", "|---|---|---|---|"]
        lines += [f"| {r.label} | {int(r.lag)} | {r.pearson_r:+.2f} | {r.q:.3g} |"
                  for r in t.itertuples()]
        lines.append("")
    lines += [
        "## Reading this",
        "- All series are 12-month changes. Lag = months the indicator leads the target; "
        "lags shorter than the publication delay are not tested.",
        "- Correlation is not causation, and a single strong indicator may just be "
        "standing in for the business cycle.",
        "- FRED returns today's revised data, not what was published at the time. "
        "Before using an indicator as a forecasting input, re-check it with ALFRED "
        "vintage data.",
        "- Neighbouring lags are strongly related, so treat the best lag as approximate.",
        "- Per-market results are in top_indicators.csv; those series are noisier.",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


# ---------------------------------------------------------------------------
# Synthetic data (doubles as the test)
# ---------------------------------------------------------------------------
def make_synthetic(seed: int = 7):
    rng = np.random.default_rng(seed)
    months = pd.date_range("2008-01-01", "2025-12-01", freq="MS")
    n = len(months)

    def ar1(rho, sd, start=0.0):
        x = np.empty(n)
        x[0] = start
        for i in range(1, n):
            x[i] = rho * x[i - 1] + rng.normal(0, sd)
        return x

    macro = {}
    for sid, (_, _, how) in NATIONAL_SERIES.items():
        base = ar1(0.85, 0.05)
        macro[sid] = pd.Series(100 * np.exp(base) if how == "log" else 5 + 10 * base,
                               index=months)
    fuel_log = ar1(0.5, 0.15)
    macro["WJFUELUSGULF"] = pd.Series(2.5 * np.exp(fuel_log), index=months)
    unrate = 5 + ar1(0.5, 0.6)
    macro["UNRATE"] = pd.Series(unrate, index=months)
    noise = pd.Series(100 * np.exp(ar1(0.85, 0.05)), index=months)

    airports = pd.DataFrame({
        "AIRPRT_CD": ["DFW", "ORD", "LAX", "MIA", "PHX", "CLT", "LHR", "MEX", "NRT", "CDG"],
        "CNTRY_CD": ["US", "US", "US", "US", "US", "US", "GB", "MX", "JP", "FR"],
        "STATE_PROVNC_CD": ["TX", "IL", "CA", "FL", "AZ", "NC", None, None, None, None],
    })
    us = ["DFW", "ORD", "LAX", "MIA", "PHX", "CLT"]
    markets = [(o, d) for o in us for d in us if o != d][:20]
    markets += [("DFW", "LHR"), ("MIA", "MEX"), ("DFW", "NRT"), ("CLT", "CDG")]
    season = 1 + 0.15 * np.sin(2 * np.pi * (months.month - 3) / 12)
    fuel_l2 = np.roll(fuel_log, 2)
    unr_l3 = np.roll(unrate, 3)
    rows = []
    for o, d in markets:
        base_pax = rng.uniform(2000, 20000)
        base_fare = rng.uniform(150, 900)
        pax = base_pax * season * np.exp(-0.10 * (unr_l3 - 5) + rng.normal(0, 0.03, n))
        fare = base_fare * season ** 0.5 * np.exp(0.35 * fuel_l2 + rng.normal(0, 0.03, n))
        for i, m in enumerate(months):
            rows.append((o, d, m.year, m.month, round(pax[i]), pax[i] * fare[i]))
    df = pd.DataFrame(rows, columns=["ORG_AIRPRT", "DES_AIRPRT", "DEP_YEAR",
                                     "DEP_MONTH", "OD_PAX", "OD_REV_USD"])
    df = df[df["DEP_YEAR"] * 12 + df["DEP_MONTH"] > 2008 * 12 + 3]  # drop roll wrap
    return df, airports, macro, noise


def self_check(best: pd.DataFrame) -> bool:
    """Synthetic mode: did we recover the planted relationships?"""
    allb = best[best["segment"] == "ALL"].set_index(["target", "indicator"])
    checks = []
    fuel = allb.loc[("fare", "WJFUELUSGULF")]
    checks.append(("fare follows jet fuel, lag 2, positive",
                   fuel["lag"] == 2 and fuel["pearson_r"] > 0 and fuel["significant"]))
    unr = allb.loc[("pax", "UNRATE")]
    checks.append(("passengers fall with unemployment, lag 3, negative",
                   unr["lag"] == 3 and unr["pearson_r"] < 0 and unr["significant"]))
    noise_hits = best[(best["indicator"] == "SYNTH_NOISE") & best["significant"]]
    checks.append(("noise series is not flagged", noise_hits.empty))
    print("\nSelf-check:")
    for name, ok in checks:
        print(f"  [{'PASS' if ok else 'FAIL'}] {name}")
    return all(ok for _, ok in checks)


# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--pax-csv")
    ap.add_argument("--airport-csv")
    ap.add_argument("--out", default="outputs/macro")
    ap.add_argument("--max-lag", type=int, default=12)
    ap.add_argument("--top-markets", type=int, default=10)
    ap.add_argument("--keep-covid", action="store_true",
                    help="keep Mar 2020 - Dec 2022 target months")
    ap.add_argument("--cache-dir", default=".fred_cache")
    ap.add_argument("--synthetic", action="store_true",
                    help="use generated AA and macro data (no network)")
    ap.add_argument("--live-fred", action="store_true",
                    help="with --synthetic: download real FRED data instead")
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    cache_dir = Path(args.cache_dir)
    synthetic_extra = None

    if args.synthetic:
        df, airports, fake_macro, noise = make_synthetic()
        if args.live_fred:
            fetch = lambda sid: fetch_fred(sid, cache_dir)  # noqa: E731
        else:
            def fetch(sid):  # fake national series; local ones get noise
                if sid in fake_macro:
                    return fake_macro[sid]
                rng = np.random.default_rng(zlib.crc32(sid.encode()))
                return pd.Series(100 * np.exp(np.cumsum(rng.normal(0, 0.02, len(noise)))),
                                 index=noise.index)
            synthetic_extra = {"SYNTH_NOISE": ("Pure noise (synthetic)", 0, "log", noise)}
    else:
        if not (args.pax_csv and args.airport_csv):
            ap.error("--pax-csv and --airport-csv are required unless --synthetic")
        df, airports = load_aa(args.pax_csv, args.airport_csv)
        fetch = lambda sid: fetch_fred(sid, cache_dir)  # noqa: E731

    targets, market_info = build_targets(df, airports, args.top_markets)
    print(f"Segments: {', '.join(targets)}")
    indicators = build_indicators(market_info, fetch, synthetic_extra)
    print(f"Indicators loaded: {len(indicators)}")

    res = scan(targets, indicators, args.max_lag, args.keep_covid)
    best = best_lags(res)
    top = best[best["significant"]].sort_values("abs_r", ascending=False)

    res.sort_values(["segment", "target", "indicator", "lag"]).to_csv(
        out / "correlations.csv", index=False)
    best.drop(columns="abs_r").to_csv(out / "top_indicators.csv", index=False)
    for target in ("fare", "pax"):
        plot_heatmap(res, target, out / f"lag_heatmap_{target}.png")
    plot_overlay(top, targets, {k.split("|")[-1]: v for k, v in indicators.items()},
                 args.keep_covid, out / "overlay_top5.png")
    write_summary(best, res, out / "summary.md", args.synthetic)

    print(f"\n{len(res):,} tests, {int(res['significant'].sum()):,} significant "
          f"(q < {Q_THRESHOLD}). Outputs in {out.resolve()}")
    top_all = top[top["segment"] == "ALL"]
    if top_all.empty:
        print("No indicator passes the significance threshold for all markets.")
    else:
        print(top_all[["target", "label", "lag", "pearson_r", "q"]]
              .head(10).to_string(index=False))

    if args.synthetic and not args.live_fred:
        sys.exit(0 if self_check(best) else 1)


if __name__ == "__main__":
    main()
