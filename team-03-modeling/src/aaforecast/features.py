"""Feature engineering for the Anchor and Adjust model.

Every row is one (market, forecast origin, horizon). Time-series features slice
observations through the origin month. Release/revision availability and current
public airport snapshots require separate source review; calendar slicing alone
does not establish a strict historical replay. External features are opt-in.

Each row also carries the seasonal anchor: the market's trailing 12-month
monthly level times its own seasonal shape for the target calendar month,
averaged over the two most recent years not distorted by the pandemic.

Feature names carry a driver-family prefix used by the driver bridge:

    mom_   market momentum (recent level, growth, volatility, recovery)
    sea_   seasonality and calendar of the target month
    net_   airport and network trends around the market
    mkt_   static market profile (geography, hubs, haul length)
    dem_   demographics (catchment population, FAA enplanements, migration ties)
    eco_   US economy and fuel
    fx_    exchange rate of the foreign endpoint
    fare_  average fare level and change
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd

from . import calendar as cal
from .config import PANDEMIC_END, PANDEMIC_START, RunConfig
from .data import Panel
from .external import External
from .external.fred import COUNTRY_CURRENCY

DRIVER_GROUPS = {
    "mom": "Market momentum",
    "sea": "Seasonality and calendar",
    "net": "Airport and network trend",
    "mkt": "Market profile",
    "dem": "Demographics and migration",
    "eco": "Economy and fuel",
    "fx": "Exchange rates",
    "fare": "Fares",
}
KEY_COLS = ["market_id", "org", "des", "origin_idx", "horizon", "target_idx", "origin", "target_month"]


def anchor_index(origin_idx: int, horizon: int) -> int:
    """Most recent month on or before the origin with the same calendar month as the target."""
    return origin_idx + horizon - 12 * math.ceil(horizon / 12)


def _safe_log_ratio(a, b):
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.log1p(a) - np.log1p(b)


def _macro_features(macro: pd.DataFrame | None, months: pd.DatetimeIndex) -> pd.DataFrame:
    """Macro features per origin month, lagged one month for publication delay."""
    if macro is None or macro.empty:
        return pd.DataFrame(index=months)
    full = pd.date_range(min(macro.index.min(), months.min()), months.max(), freq="MS")
    m = macro.reindex(full).ffill()
    out = pd.DataFrame(index=full)
    for col in ("jet_fuel", "brent"):
        if col in m:
            out[f"eco_{col}_log"] = np.log(m[col])
            out[f"eco_{col}_chg12"] = np.log(m[col]).diff(12)
    if "unemployment" in m:
        out["eco_unemployment"] = m["unemployment"]
        out["eco_unemployment_chg12"] = m["unemployment"].diff(12)
    if "consumer_sentiment" in m:
        out["eco_sentiment"] = m["consumer_sentiment"]
        out["eco_sentiment_chg12"] = m["consumer_sentiment"].diff(12)
    for col in ("cpi_airfare", "cpi", "real_disp_income"):
        if col in m:
            out[f"eco_{col}_yoy"] = np.log(m[col]).diff(12)
    fx_cols = [c for c in m.columns if c.startswith("fx_")]
    for c in fx_cols:
        out[c] = np.log(m[c]).diff(12)  # +0.1 means the dollar bought ~10% more local currency
    return out.shift(1).reindex(months)


def _vintage(month: pd.Timestamp, release_month: int) -> int:
    """Latest annual vintage published by ``month`` if vintage Y is released in ``release_month`` of Y+1."""
    return month.year - 1 if month.month >= release_month else month.year - 2


def _annual_lookup(table: pd.DataFrame | None, value: str, airports: np.ndarray, year: int, min_year: int):
    """Log level and one-year growth for each airport at a given vintage year."""
    n = len(airports)
    if table is None or table.empty:
        return np.full(n, np.nan), np.full(n, np.nan)
    piv = table.pivot_table(index="airport", columns="year", values=value, aggfunc="sum")
    if year < min_year:
        return np.full(n, np.nan), np.full(n, np.nan)
    year = min(year, int(piv.columns.max()))
    cur = piv.get(year)
    prev = piv.get(year - 1)
    cur_v = cur.reindex(airports).to_numpy(float) if cur is not None else np.full(n, np.nan)
    prev_v = prev.reindex(airports).to_numpy(float) if prev is not None else np.full(n, np.nan)
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.log(cur_v), np.log(cur_v) - np.log(prev_v)


class FeatureBuilder:
    """Builds feature rows for any set of forecast origins."""

    def __init__(self, panel: Panel, attrs: pd.DataFrame, ext: External | None, cfg: RunConfig, airports: pd.DataFrame):
        self.p = panel
        self.cfg = cfg
        self.Y = panel.values(cfg.target_col)
        self.P = panel.pax
        self.R = panel.rev
        self.attrs = attrs
        self.ext = ext or External()
        self.org = panel.markets["org"].to_numpy()
        self.des = panel.markets["des"].to_numpy()
        codes = pd.Index(sorted(set(self.org) | set(self.des)))
        self.org_code = codes.get_indexer(self.org)
        self.des_code = codes.get_indexer(self.des)
        self.n_airports = len(codes)
        self.cal = cal.month_features(panel.months.append(pd.date_range(panel.months[-1], periods=25, freq="MS")[1:]))
        self.macro = _macro_features(self.ext.macro, panel.months)
        pstart, pend = pd.Timestamp(PANDEMIC_START), pd.Timestamp(PANDEMIC_END)
        self.pandemic = ((panel.months >= pstart) & (panel.months <= pend)).astype(int)
        # Pre-pandemic level for recovery features (calendar 2019 if present).
        pre = (panel.months.year == 2019)
        self.pre_level = self.Y[:, pre].sum(axis=1) if pre.sum() == 12 else np.full(len(self.org), np.nan)
        self._metros = airports["metro"]
        self.static = self._static_features()

    # ------------------------------------------------------------------ static
    def _static_features(self) -> pd.DataFrame:
        a = self.attrs
        s = pd.DataFrame(index=a.index)
        s["mkt_log_distance"] = np.log(a["distance_km"].clip(lower=1))
        for col in ("domestic", "org_us", "org_hub", "des_hub"):
            s[f"mkt_{col}"] = a[col]
        for col in ("org_lat", "org_lon", "des_lat", "des_lon", "org_world_area", "des_world_area", "org_runway_ft", "des_runway_ft"):
            s[f"mkt_{col}"] = a[col]
        mig = self.ext.migration
        if mig is not None and not mig.empty:
            key = pd.MultiIndex.from_arrays([self._metros.reindex(self.org).to_numpy(), self._metros.reindex(self.des).to_numpy()])
            flows = mig.set_index(["org_metro", "des_metro"])["movers"]
            s["dem_log_migration"] = np.log1p(flows.reindex(key).to_numpy(float))
        currency = a["foreign_country"].map(COUNTRY_CURRENCY)
        self._currency = currency.to_numpy()
        return s

    # ------------------------------------------------------------------ origins
    def origin_rows(self, t: int, horizons=None, min_ttm: float | None = None) -> pd.DataFrame:
        """Feature rows for origin index ``t`` across horizons."""
        if t < 11:
            raise ValueError("Need at least 12 months of history at the origin")
        horizons = horizons or self.cfg.horizons
        min_ttm = self.cfg.min_ttm if min_ttm is None else min_ttm
        Y, P, R = self.Y, self.P, self.R
        n_months = Y.shape[1]
        win = Y[:, t - 11 : t + 1]
        ttm = win.sum(axis=1)
        keep = np.flatnonzero(ttm >= min_ttm)
        if len(keep) == 0:
            return pd.DataFrame()
        y, win, ttm = Y[keep], win[keep], ttm[keep]
        p, r = P[keep], R[keep]

        f: dict[str, np.ndarray] = {}
        logwin = np.log1p(win)
        for k in range(12):
            f[f"mom_lag{k}"] = logwin[:, 11 - k]
        f["mom_log_ttm"] = np.log1p(ttm)
        if t >= 23:
            f["mom_growth_ttm"] = _safe_log_ratio(ttm, y[:, t - 23 : t - 11].sum(axis=1))
        else:
            f["mom_growth_ttm"] = np.full(len(keep), np.nan)
        f["mom_growth_3m"] = (
            _safe_log_ratio(y[:, t - 2 : t + 1].sum(1), y[:, t - 14 : t - 11].sum(1)) if t >= 14 else np.full(len(keep), np.nan)
        )
        f["mom_growth_6m"] = (
            _safe_log_ratio(y[:, t - 5 : t + 1].sum(1), y[:, t - 17 : t - 11].sum(1)) if t >= 17 else np.full(len(keep), np.nan)
        )
        f["mom_volatility"] = np.diff(logwin, axis=1).std(axis=1)
        f["mom_zero_months"] = (win == 0).sum(axis=1).astype(float)
        f["mom_recovery_vs_2019"] = _safe_log_ratio(ttm, self.pre_level[keep]) if t >= 23 else np.zeros(len(keep))
        f["mom_pandemic_months"] = np.full(len(keep), float(self.pandemic[max(0, t - 23) : t + 1].sum()))

        p_ttm = p[:, t - 11 : t + 1].sum(1)
        r_ttm = r[:, t - 11 : t + 1].sum(1)
        with np.errstate(divide="ignore", invalid="ignore"):
            fare = np.where(p_ttm > 0, r_ttm / p_ttm, np.nan)
            f["fare_log_avg"] = np.log(fare)
            if t >= 23:
                p_prev = p[:, t - 23 : t - 11].sum(1)
                r_prev = r[:, t - 23 : t - 11].sum(1)
                f["fare_growth"] = np.log(fare) - np.log(np.where(p_prev > 0, r_prev / p_prev, np.nan))
            else:
                f["fare_growth"] = np.full(len(keep), np.nan)
            f["fare_yield_per_km"] = np.log(fare / self.attrs["distance_km"].to_numpy()[keep].clip(min=1))

        # Airport and network context, measured on passengers across all markets.
        P_ttm = P[:, t - 11 : t + 1].sum(1)
        P_prev = P[:, t - 23 : t - 11].sum(1) if t >= 23 else None
        org_tot = np.bincount(self.org_code, P_ttm, self.n_airports)
        des_tot = np.bincount(self.des_code, P_ttm, self.n_airports)
        oc, dc = self.org_code[keep], self.des_code[keep]
        f["net_org_log_ttm"] = np.log1p(org_tot[oc])
        f["net_des_log_ttm"] = np.log1p(des_tot[dc])
        f["net_market_share_of_org"] = p_ttm / np.maximum(org_tot[oc], 1)
        if P_prev is not None:
            org_prev = np.bincount(self.org_code, P_prev, self.n_airports)
            des_prev = np.bincount(self.des_code, P_prev, self.n_airports)
            f["net_org_growth"] = _safe_log_ratio(org_tot, org_prev)[oc]
            f["net_des_growth"] = _safe_log_ratio(des_tot, des_prev)[dc]
            f["net_network_growth"] = np.full(len(keep), float(_safe_log_ratio(P_ttm.sum(), P_prev.sum())))
        else:
            for c in ("net_org_growth", "net_des_growth", "net_network_growth"):
                f[c] = np.full(len(keep), np.nan)
        if t >= 14:
            f["net_network_growth_3m"] = np.full(
                len(keep), float(_safe_log_ratio(P[:, t - 2 : t + 1].sum(), P[:, t - 14 : t - 11].sum()))
            )
        else:
            f["net_network_growth_3m"] = np.full(len(keep), np.nan)

        origin = self.p.months[t]
        # Demographics with publication lags: county estimates each spring, FAA in the fall.
        org_ap, des_ap = self.org[keep], self.des[keep]
        pop_v = _vintage(origin, 4)
        enp_v = _vintage(origin, 10)
        for side, aps in (("org", org_ap), ("des", des_ap)):
            uniq, inv = np.unique(aps, return_inverse=True)
            lvl, gr = _annual_lookup(self.ext.population, "catchment_pop", uniq, pop_v, 2020)
            f[f"dem_{side}_log_pop"], f[f"dem_{side}_pop_growth"] = lvl[inv], gr[inv]
            lvl, gr = _annual_lookup(self.ext.cbsa_pop, "cbsa_pop", uniq, pop_v, 2020)
            f[f"dem_{side}_metro_pop_growth"] = gr[inv]
            lvl, gr = _annual_lookup(self.ext.enplanements, "enplanements", uniq, enp_v, 2018)
            f[f"dem_{side}_log_enplanements"], f[f"dem_{side}_enpl_growth"] = lvl[inv], gr[inv]

        # Macro and FX at the origin.
        macro_row = self.macro.loc[origin] if len(self.macro.columns) else pd.Series(dtype=float)
        for col, val in macro_row.items():
            if col.startswith("eco_"):
                f[col] = np.full(len(keep), float(val))
        fx_val = np.full(len(keep), np.nan)
        cur = self._currency[keep]
        for ccy in pd.unique(cur[pd.notna(cur)]):
            col = f"fx_{ccy}"
            if col in macro_row.index:
                fx_val[cur == ccy] = macro_row[col]
        f["fx_foreign_chg12"] = fx_val

        base = pd.DataFrame(f)
        static = self.static.iloc[keep].reset_index(drop=True)
        base = pd.concat([base, static], axis=1)

        frames = []
        for h in horizons:
            a = anchor_index(t, h)
            tgt = t + h
            last_same = y[:, a]
            shape, n_years = self._seasonal_shape(y, t, a)
            anchor = np.expm1(np.log1p(ttm / 12.0) + shape)
            fh = base.copy()
            fh["sea_horizon"] = float(h)
            fh["sea_shape"] = shape
            fh["sea_shape_years"] = n_years
            fh["sea_anchor_share"] = _safe_log_ratio(last_same, ttm / 12.0)
            if a >= 12:
                fh["sea_anchor_yoy"] = _safe_log_ratio(last_same, y[:, a - 12])
                prior = y[:, a - 12]
                prior_ttm = y[:, max(0, a - 23) : a - 11].sum(1) if a >= 23 else np.full(len(keep), np.nan)
                fh["sea_anchor_share_prev"] = _safe_log_ratio(prior, prior_ttm / 12.0)
            else:
                fh["sea_anchor_yoy"] = np.nan
                fh["sea_anchor_share_prev"] = np.nan
            fh["sea_anchor_pandemic"] = float(self.pandemic[a])
            tc = self.cal.iloc[tgt]
            ac = self.cal.iloc[a]
            for col in ("cal_month", "cal_weekend_days", "cal_easter", "cal_month_sin", "cal_month_cos"):
                fh[f"sea_{col}"] = float(tc[col])
            fh["sea_easter_shift"] = float(tc["cal_easter"] - ac["cal_easter"])
            fh["sea_weekend_shift"] = float(tc["cal_weekend_days"] - ac["cal_weekend_days"])
            fh["sea_days_shift"] = float(tc["cal_days"] - ac["cal_days"])

            fh.insert(0, "last_same_month", last_same)
            fh.insert(0, "anchor", anchor)
            # Absence is not evidence of zero demand: retain genuine reported zeros.
            target = np.where(self.p.observed[keep, tgt], y[:, tgt], np.nan) if tgt < n_months else np.nan
            fh.insert(0, "y", target)
            fh.insert(0, "target_month", self.cal.index[tgt])
            fh.insert(0, "origin", origin)
            fh.insert(0, "target_idx", tgt)
            fh.insert(0, "horizon", h)
            fh.insert(0, "origin_idx", t)
            fh.insert(0, "des", des_ap)
            fh.insert(0, "org", org_ap)
            fh.insert(0, "market_id", keep)
            fh["ttm"] = ttm
            fh["target_pandemic"] = float(self.pandemic[tgt]) if tgt < n_months else 0.0
            frames.append(fh)
        out = pd.concat(frames, ignore_index=True)
        fcols = feature_columns(out)
        out[fcols] = out[fcols].replace([np.inf, -np.inf], np.nan).astype("float32")
        disabled = tuple(f"{g}_" for g in self.cfg.disabled_groups)
        if disabled:
            out = out.drop(columns=[c for c in fcols if c.startswith(disabled)])
        return out

    def _seasonal_shape(self, y: np.ndarray, t: int, a: int, years: int = 2) -> tuple[np.ndarray, float]:
        """Average log ratio of the target's calendar month to its trailing-12 level.

        Uses up to ``years`` recent years whose month and trailing window avoid the
        pandemic. Falls back to the latest year when no clean year exists.
        """
        acc = np.zeros(len(y))
        used = 0
        fallback = None
        for k in range(4):
            ak, tk = a - 12 * k, t - 12 * k
            if ak < 0 or tk < 11:
                break
            level = y[:, tk - 11 : tk + 1].sum(axis=1) / 12.0
            ratio = _safe_log_ratio(y[:, ak], level)
            if fallback is None:
                fallback = ratio
            clean = self.pandemic[ak] == 0 and self.pandemic[tk - 11 : tk + 1].sum() == 0
            if clean:
                acc += ratio
                used += 1
                if used == years:
                    break
        if used == 0:
            return fallback, 0.0
        return acc / used, float(used)

    def build(self, origins: list[int], horizons=None) -> pd.DataFrame:
        frames = [self.origin_rows(t, horizons) for t in origins]
        frames = [f for f in frames if len(f)]
        return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def feature_columns(df: pd.DataFrame) -> list[str]:
    prefixes = tuple(f"{g}_" for g in DRIVER_GROUPS)
    return [c for c in df.columns if c.startswith(prefixes)]


def driver_group(col: str) -> str:
    return col.split("_", 1)[0]


def training_origins(n_months: int, step: int, last_origin: int | None = None) -> list[int]:
    """Origins spaced ``step`` months apart, aligned so the final month is included."""
    last = n_months - 1 if last_origin is None else last_origin
    return sorted(range(last, 10, -step))
