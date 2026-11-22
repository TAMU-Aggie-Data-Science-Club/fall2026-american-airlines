"""Produce the 12 to 18 month planning forecast and its companion views."""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from . import explain, scenario
from .backtest import apply_aggregate_interval, apply_intervals
from .config import AA_HUBS, RunConfig
from .features import FeatureBuilder, driver_group, training_origins
from .model import AnchorAdjustModel, relative_adjustment, training_mask

log = logging.getLogger(__name__)


def run_forecast(builder: FeatureBuilder, cfg: RunConfig, interval_table: pd.DataFrame | None,
                 aggregate_table: pd.DataFrame | None = None, bridge_top_n: int = 2000) -> dict[str, pd.DataFrame]:
    n_months = len(builder.p.months)
    T = n_months - 1
    full = builder.build(training_origins(n_months, cfg.origin_step))
    train = full[full["target_idx"] <= T - cfg.label_lag_months]
    model = AnchorAdjustModel(cfg).fit(train)
    log.info("Final model trained on data through %s (shrink %.2f, network growth %+.1f%%)",
             builder.p.months[T].strftime("%Y-%m"), cfg.shrink, 100 * cfg.network_growth)

    rows = builder.origin_rows(T, min_ttm=0.0).reset_index(drop=True)
    rows["forecast"] = model.predict(rows)
    if interval_table is not None:
        rows = apply_intervals(rows, interval_table)
    else:
        log.warning("No backtest interval table found; run the backtest for prediction intervals")
        rows["lower"] = np.nan
        rows["upper"] = np.nan

    if cfg.minimal_forecast:
        cols = ["org", "des", "origin", "target_month", "horizon", "anchor", "last_same_month", "forecast", "lower", "upper"]
        network = rows.groupby("target_month", as_index=False)[["anchor", "last_same_month", "forecast"]].sum()
        return {"market_forecast": rows[cols],
                "network_forecast": apply_aggregate_interval(network, aggregate_table, "network")}

    # Driver bridge for the largest markets (they carry most of the volume).
    top_ids = rows.drop_duplicates("market_id").nlargest(bridge_top_n, "ttm")["market_id"]
    sel = rows[rows["market_id"].isin(top_ids)]
    bridge = explain.group_shapley(model, sel, cfg.shrink)
    bridge = pd.concat([sel[["market_id", "org", "des", "horizon", "target_month", "anchor", "forecast"]], bridge], axis=1)

    out: dict[str, pd.DataFrame] = {}
    cols = ["org", "des", "origin", "target_month", "horizon", "anchor", "last_same_month", "forecast", "lower", "upper",
            "ttm", "mkt_domestic"]
    out["market_forecast"] = rows[cols].rename(columns={"anchor": "seasonal_anchor", "ttm": "trailing_12m"})
    out["driver_bridge_markets"] = bridge
    out["network_bridge"] = network_bridge(bridge)
    out["hub_summary"] = hub_summary(rows, bridge, aggregate_table)
    out["watchlist"] = watchlist(rows)
    sums = ["last_same_month", "anchor", "forecast"]
    airport = rows.groupby(["org", "target_month"], as_index=False)[sums].sum().rename(columns={"org": "airport"})
    out["airport_forecast"] = apply_aggregate_interval(airport, aggregate_table, "airport")
    network = rows.groupby("target_month", as_index=False)[sums].sum()
    out["network_forecast"] = apply_aggregate_interval(network, aggregate_table, "network")
    out["scenarios"] = scenario.run_scenarios(model, rows, rows["forecast"].to_numpy(), cfg.shrink)
    out["feature_importance"] = feature_importance(model, train, cfg)
    return out


def network_bridge(bridge: pd.DataFrame) -> pd.DataFrame:
    drv = [c for c in bridge.columns if c.startswith("drv_")]
    agg = explain.aggregate_bridge(bridge[drv], bridge["forecast"])
    return pd.DataFrame({"driver": [explain.label(c) for c in agg.index], "effect_log_points": agg.to_numpy(),
                         "effect_pct": np.expm1(agg.to_numpy())})


def hub_summary(rows: pd.DataFrame, bridge: pd.DataFrame, aggregate_table: pd.DataFrame | None = None) -> pd.DataFrame:
    """Local demand touching each AA hub over the forecast window, with the driver that moves it most.

    The band applies the backtest's monthly hub error to each month; summing
    months treats their errors as fully correlated, which is conservative.
    """
    drv = [c for c in bridge.columns if c.startswith("drv_") and c not in ("drv_typical", "drv_network")]
    out = []
    for hub in sorted(AA_HUBS):
        sel = rows[(rows["org"] == hub) | (rows["des"] == hub)]
        if sel.empty:
            continue
        b = bridge[(bridge["org"] == hub) | (bridge["des"] == hub)]
        top = ""
        if len(b):
            eff = explain.aggregate_bridge(b[drv], b["forecast"])
            k = eff.abs().idxmax()
            top = f"{explain.label(k)} ({np.expm1(eff[k]):+.1%})"
        monthly = apply_aggregate_interval(sel.groupby("target_month", as_index=False)["forecast"].sum(), aggregate_table, "hub")
        out.append({
            "hub": hub,
            "markets": sel["market_id"].nunique(),
            "same_months_last_year": sel["last_same_month"].sum(),
            "forecast": sel["forecast"].sum(),
            "lower": monthly["lower"].sum(min_count=1),
            "upper": monthly["upper"].sum(min_count=1),
            "vs_last_year_pct": sel["forecast"].sum() / sel["last_same_month"].sum() - 1,
            "vs_anchor_pct": sel["forecast"].sum() / sel["anchor"].sum() - 1,
            "largest_driver": top,
        })
    return pd.DataFrame(out).sort_values("forecast", ascending=False)


def watchlist(rows: pd.DataFrame, min_ttm: float = 5_000, threshold: float = 0.08) -> pd.DataFrame:
    """Markets whose 12 to 18 month outlook departs most from their seasonal anchor.

    Flags compare the forecast window total with the anchor total, so they
    reflect what the model expects to change rather than last year's noise.
    """
    g = rows.groupby(["org", "des"], as_index=False).agg(
        trailing_12m=("ttm", "first"),
        anchor=("anchor", "sum"),
        last_year=("last_same_month", "sum"),
        forecast=("forecast", "sum"),
        lower=("lower", lambda values: values.sum(min_count=1)),
        upper=("upper", lambda values: values.sum(min_count=1)),
        domestic=("mkt_domestic", "first"),
    )
    g = g[g["trailing_12m"] >= min_ttm].copy()
    g["vs_anchor_pct"] = g["forecast"] / g["anchor"].replace(0, np.nan) - 1
    g["vs_last_year_pct"] = g["forecast"] / g["last_year"].replace(0, np.nan) - 1
    g["change_vs_anchor"] = g["forecast"] - g["anchor"]
    g["band_width_pct"] = (g["upper"] - g["lower"]) / g["forecast"].replace(0, np.nan)
    band_cut = g["band_width_pct"].quantile(0.9)
    g["flag"] = np.select(
        [g["vs_anchor_pct"] >= threshold, g["vs_anchor_pct"] <= -threshold, g["band_width_pct"] >= band_cut],
        ["Outgrowing network", "Lagging network", "High uncertainty"],
        default="In line",
    )
    return g.sort_values("change_vs_anchor", key=np.abs, ascending=False).reset_index(drop=True)


def feature_importance(model: AnchorAdjustModel, train: pd.DataFrame, cfg: RunConfig, n: int = 60_000) -> pd.DataFrame:
    """Permutation importance on a training sample, labeled by driver family."""
    from sklearn.inspection import permutation_importance

    rows = train[training_mask(train, cfg)]
    z = pd.Series(relative_adjustment(rows), index=rows.index)
    sample = rows.sample(min(n, len(rows)), random_state=cfg.seed)
    r = permutation_importance(model.model, sample[model.features], z.loc[sample.index], n_repeats=3,
                               random_state=cfg.seed, n_jobs=1)
    imp = pd.DataFrame({"feature": model.features, "importance": r.importances_mean})
    imp["group"] = imp["feature"].map(lambda f: explain.label("drv_" + driver_group(f)))
    return imp.sort_values("importance", ascending=False).reset_index(drop=True)
