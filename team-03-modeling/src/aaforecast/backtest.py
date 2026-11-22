"""Rolling-origin backtest that mirrors how the forecast is used.

For each test origin T (for example 2024-07), the model is trained only on rows
whose target month is on or before T, then asked for months T+12 to T+18. This
is the same position a planner is in when building a schedule a year out.

Selection uses only outcomes matured by the current origin. Without eligible
selection evidence, the configured fixed fallback is used (anchor only by default).
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from . import metrics
from .config import AA_HUBS, RunConfig
from .features import FeatureBuilder, training_origins
from .model import AnchorAdjustModel, baselines, combine, training_mask

log = logging.getLogger(__name__)
METHODS = ["anchor_adjust", "seasonal_anchor", "same_month_last_year", "same_month_x_trend"]
SIZE_LABELS = ["<1k", "1k-10k", "10k-50k", "50k-200k", "200k+"]


def eligible_origin(origin_idx: int, n_months: int, horizons) -> bool:
    """Keep an origin when its shortest horizon still lands inside the panel.

    A longer horizon with no actual yet is dropped row by row. Requiring the
    longest horizon would throw away 12-to-18-month evidence for that origin.
    """
    return int(origin_idx) + min(horizons) < int(n_months)


def select_shrink(folds, origin, cfg):
    """Score weights on earlier predictions with labels available by origin."""
    cutoff = pd.Timestamp(origin) - pd.DateOffset(months=cfg.label_lag_months)
    scores = {s: [] for s in cfg.shrink_grid}
    for frame in folds:
        eligible = (frame["origin"] < pd.Timestamp(origin)) & (frame["target_month"] <= cutoff) & frame["y"].notna()
        if not eligible.any():
            continue
        for s in scores:
            values = combine(frame, frame["raw_mix"], s, cfg.network_growth)
            scores[s].append(metrics.wape(frame.loc[eligible, "y"], values[eligible.to_numpy()]))
    valid = {s: np.mean(v) for s, v in scores.items() if v and np.isfinite(v).all()}
    return min(valid, key=valid.get) if valid else cfg.shrink


def run_backtest(builder: FeatureBuilder, cfg: RunConfig) -> dict[str, pd.DataFrame]:
    months = builder.p.months
    n_months = len(months)
    origins = training_origins(n_months, cfg.origin_step)
    log.info("Building training rows for %d origins", len(origins))
    full = builder.build(origins)

    # Separate earlier calibration origins allow genuine historical selection
    # without using outcomes from the outer evaluation period prematurely.
    calibration_folds = []
    first_outer = min(pd.Timestamp(label) for label in cfg.backtest_origins)
    for T in origins:
        if months[T] >= first_outer:
            break
        train = full[full["target_idx"] <= T - cfg.label_lag_months]
        if train.empty or not training_mask(train, cfg).any():
            continue
        frame = builder.origin_rows(T, min_ttm=0.0).reset_index(drop=True)
        if frame.empty:
            continue
        frame["raw_mix"] = AnchorAdjustModel(cfg).fit(train).raw_mix(frame)
        frame["fold"] = "calibration_" + months[T].strftime("%Y-%m")
        calibration_folds.append(frame)

    for i, frame in enumerate(calibration_folds):
        weight = select_shrink(calibration_folds[:i], frame["origin"].iloc[0], cfg)
        frame["anchor_adjust"] = combine(frame, frame["raw_mix"], weight, cfg.network_growth)
    calibration_columns = ["fold", "market_id", "org", "des", "origin", "target_month", "horizon", "ttm", "y", "anchor_adjust"]
    calibration_predictions = (pd.concat([f.loc[f["y"].notna(), calibration_columns] for f in calibration_folds], ignore_index=True)
                               if calibration_folds else pd.DataFrame(columns=calibration_columns))

    folds = []
    for label in sorted(cfg.backtest_origins):
        T = builder.p.month_index(label)
        if not eligible_origin(T, n_months, cfg.horizons):
            log.warning("Skipping origin %s: not enough actuals after it", label)
            continue
        model = AnchorAdjustModel(cfg).fit(full[full["target_idx"] <= T - cfg.label_lag_months])
        test = builder.origin_rows(T, min_ttm=0.0)
        test = test.reset_index(drop=True)
        test["raw_mix"] = model.raw_mix(test)
        for name, values in baselines(test).items():
            test[name] = values
        test["fold"] = label
        folds.append(test)
        log.info("Fold %s trained on %d rows", label, int((full["target_idx"] <= T).sum()))

    if not folds:
        raise ValueError("No eligible backtest origins; check data coverage and horizons")
    # Retrospective grid is diagnostic only; selection below enforces label maturity.
    grid = {}
    for test in folds:
        for s in cfg.shrink_grid:
            valid = test["y"].notna().to_numpy()
            grid[(test["fold"].iloc[0], s)] = metrics.wape(test.loc[valid, "y"], combine(test, test["raw_mix"], s, cfg.network_growth)[valid])
    calib = pd.Series(grid).rename_axis(["fold", "shrink"]).rename("wape").reset_index()
    preds = []
    for i, test in enumerate(folds):
        s = select_shrink(calibration_folds + folds[:i], test["origin"].iloc[0], cfg)
        test["anchor_adjust"] = combine(test, test["raw_mix"], s, cfg.network_growth)
        test["shrink_used"] = s
        log.info("Fold %s: shrink %.2f, WAPE %.3f vs seasonal anchor %.3f vs last year %.3f",
                 test["fold"].iloc[0], s, metrics.wape(test["y"], test["anchor_adjust"]),
                 metrics.wape(test["y"], test["seasonal_anchor"]), metrics.wape(test["y"], test["same_month_last_year"]))
        keep = ["fold", "shrink_used", "market_id", "org", "des", "horizon", "origin", "target_month", "ttm", "y",
                "mkt_domestic", *METHODS]
        preds.append(test.loc[test["y"].notna(), keep])
    pred = pd.concat(preds, ignore_index=True)
    if pred.empty or pred.duplicated(["market_id", "origin", "horizon", "target_month"]).any():
        raise ValueError("Backtest requires nonempty unique forecast cases")
    pred["segment"] = np.select([pred["mkt_domestic"] == 1, pred["mkt_domestic"] == 0], ["Domestic", "International"], default="Unknown")
    pred["size_band"] = _size_bucket(pred["ttm"])

    chosen = select_shrink(calibration_folds + folds, months[-1], cfg)
    out = {"predictions": pred}
    out["calibration"] = calib.assign(selected_for_forecast=lambda d: d["shrink"] == chosen)
    out["overall"] = metrics.summarize(pred, METHODS)
    out["by_horizon"] = metrics.summarize(pred, METHODS, ["horizon"])
    out["by_fold"] = metrics.summarize(pred, METHODS, ["fold"])
    out["by_segment"] = metrics.summarize(pred, METHODS, ["segment"])
    out["by_size"] = metrics.summarize(pred, METHODS, ["size_band"])

    # Aggregation levels planners use: origin airport, AA hub touch, whole network.
    airport = pred.groupby(["fold", "org", "target_month"], as_index=False)[["y", *METHODS]].sum()
    out["airport_level"] = metrics.summarize(airport, METHODS)
    hub_rows, hub_frames = [], []
    for hub in sorted(AA_HUBS):
        sel = pred[(pred["org"] == hub) | (pred["des"] == hub)]
        g = sel.groupby(["fold", "target_month"], as_index=False)[["y", *METHODS]].sum()
        hub_frames.append(g)
        s = metrics.summarize(g, METHODS)
        s.insert(0, "hub", hub)
        hub_rows.append(s)
    out["hub_level"] = pd.concat(hub_rows, ignore_index=True)
    network = pred.groupby(["fold", "target_month"], as_index=False)[["y", *METHODS]].sum()
    out["network_level"] = metrics.summarize(network, METHODS)
    cutoff = months[-1] - pd.DateOffset(months=cfg.label_lag_months)
    levels = {"airport": airport, "hub": pd.concat(hub_frames, ignore_index=True), "network": network}
    out["aggregate_interval_table"] = aggregate_intervals(
        {name: frame[frame["target_month"] <= cutoff] for name, frame in levels.items()}, cfg.interval)

    out["significance"] = pd.DataFrame([metrics.paired_test(pred, "anchor_adjust", ref) for ref in METHODS[1:]])

    table, cov = interval_table(pred, cfg.interval, cfg.label_lag_months, calibration_predictions)
    out["interval_calibration_predictions"] = calibration_predictions
    matured = pred[pred["target_month"] <= cutoff]
    if len(matured):
        errors = matured.assign(err=np.log1p(matured["y"]) - np.log1p(matured["anchor_adjust"]), size=_size_bucket(matured["ttm"]))
        table = _quantiles(errors, (1 - cfg.interval) / 2)
    else:
        table = table.iloc[:0]
    out["interval_table"] = table
    out["interval_coverage"] = cov
    return out


def _size_bucket(ttm: pd.Series) -> pd.Series:
    edges = [0, 1_000, 10_000, 50_000, 200_000, np.inf]
    return pd.cut(ttm, edges, labels=SIZE_LABELS, right=False)


def _quantiles(p: pd.DataFrame, a: float, min_n: int = 50) -> pd.DataFrame:
    """Error quantiles per (horizon, size); thin cells fall back to the whole horizon, then to all rows."""
    def q(g):
        return pd.Series({"q_low": g.quantile(a), "q_high": g.quantile(1 - a), "n": len(g)})

    cell = p.groupby(["horizon", "size"], observed=True)["err"].apply(q).unstack().reset_index()
    by_h = p.groupby("horizon")["err"].apply(q).unstack()
    overall = q(p["err"])
    grid = pd.MultiIndex.from_product([sorted(p["horizon"].unique()), SIZE_LABELS], names=["horizon", "size"]).to_frame(index=False)
    cell["size"] = cell["size"].astype(str)
    t = grid.merge(cell, on=["horizon", "size"], how="left")
    thin = t["n"].isna() | (t["n"] < min_n)
    for col in ("q_low", "q_high"):
        fallback = t["horizon"].map(by_h[col]).where(t["horizon"].map(by_h["n"]) >= min_n, overall[col])
        t[col] = t[col].where(~thin, fallback)
    return t[["horizon", "size", "q_low", "q_high"]]


def interval_table(pred: pd.DataFrame, level: float, label_lag_months: int = 1,
                   calibration_predictions: pd.DataFrame | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Empirical log-error quantiles by horizon and market size.

    Historical coverage uses only earlier, matured outcomes. Uncalibrated folds
    are recorded explicitly rather than borrowing future residuals.
    """
    a = (1 - level) / 2
    p = pred.assign(err=np.log1p(pred["y"]) - np.log1p(pred["anchor_adjust"]), size=_size_bucket(pred["ttm"]))
    history = p
    if calibration_predictions is not None and len(calibration_predictions):
        earlier = calibration_predictions.assign(
            err=np.log1p(calibration_predictions["y"]) - np.log1p(calibration_predictions["anchor_adjust"]),
            size=_size_bucket(calibration_predictions["ttm"]))
        history = pd.concat([earlier, p], ignore_index=True)
    cov_rows = []
    for fold in p["fold"].unique():
        current = p[p["fold"] == fold]
        origin = pd.Timestamp(current["origin"].iloc[0])
        cutoff = origin - pd.DateOffset(months=label_lag_months)
        other = history[(history["origin"] < origin) & (history["target_month"] <= cutoff)]
        if len(other) < 50:
            cov_rows.append({"fold": fold, "target": level, "coverage": np.nan,
                             "volume_weighted_coverage": np.nan, "mean_width": np.nan,
                             "calibration_rows": len(other), "interval_rows": 0,
                             "evaluation_rows": len(current), "status": "unavailable"})
            continue
        cur = apply_intervals(p[p["fold"] == fold], _quantiles(other, a), "anchor_adjust")
        inside = (cur["y"] >= cur["lower"]) & (cur["y"] <= cur["upper"])
        available = cur["lower"].notna() & cur["upper"].notna()
        denominator = cur.loc[available, "y"].sum()
        cov_rows.append({"fold": fold, "target": level, "coverage": inside[available].mean(),
                         "volume_weighted_coverage": (inside[available] * cur.loc[available, "y"]).sum() / denominator if denominator else np.nan,
                         "mean_width": (cur["upper"] - cur["lower"]).mean(),
                         "calibration_rows": len(other), "interval_rows": int(available.sum()),
                         "evaluation_rows": len(cur), "status": "evaluated" if available.any() else "unavailable"})
    return _quantiles(p, a), pd.DataFrame(cov_rows)


def aggregate_intervals(levels: dict[str, pd.DataFrame], level: float) -> pd.DataFrame:
    """Interval quantiles for totals, from backtest errors measured at that total.

    Summing market bounds would assume every market misses in the same
    direction at once, which makes airport and network bands far too wide.
    """
    a = (1 - level) / 2
    rows = []
    for name, df in levels.items():
        err = np.log1p(df["y"]) - np.log1p(df["anchor_adjust"])
        rows.append({"level": name, "q_low": err.quantile(a), "q_high": err.quantile(1 - a), "n": len(err)})
    return pd.DataFrame(rows)


def apply_aggregate_interval(df: pd.DataFrame, table: pd.DataFrame | None, level: str, forecast_col: str = "forecast") -> pd.DataFrame:
    d = df.copy()
    if table is None or level not in set(table["level"]):
        d["lower"] = np.nan
        d["upper"] = np.nan
        return d
    q = table.set_index("level").loc[level]
    d["lower"] = np.clip(np.expm1(np.log1p(d[forecast_col]) + q["q_low"]), 0, None)
    d["upper"] = np.expm1(np.log1p(d[forecast_col]) + q["q_high"])
    return d


def apply_intervals(df: pd.DataFrame, table: pd.DataFrame, forecast_col: str = "forecast") -> pd.DataFrame:
    t = table.copy()
    t["size"] = pd.Categorical(t["size"].astype(str), categories=SIZE_LABELS)
    d = df.drop(columns=["size"], errors="ignore").assign(size=_size_bucket(df["ttm"]))
    d = d.merge(t, on=["horizon", "size"], how="left")
    base = np.log1p(d[forecast_col])
    d["lower"] = np.clip(np.expm1(base + d["q_low"]), 0, None)
    d["upper"] = np.expm1(base + d["q_high"])
    return d.drop(columns=["size", "q_low", "q_high"])
