"""End-to-end steps used by the command line."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass

import pandas as pd

from . import config, data, external, validate
from .backtest import run_backtest
from .config import RunConfig
from .eda import run_eda
from .features import FeatureBuilder
from .forecast import run_forecast
from .geo import market_attributes
from .report import backtest_report, forecast_report
from .runs import run_directory, validate_backtest

log = logging.getLogger(__name__)


@dataclass
class Inputs:
    panel: data.Panel
    airports: pd.DataFrame
    builder: FeatureBuilder


def load_inputs(cfg: RunConfig) -> Inputs:
    step_validate(cfg.passenger_only)
    df = data.load_pax()
    airports = airport_reference(df) if cfg.passenger_only else data.load_airports()
    if cfg.fast:
        top = df.groupby(["org", "des"])["od_pax"].sum().nlargest(cfg.fast_top_n).index
        df = df.set_index(["org", "des"]).loc[top].reset_index()
        log.info("Fast mode: %d largest markets", cfg.fast_top_n)
    panel = data.build_panel(df)
    attrs = market_attributes(panel.markets, airports)
    ext = external.load() if cfg.use_external else None
    builder = FeatureBuilder(panel, attrs, ext, cfg, airports)
    return Inputs(panel, airports, builder)


def _out(*parts):
    d = config.OUTPUT_DIR.joinpath(*parts)
    d.mkdir(parents=True, exist_ok=True)
    return d


def airport_reference(df):
    """Explicit unknown metadata for a passenger-only run; never invent geography."""
    codes = sorted(set(df["org"]) | set(df["des"]))
    reference = pd.DataFrame(index=pd.Index(codes, name="airport"))
    reference["metro"] = codes
    reference["country"] = ""
    for col in ("lat", "lon", "world_area", "runway_ft"):
        reference[col] = float("nan")
    return reference


def step_validate(passenger_only=False) -> str:
    raw = data.read_pax_raw()
    raw_checks = validate.check_raw(raw)
    if any(c.status == "FAIL" for c in raw_checks):
        d = _out("validation")
        (d / "data_quality.md").write_text(validate.to_markdown(raw_checks), encoding="utf-8")
        raise ValueError("Raw data validation failed; see local outputs/validation/data_quality.md")
    clean = data.load_pax(use_cache=False)
    airports = airport_reference(clean) if passenger_only else data.load_airports()
    checks = raw_checks + validate.check_clean(clean, airports)
    if passenger_only:
        checks.append(validate.Check("airport metadata", "WARN", "Passenger-only run: geography and segment metadata unavailable"))
    text = validate.to_markdown(checks)
    d = _out("validation")
    (d / "data_quality.md").write_text(text, encoding="utf-8")
    validate.to_frame(checks).to_csv(d / "data_quality.csv", index=False)
    if any(c.status == "FAIL" for c in checks):
        raise ValueError("Data validation failed; see local outputs/validation/data_quality.md")
    return text


def step_fetch(force: bool = False) -> dict:
    airports = data.load_airports()
    used = set(data.load_pax()[["org", "des"]].to_numpy().ravel())
    status = external.fetch_all(airports.loc[sorted(used)], force=force)
    (_out("validation") / "external_status.json").write_text(json.dumps(status, indent=2))
    return status


def step_eda(passenger_only=False) -> str:
    step_validate(passenger_only)
    df = data.load_pax()
    airports = airport_reference(df) if passenger_only else data.load_airports()
    return run_eda(data.build_panel(df), airports, _out("eda"))


def _tag(cfg: RunConfig) -> str:
    return cfg.target + ("_fast" if cfg.fast else "")


def step_backtest(cfg: RunConfig) -> str:
    with run_directory("backtest", cfg) as d:
        inp = load_inputs(cfg)
        res = run_backtest(inp.builder, cfg)
        for name, frame in res.items():
            frame.to_csv(d / ("predictions.csv.gz" if name == "predictions" else f"{name}.csv"), index=False)
        backtest_report(res, d)
    return str(d)


def step_forecast(cfg: RunConfig, shrink_override: float | None = None, backtest_run=None) -> str:
    bt = config.OUTPUT_DIR / "no_calibration" if backtest_run is None else config.OUTPUT_DIR / "backtest" / backtest_run
    if backtest_run is not None:
        validate_backtest(bt, cfg)
    table = pd.read_csv(bt / "interval_table.csv") if (bt / "interval_table.csv").exists() else None
    agg_path = bt / "aggregate_interval_table.csv"
    agg = pd.read_csv(agg_path) if agg_path.exists() else None
    if shrink_override is not None:
        cfg.shrink = shrink_override
    elif (bt / "calibration.csv").exists():
        cal = pd.read_csv(bt / "calibration.csv")
        cfg.shrink = float(cal.loc[cal["selected_for_forecast"], "shrink"].iloc[0])
    with run_directory("forecast", cfg) as d:
        inp = load_inputs(cfg)
        res = run_forecast(inp.builder, cfg, table, agg)
        for name, frame in res.items():
            frame.to_csv(d / f"{name}.csv", index=False)
        label = "passengers" if cfg.target == "pax" else "revenue (USD)"
        forecast_report(res, d, label)
    return str(d)
