"""Command line entry point: python -m aaforecast <command>."""

from __future__ import annotations

import argparse
import logging
import sys
import warnings

from .config import RunConfig

COMMANDS = ["validate", "fetch", "fetch-airports", "eda", "backtest", "forecast", "experiment", "submission-report", "all"]


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="aaforecast", description="12 to 18 month O&D demand forecasting")
    ap.add_argument("command", choices=COMMANDS)
    ap.add_argument("--target", choices=["pax", "rev"], default="pax", help="forecast passengers or revenue")
    ap.add_argument("--fast", action="store_true", help="largest markets only, for quick iteration")
    ap.add_argument("--no-external", action="store_true", help="skip public datasets")
    ap.add_argument("--external", action="store_true", help="opt in to external features; historical vintages require review")
    ap.add_argument("--passenger-only", action="store_true", help="run without airport metadata; geography and segments unavailable")
    ap.add_argument("--full-details", action="store_true", help="compute expensive explanations and scenarios")
    ap.add_argument("--label-lag-months", type=int, default=1)
    ap.add_argument("--backtest-run", help="explicit completed backtest run ID or directory")
    ap.add_argument("--backtest-origins", nargs="+", help="monthly origins, YYYY-MM")
    ap.add_argument("--horizons", nargs="+", type=int)
    ap.add_argument("--experiment-config", default="configs/ablations.json")
    ap.add_argument("--forecast-run")
    ap.add_argument("--experiment-run")
    ap.add_argument("--origin-step", type=int, default=3, help="months between training origins")
    ap.add_argument("--force", action="store_true", help="re-download external data")
    ap.add_argument("--network-growth", type=float, default=0.0,
                    help="annual network growth assumption, e.g. 0.03 for +3%% (default flat)")
    ap.add_argument("--shrink", type=float, default=None,
                    help="weight on the learned market mix; default is the backtest's choice")
    args = ap.parse_args(argv)
    if args.origin_step < 1 or args.label_lag_months < 0:
        ap.error("origin step must be positive and label lag nonnegative")
    if args.horizons and any(h < 1 or h > 24 for h in args.horizons):
        ap.error("horizons must be in 1..24")

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s", datefmt="%H:%M:%S")
    warnings.filterwarnings("ignore", category=UserWarning, module="openpyxl")
    cfg = RunConfig(target=args.target, fast=args.fast, use_external=args.external and not args.no_external,
                    origin_step=args.origin_step, network_growth=args.network_growth,
                    minimal_forecast=not args.full_details, label_lag_months=args.label_lag_months)
    cfg.passenger_only = args.passenger_only
    if args.horizons:
        cfg.horizons = tuple(sorted(set(args.horizons)))
    if args.backtest_origins:
        cfg.backtest_origins = tuple(args.backtest_origins)

    from . import pipeline

    steps = ["validate", "eda", "backtest", "forecast"] if args.command == "all" else [args.command]
    latest_backtest = args.backtest_run
    for step in steps:
        logging.info("Running %s", step)
        if step == "validate":
            print(pipeline.step_validate(cfg.passenger_only))
        elif step == "fetch":
            print(pipeline.step_fetch(force=args.force))
        elif step == "fetch-airports":
            from .external.airports import fetch_reference
            from .config import EXTERNAL_DIR
            print(fetch_reference(EXTERNAL_DIR / "public_airports", force=args.force))
        elif step == "eda":
            print(pipeline.step_eda(cfg.passenger_only))
        elif step == "backtest":
            latest_backtest = pipeline.step_backtest(cfg)
            print(latest_backtest)
        elif step == "forecast":
            print(pipeline.step_forecast(cfg, args.shrink, latest_backtest))
        elif step == "experiment":
            import json
            from pathlib import Path
            from .experiments import run_experiments
            print(run_experiments(json.loads(Path(args.experiment_config).read_text()), cfg))
        elif step == "submission-report":
            from pathlib import Path
            from .submission import build_submission
            if not args.backtest_run:
                ap.error("submission-report requires --backtest-run")
            print(build_submission(Path(args.backtest_run), args.forecast_run, args.experiment_run))
    return 0


if __name__ == "__main__":
    sys.exit(main())
