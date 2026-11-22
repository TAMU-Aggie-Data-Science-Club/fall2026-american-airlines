"""Reproduce the integrated release locally; real data/results remain ignored."""
import argparse
from dataclasses import replace
import json
import os
from pathlib import Path


def main():
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=root / "configs/release.json")
    parser.add_argument("--pax-file", type=Path)
    parser.add_argument("--airport-file", type=Path)
    parser.add_argument("--ablations", action="store_true")
    parser.add_argument("--full-details", action="store_true")
    args = parser.parse_args()
    if args.pax_file:
        os.environ["AAF_PAX_FILE"] = str(args.pax_file.resolve())
    if args.airport_file:
        os.environ["AAF_AIRPORT_FILE"] = str(args.airport_file.resolve())
    os.environ.setdefault("OMP_NUM_THREADS", "4")
    # Import after optional input overrides: config captures local paths at import.
    from aaforecast import config
    from aaforecast.config import RunConfig
    from aaforecast.pipeline import step_backtest, step_forecast, step_eda
    from aaforecast.experiments import run_experiments
    from aaforecast.submission import build_submission
    settings = json.loads(args.config.read_text(encoding="utf-8"))
    for key in ("horizons", "backtest_origins", "disabled_groups"):
        if key in settings:
            settings[key] = tuple(settings[key])
    cfg = RunConfig(**settings)
    cfg.minimal_forecast = not args.full_details
    backtest = step_backtest(cfg)
    forecast = step_forecast(replace(cfg), backtest_run=backtest)
    step_eda(cfg.passenger_only)
    experiment = None
    if args.ablations:
        experiment = run_experiments(json.loads((root / "configs/ablations.json").read_text()), cfg)
    report = build_submission(backtest, forecast, experiment)
    pointer = config.OUTPUT_DIR / "release_latest.json"
    pointer.write_text(json.dumps({"backtest": backtest, "forecast": forecast,
        "experiment": str(experiment) if experiment else None, "report": str(report)}, indent=2), encoding="utf-8")
    print(f"Local release artifacts recorded in {pointer}")


if __name__ == "__main__":
    main()
