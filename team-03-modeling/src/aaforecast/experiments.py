"""Finite, local feature-ablation trials with failures and review gates recorded."""
from dataclasses import replace
import json
from pathlib import Path
import pandas as pd

from .config import RunConfig
from .pipeline import load_inputs
from .backtest import run_backtest
from .report import backtest_report, md_table
from .runs import run_directory
from .privacy import scan_repository
from . import config

ALLOWED_GROUPS = {"mom", "fare", "net", "eco", "fx", "dem", "sea"}


def validate_spec(spec):
    trials = spec.get("trials", [])
    if not 1 <= len(trials) <= 4:
        raise ValueError("A bounded experiment requires 1..4 trials")
    names = [trial["name"] for trial in trials]
    if len(set(names)) != len(names):
        raise ValueError("Trial names must be unique")
    for trial in trials:
        if set(trial) - {"name", "disabled_groups"}:
            raise ValueError("Only named feature-group ablations are allowed; splits and budgets are fixed")
        if set(trial.get("disabled_groups", [])) - ALLOWED_GROUPS:
            raise ValueError("Unknown or protected feature group")
    return trials


def run_experiments(spec, cfg, builder_factory=None):
    """All candidates share config/splits. No automatic promotion or holdout tuning."""
    trials = validate_spec(spec)
    if cfg.gbm_params.get("max_iter", 300) > 300 or cfg.origin_step < 1:
        raise ValueError("Trial training budget exceeded")
    if scan_repository(config.ROOT):
        raise ValueError("Source-tree artifact preflight failed; run python -m aaforecast.privacy locally")
    with run_directory("experiments", cfg) as directory:
        (directory / "spec.json").write_text(json.dumps(spec, indent=2), encoding="utf-8")
        registry, rows, reference_keys = [], [], None
        for trial in trials:
            trial_cfg = replace(cfg, disabled_groups=tuple(trial.get("disabled_groups", [])))
            record = {"name": trial["name"], "gates": {"G0": "passed"}, "status": "running"}
            try:
                with run_directory("backtest", trial_cfg) as trial_dir:
                    record["run"] = str(trial_dir)
                    builder = builder_factory(trial_cfg) if builder_factory else load_inputs(trial_cfg).builder
                    record["gates"]["G1"] = "passed"
                    result = run_backtest(builder, trial_cfg)
                    record["gates"]["G2"] = "timing_contract_applied_source_availability_requires_review"
                    record["gates"]["G3"] = "passed"
                    keys = ["market_id", "origin", "horizon", "target_month"]
                    pred = result["predictions"].sort_values(keys).reset_index(drop=True)
                    actual_keys = pred[keys + ["y"]]
                    if pred.duplicated(keys).any():
                        raise ValueError("Duplicate forecast keys")
                    if reference_keys is not None and not actual_keys.equals(reference_keys):
                        raise ValueError("Candidates do not share identical evaluation cases/actuals")
                    reference_keys = actual_keys
                    record["gates"]["G4"] = "passed"
                    for name, frame in result.items():
                        frame.to_csv(trial_dir / ("predictions.csv.gz" if name == "predictions" else f"{name}.csv"), index=False)
                    backtest_report(result, trial_dir)
                    record["gates"]["G5"] = "awaiting_independent_review"
                    record["status"] = "review_ready"
                    rows.append({"trial": trial["name"], **result["overall"].iloc[0].to_dict()})
            except Exception as exc:
                record.update(status="failed", error_type=type(exc).__name__)
            registry.append(record)
            (directory / "registry.json").write_text(json.dumps(registry, indent=2), encoding="utf-8")
        comparison = pd.DataFrame(rows)
        comparison.to_csv(directory / "comparison.csv", index=False)
        (directory / "comparison.md").write_text("# Bounded experiment comparison\n\n" + md_table(comparison) +
            "\n\nRetrospective comparison; independent review required. No automatic model promotion.\n", encoding="utf-8")
    return directory
