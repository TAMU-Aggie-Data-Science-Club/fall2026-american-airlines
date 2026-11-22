"""Create local evidence mapped to the six presentation grading criteria."""
import json
from pathlib import Path
import pandas as pd
from . import config
from .report import md_table


def build_submission(backtest, forecast=None, experiment=None):
    backtest = Path(backtest)
    manifest = json.loads((backtest / "manifest.json").read_text())
    if manifest["status"] != "complete":
        raise ValueError("Submission evidence requires a complete backtest")
    overall = pd.read_csv(backtest / "overall.csv")
    by_horizon = pd.read_csv(backtest / "by_horizon.csv")
    coverage = pd.read_csv(backtest / "interval_coverage.csv")
    selected = pd.read_csv(backtest / "calibration.csv")
    cfg = manifest["config"]
    lines = ["# Final project evidence and slide brief", "",
        "LOCAL DERIVED OUTPUT: release only to an approved audience after human review.", "",
        "## 1. Problem-Solving Approach", "",
        "Forecast monthly directional O&D passenger traffic for network planning. Begin with seasonal baselines, separate the network outlook from relative market movement, then evaluate a pooled challenger on matched historical cases.", "",
        "This describes the implemented technical approach. Add the team's actual initial decisions and chronology from meeting records; code cannot establish that history.", "",
        "## 2. Data Aggregation", "",
        "Passenger/revenue records are organized into directional origin-destination-month keys and market-by-month panels. Duplicate keys are summed. The release extract had no duplicate keys, so that rule did not change it. Absent observations are excluded as targets, while historical input imputation remains an explicit zero-fill assumption. Airport and network totals are aggregates of the same market forecasts; hub-touch totals overlap and should not be summed across hubs.", "",
        f"Configuration: passenger_only={cfg['passenger_only']}; external_features={cfg['use_external']}; label_lag_months={cfg['label_lag_months']}. The lag is a configured assumption, not independently verified source metadata.", "",
        "## 3. Initial Data Exploration", ""]
    eda = config.OUTPUT_DIR / "eda" / "eda_summary.md"
    lines.append(eda.read_text(encoding="utf-8") if eda.exists() else "EDA evidence has not been generated; do not invent discoveries.")
    lines += ["", "## 4. Creativity & Uniqueness", "",
        "The seasonal-anchor / planner network-outlook / learned market-mix decomposition gives interpretable planning controls. Historical model-weight selection explicitly waits for matured outcomes. Synthetic negative controls test timing violations; bounded ablations record failures and require independent review. These are engineering choices to justify, not claims of research novelty or guaranteed accuracy gains.", "",
        "## 5. Methodology", "",
        f"Horizons: {cfg['horizons']}. Outer origins: {cfg['backtest_origins']}. Training uses observed labels through origin minus configured lag. Earlier calibration models supply only outcomes matured by the current selection cutoff. Model: histogram gradient boosting on relative log movement, rescaled to the seasonal network outlook. Hyperparameters are fixed for this run.", "",
        "Evaluation is retrospective. The previously inspected historical period is not described as an untouched holdout. External revised-current data must not be described as historical vintage evidence. Features and sensitivity/importance outputs are not causal effects.", "",
        "## 6. Accuracy & Performance", "",
        "WAPE is a fraction of total absolute actuals; MAE and RMSE are in target units. Positive signed bias/mean error indicates overprediction. Report baseline results even if they win.", "", md_table(overall), "",
        "### By horizon", "", md_table(by_horizon), "",
        "### Historical interval evaluation", "", md_table(coverage), "",
        "Uncalibrated historical folds remain unavailable; later residuals are not borrowed. Production interval tables and historical coverage are distinct artifacts. Aggregate intervals require separate calibration and are not sums of market bounds.", "",
        "### Model weight used for production", "", md_table(selected[selected.selected_for_forecast]), "",
        "### Runtime and provenance", "",
        f"Started: {manifest.get('started')}; finished: {manifest.get('finished')}; Python: {manifest.get('python')}; revision: {manifest.get('revision')}; dirty tree: {manifest.get('working_tree_dirty')}.",
        f"Source fingerprint: {manifest.get('source_sha256', 'not recorded in this preliminary run')}. Full config and local input fingerprints: {backtest / 'manifest.json'}.", "",
        "## Limitations and final human checks", "",
        "Confirm observation release/revision timing, duplicate and missing-row meanings, segment definitions, permitted audience, forecast horizons required by the rubric, and an independent rerun. Without airport metadata, geography and domestic/international comparisons are unavailable. Do not reuse earlier repository headline metrics as corrected results."]
    if forecast:
        lines += ["", "## Forecast artifacts", "", str(Path(forecast))]
    if experiment:
        lines += ["", "## Bounded experiment evidence", "", (Path(experiment) / "comparison.md").read_text(encoding="utf-8")]
    directory = config.OUTPUT_DIR / "submission" / backtest.name
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / "final_project_evidence.md"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path
