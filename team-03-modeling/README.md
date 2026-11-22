# ADSC American Airlines — Team 3

**Team contributions:** This repository integrates Vignesh Nagarajan's forecasting implementation with the team's dictionary, COVID, business EDA and macro-price analyses. See [contributors](CONTRIBUTORS.md) and the [file-by-file integration manifest](docs/INTEGRATION_MANIFEST.md).

**Status:** Integrated forecasting implementation; independent review and sponsor acceptance pending. **No raw AA extracts belong in this repository.**

## How the final work is evaluated

The sponsor review asks six questions. The package is built to answer them in that order. [The evaluation brief](docs/EVALUATION.md) states each answer, the module that implements it, and the measured result from the completed local run.

| Question | Code |
| --- | --- |
| Problem-Solving Approach | `src/aaforecast/model.py`, `pipeline.py`, `backtest.py` |
| Data Aggregation | `src/aaforecast/data.py`, `validate.py` |
| Initial Data Exploration | `src/aaforecast/eda.py` |
| Creativity & Uniqueness | `model.py`, `backtest.py`, `experiments.py` |
| Methodology | `docs/methodology.md`, `features.py`, `config.py` |
| Accuracy & Performance | `metrics.py`, `submission.py` |

Our mission is to build a reproducible market-demand forecasting and **automated ML experimentation** system for the American Airlines Network Planning project. We target 1–24-month forecasts and investigate 36-month long-haul forecasting if historical evidence supports evaluation.

## Release tiers

- **A — Defensible core:** Verified target/permissions, reproducible ingestion and EDA, time-safe forecasts, seasonal baseline, matched-case backtests, synthetic tests, independent rerun, honest report.
- **B — Ambitious target:** Everything in A plus a bounded automated experiment factory, recorded trials and failures (locally hosted MLflow when justified), a pooled ML challenger, ablations, rigorous preflight gates and independent QA.
- **C — Research stretch:** Approved external historical vintages, uncertainty, aggregation coherence, scenarios, long-haul 36-month experiments if feasible, and optional dashboard or neural-model studies.

**Tier B is our preferred final showcase; Tier A remains the non-negotiable valid fallback.**

## Connected sources of truth

- [Official club repo](https://github.com/TAMU-Aggie-Data-Science-Club/fall2026-american-airlines) — sponsor deliverables, club submission rules, PM acceptance.
- [Team learning/technical Drive](https://drive.google.com/drive/folders/1Jlm6jUfIagWXPhwDPKvIEHW5PiGNsP61) — existing learning handbook and technical plan, preserved unchanged.
- [JAEGERS](https://github.com/AsvM45/jaegers) — process inspiration only (agent issues, isolated branches, PR review and synthetic CI).

## Workflow

GitHub issues hold bounded tasks and evidence; GitHub Projects, when configured, shows the *same issues* by release tier/status; feature branches and PRs implement accepted tasks; CI checks synthetic data only; human reviewers gate merges. Agent comments do not launch agents.

Read the issue backlog and docs before implementation. **This repository is not automatically an approved processing or storage environment for the released AA data.**

## Start here
- [Six-person team guide](TEAM_START_HERE.md)
- [AI agent quickstart](AGENT_START_HERE.md) and [binding agent rules](AGENTS.md)
- [Three-tier delivery vision](docs/PROJECT_VISION.md)
- [GitHub workflow and Project board setup](docs/GITHUB_OPERATING_SYSTEM.md)
- [Technical stack](docs/TECH_STACK.md) and [open decisions](docs/DECISIONS.md)
- [Tier A core](https://github.com/AsvM45/ADSC_AA_Team3/issues/1), [Tier B automated experimentation](https://github.com/AsvM45/ADSC_AA_Team3/issues/2), [Tier C research](https://github.com/AsvM45/ADSC_AA_Team3/issues/3)

**Current implementation:** The combined repository contains the `aaforecast` forecasting package, synthetic regression tests, a bounded local ablation runner and a synthetic-only CI definition. Independent scientific review, sponsor acceptance and the Project board remain separate human tasks. See [final readiness](docs/FINAL_READINESS.md), [integration provenance](docs/INTEGRATION.md), [release contract](docs/RELEASE_CONTRACT.md) and [supplementary sources](docs/EXTERNAL_SOURCES.md).

The forecast has three layers: each market's seasonal anchor, a planner-controlled network outlook, and a pooled gradient-boosted model of relative market movement. [Methodology](docs/methodology.md) explains the features, baselines, historical calibration, intervals and limitations. The experiment runner compares bounded feature ablations on identical forecast cases and records failures instead of silently promoting a winner.

## Run locally

Tested environment: Python 3.13. Install the recorded dependency versions, then the editable package:

```powershell
python -m venv .venv
.venv/Scripts/python -m pip install -r requirements-lock.txt
.venv/Scripts/python -m pip install --no-deps -e .
.venv/Scripts/python -m pytest -q
.venv/Scripts/python -m aaforecast.privacy
```

Keep approved inputs in `data/raw/`, or set `AAF_PAX_FILE` and `AAF_AIRPORT_FILE` to existing approved local paths. No source rows, fitted models or result artifacts belong in commits. Example commands:

```powershell
.venv/Scripts/python -m aaforecast fetch-airports
.venv/Scripts/python -m aaforecast validate
.venv/Scripts/python -m aaforecast eda
.venv/Scripts/python -m aaforecast backtest
.venv/Scripts/python -m aaforecast forecast --backtest-run <completed-run-directory>
.venv/Scripts/python -m aaforecast experiment --experiment-config configs/ablations.json
.venv/Scripts/python -m aaforecast submission-report --backtest-run <completed-run-directory> --forecast-run <forecast-directory>
```

`--passenger-only` explicitly permits a run without airport metadata; geographic and domestic/international results are then unavailable. The submission run is `configs/release.json`: horizons 12–24, with a horizon omitted when its target month is not in the extract yet. A bare `backtest` command still uses 12–18 months and a three-month origin step; pass the release config to reproduce the submitted numbers. `--fast` is a development smoke run, not final evidence. Default forecasts skip expensive explanations/scenarios; use `--full-details` after validating the minimal run.

External features are excluded by default. The imported fetchers support the supplementary Census/FAA/county sources, but source releases and historical vintages require review before a historical-availability claim. Use `fetch` to inspect local source retrieval status and `--external` only for an explicitly reviewed experiment.

Backtests, forecasts, experiment registries and the six-section presentation evidence report are written under ignored `outputs/`. Each new run has a distinct directory and manifest; forecast calibration requires an explicitly selected compatible completed backtest. Historical results from the source repository are not the corrected integration results. Human review is still required before any final submission or publication.

To reproduce the release configuration, including its six-month training-origin spacing and optional controlled ablations:

```powershell
.venv/Scripts/python scripts/reproduce_release.py --pax-file <approved-passenger-csv> --ablations
```

The public airport reference must be fetched first, or supplied using `--airport-file`. `configs/release.json` records the exact run settings. Existing team analyses remain runnable as `useful_functions/covid_impact_analysis.py`, `eda_business_indicators.py`, and `macro_price_analysis.py` with their `--pax-csv`, `--airport-csv`, and `--out` arguments. The route dictionaries use `useful_functions/Data_Organizer.py` and the same canonical input paths. All their result directories stay local.

The original forecasting figure generator is also integrated. Rebuild its figures with `scripts/make_figures.py --backtest-dir <completed-run> --forecast-dir <compatible-forecast-run>`. It saves to ignored `outputs/figures/`, checks run provenance, and records unavailable optional charts rather than reusing older figures. `--full-details` on the forecast command enables companion explanations and diagnostics; their interpretation remains subject to the documented limitations.
