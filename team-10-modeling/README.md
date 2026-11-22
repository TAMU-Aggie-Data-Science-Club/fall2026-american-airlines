# Team 10 modeling — two-year passenger forecast

**Question (Deliverables 3–5):** how many passengers will each origin–destination route carry in
each of the next 24 months, forecast once, from today?

**Target:** `OD_PAX` per route per departure month (`AA_PAX_FORECASTING`), with airport attributes
from `AA_AIRPORT_INFO`. **Reference forecast:** the seasonal naive (same route, same month, the year
before the forecast origin). **Challengers:** year-over-year growth models (Ridge and gradient-boosted
trees) applied on top of that reference.

## Real data vs fake data

The AA files are under NDA and are **not** in this repo. The code looks for them in the git-ignored
`data/raw/` folder (`*PAX_FORECASTING*.csv` and `*AIRPORT_INFO*.csv`, from the team Drive link).

- **Files present → REAL mode.** Tables and figures go to `data/interim/team-10/` (git-ignored).
  Copy only rounded numbers into the reports in `results/`, and ask a PM before committing a
  real-data chart.
- **Files absent → FAKE mode.** `scripts/make_fake_data.py` generates a random panel with the same
  columns in `data/fake/` (git-ignored). Outputs go to `results/generated/` and `visuals/`, and every
  table and chart is stamped **FAKE DATA**. They only prove the pipeline runs. They say nothing
  about American Airlines.

The first line of every notebook prints which mode it is in.

<!-- FILL:readme_status -->
**Reports in results/ were filled from a real-data run on 2026-11-22** (rounded error metrics only).
<!-- /FILL -->

## Run order

From the repo root, with the D1–D2 packages installed (pandas, numpy, matplotlib, scikit-learn):

| # | notebook | deliverable | what it decides |
|---|---|---|---|
| 1 | `notebooks/01_data_audit.ipynb` | D3 | Grain, duplicates, missing cells vs missing route-months, airport join |
| 2 | `notebooks/02_eda_demand.ipynb` | D3 | Skew, seasonality, the 2020 disruption window, the growth cap (with CHECK lines) |
| 3 | `notebooks/03_leakage_and_features.ipynb` | D3 | What is known at the origin for all 24 horizons; the feature frame |
| 4 | `notebooks/04_model_comparison.ipynb` | D4 | Scores four candidates at two origins; applies the pre-registered carry rule |
| 5 | `notebooks/05_two_year_forecast.ipynb` | D5 | Tests the carried model once, builds the 80% band, diagnostics, forward forecast |

Then run `python team-10-modeling/scripts/build_deliverables.py`. It reads the scores that 04 and
05 saved and writes filled copies of the reports (rounded numbers only) plus the final slide deck
(`team10_final_slides.pptx`, needs `pip install python-pptx`). Everything goes to
`data/interim/team-10/` in REAL mode. Review the reports there, then copy them into `results/`.
`TEAM10_FORCE_FAKE=1` forces FAKE mode for testing even when the real files are present.

Shared code lives in `src/`: `data.py` (files, modes, output folders), `features.py` (cleaning,
airport and route features, training pairs) and `forecasting.py` (origins, forecast sets,
candidates, scoring, carry rule). Notebooks import it and never copy logic.

## Reports

| file | contents |
|---|---|
| `results/feature-set.md` | D3: every kept, engineered and dropped input, and why |
| `results/model-comparison.md` | D4: validation design, candidates, metrics, trade-offs |
| `results/baseline-decision.md` | D4: the carry rule (fixed before scoring) and how the decision is recorded |
| `results/forecast-evaluation.md` | D5: test design, band, diagnostics, whether the data supports two years |
| `results/final-model-report.md` | D5: training procedure, what the model does, business use, risks, monitoring |
| `results/generated/` | Tables written by the notebooks (FAKE mode only; REAL mode writes to `data/interim/team-10/`) |

## Design notes

The structure follows the pattern of teams that have already finished D3–D5: EDA → leakage →
feature set → a fixed "beat the seasonal naive" bar → one-shot 24-month forecast. It was written
independently for team 10. Two choices are our own:

- **Growth on top of the naive.** Fitted models predict year-over-year log growth, capped at
  0.8×–1.25× per year, instead of raw passenger levels. A level model trained across 2019–2022 reads
  the collapse and rebound as a trend.
- **A clean band calibration.** The 80% band is calibrated only on months before the test window.

An earlier side analysis on a synthetic fare file is kept in `fare-side-study/`. It is not part of
the submission.
