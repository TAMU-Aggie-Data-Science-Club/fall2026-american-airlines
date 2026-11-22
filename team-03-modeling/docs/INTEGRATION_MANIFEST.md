# Integration file manifest

Every tracked path in the two source snapshots is accounted for below. Comparison ignores UTF-8 BOM and line endings; adapted files retain the source functionality with the integration corrections documented in INTEGRATION.md. No confidential input, result value or derived result fingerprint is included.

Forecasting source: `vignesh-nagarajan-vn/American-Airlines-F26` at `0d10bc9927903fe99cad76fe8f5e21b18e0ec934`.

| Original path | Integrated disposition |
| --- | --- |
| `.gitignore` | Integrated and adapted; original authorship retained. |
| `README.md` | Integrated and adapted; original authorship retained. |
| `docs/figures/accuracy_by_horizon.png` | Historical result artifact retained in source repository; corrected outputs regenerated locally and excluded from Git. |
| `docs/figures/accuracy_by_level.png` | Historical result artifact retained in source repository; corrected outputs regenerated locally and excluded from Git. |
| `docs/figures/accuracy_by_size_segment.png` | Historical result artifact retained in source repository; corrected outputs regenerated locally and excluded from Git. |
| `docs/figures/driver_bridge.png` | Historical result artifact retained in source repository; corrected outputs regenerated locally and excluded from Git. |
| `docs/figures/feature_importance.png` | Historical result artifact retained in source repository; corrected outputs regenerated locally and excluded from Git. |
| `docs/figures/hub_accuracy.png` | Historical result artifact retained in source repository; corrected outputs regenerated locally and excluded from Git. |
| `docs/figures/hub_outlook.png` | Historical result artifact retained in source repository; corrected outputs regenerated locally and excluded from Git. |
| `docs/figures/interval_coverage.png` | Historical result artifact retained in source repository; corrected outputs regenerated locally and excluded from Git. |
| `docs/figures/network_outlook.png` | Historical result artifact retained in source repository; corrected outputs regenerated locally and excluded from Git. |
| `docs/figures/scenarios.png` | Historical result artifact retained in source repository; corrected outputs regenerated locally and excluded from Git. |
| `docs/figures/shrink_calibration.png` | Historical result artifact retained in source repository; corrected outputs regenerated locally and excluded from Git. |
| `docs/figures/significance.png` | Historical result artifact retained in source repository; corrected outputs regenerated locally and excluded from Git. |
| `docs/methodology.md` | Integrated and adapted; original authorship retained. |
| `docs/results.md` | Historical result artifact retained in source repository; corrected outputs regenerated locally and excluded from Git. |
| `pyproject.toml` | Integrated and adapted; original authorship retained. |
| `requirements.txt` | Integrated and adapted; original authorship retained. |
| `scripts/make_figures.py` | Integrated and adapted; original authorship retained. |
| `src/aaforecast/__init__.py` | Retained unchanged (normalized text). |
| `src/aaforecast/__main__.py` | Retained unchanged (normalized text). |
| `src/aaforecast/backtest.py` | Integrated and adapted; original authorship retained. |
| `src/aaforecast/calendar.py` | Integrated and adapted; original authorship retained. |
| `src/aaforecast/cli.py` | Integrated and adapted; original authorship retained. |
| `src/aaforecast/config.py` | Integrated and adapted; original authorship retained. |
| `src/aaforecast/data.py` | Integrated and adapted; original authorship retained. |
| `src/aaforecast/eda.py` | Integrated and adapted; original authorship retained. |
| `src/aaforecast/explain.py` | Retained unchanged (normalized text). |
| `src/aaforecast/external/__init__.py` | Retained unchanged (normalized text). |
| `src/aaforecast/external/catchment.py` | Retained unchanged (normalized text). |
| `src/aaforecast/external/census.py` | Retained unchanged (normalized text). |
| `src/aaforecast/external/faa.py` | Retained unchanged (normalized text). |
| `src/aaforecast/external/fred.py` | Retained unchanged (normalized text). |
| `src/aaforecast/external/http.py` | Integrated and adapted; original authorship retained. |
| `src/aaforecast/features.py` | Integrated and adapted; original authorship retained. |
| `src/aaforecast/forecast.py` | Integrated and adapted; original authorship retained. |
| `src/aaforecast/geo.py` | Integrated and adapted; original authorship retained. |
| `src/aaforecast/metrics.py` | Integrated and adapted; original authorship retained. |
| `src/aaforecast/model.py` | Retained unchanged (normalized text). |
| `src/aaforecast/pipeline.py` | Integrated and adapted; original authorship retained. |
| `src/aaforecast/report.py` | Integrated and adapted; original authorship retained. |
| `src/aaforecast/scenario.py` | Integrated and adapted; original authorship retained. |
| `src/aaforecast/stats_tests.py` | Retained unchanged (normalized text). |
| `src/aaforecast/validate.py` | Integrated and adapted; original authorship retained. |
| `tests/conftest.py` | Retained unchanged (normalized text). |
| `tests/test_data.py` | Integrated and adapted; original authorship retained. |
| `tests/test_features_model.py` | Integrated and adapted; original authorship retained. |
| `tests/test_geo_calendar.py` | Retained unchanged (normalized text). |
| `tests/test_metrics.py` | Retained unchanged (normalized text). |

Team repository base: `AsvM45/ADSC_AA_Team3` at `dc3baebb9ddd8f66e8adc1c95700aea3d7c55eb9`.

| Original team path | Integrated disposition |
| --- | --- |
| `.github/ISSUE_TEMPLATE/engineering_task.md` | Retained unchanged (normalized text). |
| `.github/PULL_REQUEST_TEMPLATE.md` | Retained unchanged (normalized text). |
| `.gitignore` | Retained and adapted for integrated implementation. |
| `AGENTS.md` | Retained unchanged (normalized text). |
| `AGENT_START_HERE.md` | Retained and adapted for integrated implementation. |
| `README.md` | Retained and adapted for integrated implementation. |
| `TEAM_START_HERE.md` | Retained unchanged (normalized text). |
| `docs/DECISIONS.md` | Retained unchanged (normalized text). |
| `docs/GITHUB_OPERATING_SYSTEM.md` | Retained unchanged (normalized text). |
| `docs/PROJECT_VISION.md` | Retained unchanged (normalized text). |
| `docs/TECH_STACK.md` | Retained and adapted for integrated implementation. |
| `useful_functions/Data_Organizer.py` | Retained and adapted for integrated implementation. |
| `useful_functions/covid_impact_analysis.py` | Retained and adapted for integrated implementation. |
| `useful_functions/eda_business_indicators.py` | Retained and adapted for integrated implementation. |
| `useful_functions/macro_price_analysis.py` | Retained and adapted for integrated implementation. |

New integration modules, release scripts, configurations, tests and documentation are additional work; the table above accounts for the original files rather than assigning their authorship to the integration committer.

See [named contributions](../CONTRIBUTORS.md), [integration corrections](INTEGRATION.md), and [release contract](RELEASE_CONTRACT.md).
