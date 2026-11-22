# Team 3 contributions

This project combines the team's existing work from both repositories. Names and handles below come from the recorded Git authors; handles are retained where a real name has not been verified.

| Contributor | Work incorporated | Source evidence |
| --- | --- | --- |
| **Vignesh Nagarajan** | Original `aaforecast` package and CLI; Anchor and Adjust forecasting approach, feature/model pipeline, backtests, uncertainty, explanations, scenarios, external-source modules, synthetic tests, methodology and figure generator. Also the Team 3 macro-price analysis. | [Forecasting repository](https://github.com/vignesh-nagarajan-vn/American-Airlines-F26/tree/0d10bc9927903fe99cad76fe8f5e21b18e0ec934); Team commit `ce83225`. |
| **Sashwath Nandhakumar** | COVID impact analysis in `useful_functions/covid_impact_analysis.py`. | Team commit `edbcd1b`. |
| **cheiman2007** | Business indicators and general EDA in `useful_functions/eda_business_indicators.py`. | Team commit `808f44b`. |
| **InTheHiggsField** | Original route/year/month passenger and revenue dictionaries in `useful_functions/Data_Organizer.py`; the integrated adapter preserves this interface. | Team commit `f16cf76`. |
| **Asvath Madhan / AsvM45** | Team repository, roadmap, engineering workflow and documentation; integration and release coordination, with AI-assisted implementation and verification. | Team history through `dc3baeb` and subsequent integration commits. |

The integrated implementation builds on Vignesh's forecasting work. Subsequent integration changes include label-maturity and calibration corrections, observed-target masks, matched-case metrics, input validation, run provenance, controlled ablations, public airport references, reproducible release commands, synthetic CI and local rubric evidence. Integration commit authorship does not replace the original contributors' authorship.

All four original team analysis scripts remain present. Their existing region definitions and exploratory interpretation are documented separately from the forecasting model contract. The [integration manifest](docs/INTEGRATION_MANIFEST.md) accounts for every tracked file from both source snapshots.

Original history remains in each source repository. This integration retains Team 3 history and selectively imports the forecasting source; it does not import the second repository's entire Git ancestry. Historical result write-ups and image artifacts remain at their original source and are not the corrected release evidence. Updated evidence and figures are regenerated locally from approved inputs.

Scientific acceptance and independent human review remain pending; these credits describe verified work incorporated, not approvals.
