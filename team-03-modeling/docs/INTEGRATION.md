# Forecasting integration

The implementation imports source, synthetic tests, dependency definitions and methodology from the Team 3 repository `vignesh-nagarajan-vn/American-Airlines-F26`, inspected at commit `0d10bc9` (November 21, 2026). Original authorship and development history remain available in that repository. This is a selective source integration, not a history-rewriting operation or a Git ancestry merge.

Team 3 base: `dc3baebb9ddd8f66e8adc1c95700aea3d7c55eb9`. Package and CLI remain `aaforecast`. Existing `useful_functions/` analyses remain team contributions; they are descriptive tools and currently use definitions that need reconciliation with the model contract. The official ADSC repository is outside this implementation scope.

Imported historical figures and result headlines are deliberately excluded. Corrected result artifacts, input fingerprints, model diagnostics and forecasts are generated under ignored `outputs/` for authorized local review.

Changes after import: matured-label model selection with earlier calibration models; historical-only interval calibration; missing-target observation masks; fail-fast validation; MAE/RMSE and finite matched-case evaluation; explicit calibration compatibility and run provenance; configurable horizons 1–24; minimal forecast mode; passenger-only mode with unknown geography; bounded ablations and failure records; synthetic regressions and CI; narrow source artifact guard; rubric-mapped local evidence report.

The open draft documentation PR #23 predates this integration. Review its guidance against the implemented state; it is not automatically merged or treated as accepted. Independent scientific review and sponsor-approved publication remain human decisions.

The original Team 3 route-dictionary builder is now an import-safe adapter over canonical ingestion, retaining its route/year/month structure and US/Canada region grouping. Existing COVID, business EDA and macro-analysis scripts remain available in `useful_functions/`; they are not replaced by a template exercise. The public airport-reference command supplies a reproducible alternative to the absent sponsor reference, with source provenance and explicitly limited metro/DOT metadata.
