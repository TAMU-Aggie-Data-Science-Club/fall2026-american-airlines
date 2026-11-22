# Team 03 — Shared End-State Vision
**Updated November 10, 2026 · Proposed plan, not verified implementation or sponsor approval**

## Mission
Deliver a **reproducible, scientifically defensible market-demand forecasting and automated experimentation system** for American Airlines Network Planning, rather than one model notebook. Forecast 1–24 months ahead, investigate 36-month long-haul international feasibility, and teach all six members to understand their contributions.

Our central engineering differentiator is rigorous **automated ML experimentation and automated validity testing**, including tests designed to catch invalid modeling before any forecast is compared or presented.

## Tier A — defensible core (required)
[Tracked by epic #1](https://github.com/AsvM45/ADSC_AA_Team3/issues/1) and child issues #4–#11.
- PM-approved target, market, data/access policy, observation units and release semantics.
- EDA, validation, correctly timed features and training labels, seasonal-naive baseline.
- Rolling-origin historical evaluation with preserved model-selection cutoff and untouched final holdout.
- Fair matched-case MAE, RMSE, signed bias, case counts and failure reporting.
- Synthetic CI, independent rerun, approved results and official PM acceptance. A negative result is acceptable.

**Acceptance:** A team member can independently run one documented baseline/backtest in approved conditions, inspect evidence and explain its limitations. No accuracy gain is guaranteed.

## Tier B — ambitious final showcase (primary target, conditional on ramp)
[Tracked by epic #2](https://github.com/AsvM45/ADSC_AA_Team3/issues/2) and child issues #12–#17.
- Versioned `experiment` CLI with bounded trials and seeds.
- Six gates: **G0** privacy/config/compute preflight; **G1** schema/data integrity; **G2** origin feature timing, label maturity and selection cutoff; **G3** approved model execution and failures; **G4** matched-case evaluation; **G5** reproducibility/evidence/review.
- Synthetic negative controls deliberately fail on future-known features, immature labels, invalid joins and case mismatch.
- Pooled Ridge or horizon-aware XGBoost challenger, feature-group ablations and simple bounded tuning.
- Immutable trial manifests at first; MLflow+SQLite *locally* only when beneficial and approved.
- Automated comparison report and recorded independent QA; human chooses what to promote.

**Acceptance:** A small configuration change triggers a genuine controlled experiment, tests reject an intentionally invalid input and a reviewer can reproduce the outcomes. No agent auto-merge or automatic promotion.

## Tier C — broader research (conditional)
[Tracked by epic #3](https://github.com/AsvM45/ADSC_AA_Team3/issues/3) and child issues #18–#20.
- Count feasible 36-month long-haul market origins and matured labels; attempt only a defensible experiment.
- Approved historical-vintage macro/capacity/calendar features, clearly hypothetical scenarios.
- Calibrated uncertainty and coverage, coherent aggregate forecasts, and approved local visualization.
- Optional Databricks, Spark or neural experiments **only** if data permissions, compute budget and baseline evidence warrant them.

**Acceptance:** Each researched extension has defensible evidence or a documented negative/infeasible outcome. It never displaces Tier A/B.

## System flow
Approved data environment -> ingest/schema/source-version -> market/target contract -> as-of feature and matured-label builder -> frozen origins/splits -> baseline + bounded ML candidates -> rolling-origin evaluator -> paired score + failure analysis -> experiment registry -> evidence report -> human PM-reviewed submission.

GitHub holds issue contracts, code, tests, synthetic data and public-safe docs. Real AA data, derived artifacts and model training require sponsor-approved storage/compute.

## Final demonstration
**Demo A:** Run a real, reproducible permitted forecast/backtest on a known origin/horizon; show baseline versus challenger, bias, counts, horizons and limitations.
**Demo B:** Change a feature-group hypothesis; run bounded experiment factory; show ledger and matched-case comparison; deliberately inject future info into a *synthetic* trial and demonstrate automatic rejection.

A synthetic demonstration is **labeled as synthetic**, never represented as sponsor-data validation.

## Team learning and capacity
The existing [Team Drive](https://drive.google.com/drive/folders/1Jlm6jUfIagWXPhwDPKvIEHW5PiGNsP61) handbook governs all six personal lesson sequences; technical plan governs shared prerequisites and acceptance. Additional hours are **not yet confirmed** for named members. Older rough scope estimates were ~429 person-hours core including study+contingency and ~1,262 full moonshot including core; re-estimate this experiment-focused backlog from accepted work and actual sustainable hours. Count each person's hour once.

## Date and approval uncertainty
Official D3 was published due **November 9, 2026**; confirm actual status with PM. Previously recorded presentation is **December 1** with only a proposed **November 25** internal finish; do not assert sponsor deadlines without new verification. AA target semantics, data permission, business weighting and earlier cutoff remain pending.

## Repository links
- Team 03 private: https://github.com/AsvM45/ADSC_AA_Team3
- Official ADSC: https://github.com/TAMU-Aggie-Data-Science-Club/fall2026-american-airlines
- Team Drive: https://drive.google.com/drive/folders/1Jlm6jUfIagWXPhwDPKvIEHW5PiGNsP61
- Datathon engineering reference: https://github.com/AsvM45/jaegers

**Provenance:** Synthesized from prior Team 3 technical-plan package and the November 10 shared vision. This file is a tracked proposal; no real AA data were accessed or copied here.
