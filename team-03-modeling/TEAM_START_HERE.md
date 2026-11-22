# Team 03 — start here

**Goal:** Build a reproducible American Airlines market-demand forecasting system with an automated ML experiment-and-testing platform, not just notebooks.

## What we are trying to deliver
- **A — required:** approved market/target definitions, clean data audit, seasonally aware baseline, valid rolling-origin backtest, MAE/RMSE/bias and independent reproduction. [Core epic #1](https://github.com/AsvM45/ADSC_AA_Team3/issues/1).
- **B — our ambitious final demo:** automated experiment runner, synthetic leakage-failure tests, pooled ML model, feature ablations, run registry and auto-generated comparisons. [Experiment epic #2](https://github.com/AsvM45/ADSC_AA_Team3/issues/2).
- **C — research if feasible:** 36-month long-haul, approved vintage external variables, uncertainty, scenarios and dashboard. [Research epic #3](https://github.com/AsvM45/ADSC_AA_Team3/issues/3).

## How we work
1. Read this note, relevant handbook lesson and the issue you want to work on.
2. Claim a bounded issue with a human owner/reviewer and define the expected files, tests and learning prerequisite.
3. Use a separate `human/<name-issue>` feature branch. If AI agents work in parallel, they require separate physical worktrees and file ownership.
4. Write a small change, run synthetic tests, understand it, open a PR and request independent review.
5. Record blockers honestly. Don't count a lesson, generated code or a passed CI check as an accepted engineering deliverable.

**Never commit released AA data, real-data-derived figures or logs** to this repository without written PM/sponsor authorization. Keep any real-data analysis in approved locations.

## Important references
- [Complete project vision](docs/PROJECT_VISION.md)
- [GitHub workflow](docs/GITHUB_OPERATING_SYSTEM.md)
- [Technical stack](docs/TECH_STACK.md)
- [Team learning handbook and technical plan (Drive)](https://drive.google.com/drive/folders/1Jlm6jUfIagWXPhwDPKvIEHW5PiGNsP61)
- [Official ADSC deliverables](https://github.com/TAMU-Aggie-Data-Science-Club/fall2026-american-airlines/blob/main/DELIVERABLES.md)

**Current stage:** Planning issues created; tests and forecast system not implemented; D3 published Nov 9 deadline needs PM status check. Hours/assignments are not yet confirmed.
