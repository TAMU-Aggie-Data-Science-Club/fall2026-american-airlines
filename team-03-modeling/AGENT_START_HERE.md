# Team 03 — agent boot capsule

## Mission
A **reproducible 1–24-month market-demand forecasting system and bounded automated ML experiment factory** for AA Network Planning. A serious 36-month long-haul feasibility study is a stretch. Negative results and a seasonal baseline winner are acceptable.

## Read order
1. [AGENTS.md](AGENTS.md): binding scope, safety and write rules.
2. [Vision](docs/PROJECT_VISION.md), [GitHub workflow](docs/GITHUB_OPERATING_SYSTEM.md), [Decisions](docs/DECISIONS.md), [Stack](docs/TECH_STACK.md).
3. Current issue and all comments, current PRs, branch, HEAD, `git status`, worktrees.
4. Official club deliverables and relevant learning/technical plan at the [team Drive](https://drive.google.com/drive/folders/1Jlm6jUfIagWXPhwDPKvIEHW5PiGNsP61).

**Nothing in this repo represents the raw AA schema, approved data transfer, completed experiment, or sponsor signoff.**

## Issue and PR route
Epic A #1 has tasks #4–#11; B #2 has #12–#17; C #3 has #18–#20. Do not implement unclaimed issues or edit other agents' workspaces. One active implementing owner per issue and exclusive claimed file scope; a separate physical worktree per concurrent agent. AI may implement and test in explicitly granted scope, but **cannot self-merge or autonomously publish/transfer sponsor materials**.

## Scientific hard rules
Observe historical input origin, feature release timestamp, distinct training label fit cutoff, distinct model-selection cutoff, final holdout untouched for tuning. Count eligible cases, especially h=36. Baselines and ML share identical inputs and forecast-key evaluation. Report MAE, RMSE, signed bias, case counts and failures. No causal inference from feature importance.

## Testing hard rules
Synthetic-only GitHub CI. Run known-good tests and deliberate negative controls: future-only features, label immaturity, duplicate/invalid joins, mismatched forecast keys, source vintage violation and final holdout misuse. Report exact commands and failures; never invent success. Experiment factory gates G0–G5 are proposals until implemented.

## Security hard rules
No AA raw/cleaned rows, screenshots, weights, metrics, forecast artifacts or private logs in GitHub, Actions artifacts, public club PRs or hosted agents without approval. Personal Databricks is not the approved path. Local MLflow is optional in an authorized environment; a private repo is not automatically an authorized data store.

## Current state
The November 21 integration branch imports `aaforecast`, synthetic tests and methodology from the companion Team 3 repository. It adds temporal regressions, bounded local ablations, run manifests, a source artifact guard and a synthetic CI definition. Read `docs/INTEGRATION.md` and `docs/RELEASE_CONTRACT.md` before implementation. Local test results and completed manifests establish only their recorded checks; remote CI execution, independent scientific review, Project board setup and sponsor acceptance must be verified separately.
