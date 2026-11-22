# Team 03 — AI/human engineering rules

**Scope:** Entire private `AsvM45/ADSC_AA_Team3` repository. These rules are a proposal for team adoption, not proof that agents, branch protections, CI or a Project board are running.

## Read before work
1. This file and `docs/PROJECT_VISION.md`.
2. `docs/GITHUB_OPERATING_SYSTEM.md`, `docs/DECISIONS.md`, current issue and latest issue comments/PRs.
3. `docs/TECH_STACK.md` and the relevant forecasting contract before touching modeling.
4. Official club `DELIVERABLES.md` for PM submissions; current Team 3 Drive learning handbook and technical plan for personal routes and engineering task prerequisites.

**Order of authority:** PM/sponsor requirements and confidentiality permissions > verified code/data/branch state and newest accepted team decisions > approved implementation contracts > proposals/historical plans. Never invent approvals, data semantics, measurements or completed tasks.

## Before editing
- Check HEAD, current branch, `git status --short`, active PRs and issues. Never overwrite someone else's modifications.
- Work on one claimed issue with explicit owner, planned changed paths, dependencies, expected tests and human reviewer. Use `human/<slug>` or `agent/<slug>` branch; if two contributors/agents edit simultaneously, each needs a **separate physical clone/worktree**.
- Agree interface changes and any shared-file modifications with the current owner. An issue assignment/claim does **not** launch an AI process.
- An agent may write within its explicitly approved scope, test, commit/push its isolated feature branch and open a draft PR only if the task grants that action. It may not merge, force-push, change repo permissions, spend cloud credits or initiate sponsor contact without express human authorization.

## Scientific invariants
- All features must have been available at each **historical input origin**. Separately enforce **training-label maturity at fit cutoff** and **model-selection evidence cutoff**.
- An origin, forecast target, market and horizon are distinct. State units/market semantics; never guess source schema.
- Count eligible origins and matured labels, especially for horizons of 24 and 36 months. Baselines receive the same eligible information.
- Match forecast keys across candidates; report MAE, RMSE, signed bias, counts, failures and uncertainty honestly. Preserve the final holdout from repeated selection.
- Treat negative results and infeasibility findings as valid evidence. R² is not percentage accuracy; feature importance is not causality.

## Privacy and trust boundaries
- Private GitHub is **not automatically approved** for AA data, summaries, feature importance, model weights, derived metrics, plotted numbers or screenshots.
- By default commit **only general code, synthetic data and public-safe documentation**. Real-data pipelines run in a separately approved environment.
- GitHub Actions use only synthetic/public-safe fixtures. No secrets in issue bodies, PR descriptions, Actions logs or AI prompts.
- Do not upload AA material to personal Databricks/hosted MLflow/LLM services or a new Drive without PM/sponsor approval. `.gitignore` is not sufficient protection; check staged content and full history.
- Do not copy protected information to the official public ADSC repository. Human PM controls official submission.

## Engineering cycle
Read issue -> small plan -> scoped change -> deterministic synthetic/unit tests -> real approved local smoke when authorized -> inspect diff/secrets -> report exact commands/results/failed checks -> request independent review -> human decides merge.
No green-check claims for skipped CI jobs. A review-ready PR is not a completed engineering deliverable. A completed lesson is not an accepted PR.

**Reference:** JAEGERS uses an issue/branch/physical-worktree/PR evidence protocol. Adapt the discipline, not its heavyweight infrastructure.
