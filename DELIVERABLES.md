# Deliverables & Timeline

## Milestones at a glance

| # | Deliverable | Description | Owner | Branches | Target |
|---|-------------|-------------|-------|----------|--------|
| 1 | GitHub Fundamentals + ML Starter | Git workflow practice + intro XGBoost notebook on flight price prediction | All members, individually | `deliverable1/<name>` → `main` | ✅ Week 1 (Oct 6–12, 2026) |
| 2 | EDA, Data Leakage & Choosing the Right Model | Deeper EDA tied to modelling decisions; data leakage and what it cost; Lasso vs Ridge; loss functions (MSE / MAE / Huber / quantile); XGBoost vs Random Forest vs ARIMA. *Absorbs the former "EDA & Data Cleaning" and "Model Development & Evaluation" milestones — members cannot evaluate a model honestly until they can spot leakage, so the two were being taught in the wrong order.* | All members, in teams | `team-##/<name>` → `team-##/main` → `main` | Week 3–4 (due Nov 2, 2026) |
| 3 | Tuning & Error Analysis | Hyperparameter search over the chosen model, learning curves, regularisation, segment-level error analysis | Teams | `team-##/<name>` → `team-##/main` → `main` | Weeks 5–6 |
| 4 | Final Findings & Presentation | Report / dashboard / model artifact for American Airlines | Teams + PM | `team-##/<name>` → `team-##/main` → `main` | Weeks 7–8 |

> **Deliverables 2–4 are team deliverables.** Each member works on their own branch, merges up into their team's single **trunk** branch, and only that trunk opens a PR into `main` — reviewed and merged by a PM. The suggested names are `team-##/<your-name>` and `team-##/main`, but **the structure is what matters, not the names.** See [Branch conventions](#branch-conventions) below.

---

## Deliverable 1 — GitHub Fundamentals & ML Starter

> **Due:** October 12, 2026  
> **Owner:** Every team member individually  
> **Branch convention:** `deliverable1/<your-name>` (e.g. `deliverable1/alex-chen`)
>
> **Note:** this solo-branch-into-`main` convention applied to Deliverable 1 only. From Deliverable 2 onward we use team branches — see [Deliverable 2](#deliverable-2--eda-data-leakage--choosing-the-right-model) and [Branch conventions](#branch-conventions).

### What this is

Before we write production-quality ML code, everyone needs to be comfortable with the team's git workflow. This deliverable is a lightweight, hands-on exercise: you will create your own branch, complete a starter Jupyter notebook, and open a pull request back into `main`. The notebook itself walks you through pandas & numpy analysis and an intro XGBoost model for predicting flight prices.

### GitHub steps (do these first)

1. **Clone the repo** (if you haven't already):
   ```sh
   git clone <repo-url>
   cd fall2026-american-airlines
   ```

2. **Sync with main** (always do this before branching):
   ```sh
   git switch main
   git pull origin main
   ```

3. **Create your branch:**
   ```sh
   git switch -c deliverable1/<your-name>
   git push -u origin deliverable1/<your-name>
   ```

4. **Download the dataset** from Kaggle (link inside the notebook) and place the CSV at:
   ```
   deliverable1/data/Clean_Dataset.csv
   ```
   > `deliverable1/data/` is git-ignored — **do not commit the CSV file.**

5. **Open and work through** the starter notebook in `deliverable1/`. Fill in every `# TODO` cell and answer the reflection questions at the end.
   > **Note (updated):** the shared `flight_price_prediction.ipynb` template has since been replaced on `main` by per-member copies (e.g. `flight_price_prediction_Adi_Nounou.ipynb`). Work in your own named copy; that is what the merged Deliverable 1 PRs settled on.

6. **Commit your work** with a clear message:
   ```sh
   git add deliverable1/<your-notebook>.ipynb
   git commit -m "deliverable1: complete EDA and XGBoost starter notebook"
   ```

7. **Push your branch:**
   ```sh
   git push
   ```

8. **Open a pull request** on GitHub from `deliverable1/<your-name>` → `main`. Use the PR template and make sure your notebook runs top-to-bottom without errors.

### Acceptance criteria

- [ ] Your branch is pushed to GitHub.
- [ ] The notebook runs top-to-bottom with no uncaught errors (Kernel → Restart & Run All).
- [ ] All `# TODO` cells have been completed.
- [ ] Reflection questions in the final section are answered in markdown.
- [ ] A PR is open against `main` with a filled-in description.

---

## Deliverable 2 — EDA, Data Leakage & Choosing the Right Model

> **Due:** Monday, November 2, 2026
> **Owner:** every member, working inside an assigned team
> **Branch convention:** `team-##/<your-name>` → PR into `team-##/main` → PM merges `team-##/main` into `main`
> **Full brief:** [`deliverable2/README.md`](deliverable2/README.md)
> **Notebook:** [`deliverable2/flight_price_modeling.ipynb`](deliverable2/flight_price_modeling.ipynb)

### What this is

Deliverable 1 handed everyone one model — `XGBRegressor` — and they trained it. Two questions were never asked: **was that 0.98 R² real**, and **was XGBoost the right choice?**

This deliverable answers both, and it is where members stop running a recipe and start making modelling decisions.

| Section | Topic | What members actually do |
|---|---|---|
| **2** | **Deeper EDA** | Split the price distribution by cabin and see two populations, not one skewed hump; test whether the relationships are even linear; plot the **booking curve** over `days_left`; tie every observation to a modelling consequence in a summary table |
| **3** | **Data leakage** | Reproduce the Deliverable 1 result leaks-included, then remove its three leaks one at a time and measure what each was worth; see a target-leaking feature score 0.999 and understand why it is worthless; replace ad-hoc preprocessing with a `Pipeline`; read cross-validation as `mean ± std` |
| **4** | **Lasso vs Ridge** | The L1 vs L2 penalty, why L1 produces exact zeros and L2 does not, why scaling is mandatory, an `alpha` sweep showing the bias–variance tradeoff with a dial on it, and reading which features Lasso switched off |
| **5** | **Loss functions** | Loss vs metric; MSE predicts the mean while MAE predicts the median; Huber; **quantile loss** and prediction intervals with a coverage check; and a callout clearing up that **ReLU is an activation function, not a loss** |
| **6** | **XGBoost vs RF vs ARIMA** | Bagging vs boosting, head-to-head with fit times; then ARIMA on the booking curve, scored against a naive baseline — and the conclusion that it answers a different question entirely |
| **7** | **Defend your choice** | Pick one model and give three reasons that are not "it scored best"; then what this dataset can and cannot tell American Airlines |

### Why it is shaped this way

Members finished Deliverable 1 with a number that looks like a solved problem and one model they never chose. The two most valuable things to teach next are that the number was partly an illusion, and that picking a model is a decision with reasons behind it.

The sequencing is deliberate: **leakage comes before model comparison**, because comparing models on a leaky split just ranks them on who memorised best. And every modelling choice in Sections 4–6 is pointed back at an observation from the EDA in Section 2 — the skewed bimodal target motivates the loss-function section, and the booking curve is what makes the ARIMA section possible at all.

> **A note on ARIMA.** This dataset has **no dates** — no departure date, no booking date, no timestamp — so it is a cross-section with almost nothing for a time-series model to index. We include ARIMA anyway, applied to the one genuine sequence in the data: the booking curve (mean price by `days_left`, ~49 ordered points). The section is framed around the honest conclusion — ARIMA forecasts an aggregate curve and **cannot price an individual ticket** — because recognising that a model family does not match your data shape is a more valuable lesson than another tuned score. Members must beat a naive last-value baseline and report honestly if they cannot.

### How members and teams should approach it

The notebook opens with a **"How to approach this deliverable"** section that covers this in
full; the short version, so PMs can reinforce it:

**The written answers are the deliverable, not the plots.** Every section ends with a markdown
prompt asking *why*. When a cell says "plot a histogram", the deliverable is the sentence
*"...therefore I should ___ when I model."* A plot with no conclusion attached is worth nothing,
and a number with no interpretation is worth less than no number, because it invites someone to
act on it.

**Expect the R² to go down, and treat that as success.** Section 3 shows members their
Deliverable 1 score was inflated. The honest number is lower, and that is the correct outcome.
Nobody should be tuning their way back up to 0.98 — that is just recreating the bug.

**How to think about choosing a model.** The instinct from Deliverable 1 is "run them all, keep
the best score". The notebook teaches the questions in this order, because each can eliminate
options before any compute is spent:

1. **What shape is my data, and what shape does this model require?** Disqualifying, so it goes
   first. ARIMA needs an ordered series; without one, no tuning helps.
2. **What am I asking the model to minimise?** Squared error is a choice that has been made
   silently until now.
3. **What does the stakeholder need?** A readable Ridge model can beat a boosted ensemble that
   scores two points higher, if the client has to explain the price to a customer.
4. **What does it cost?** Fit time, tuning burden, ways it can break in production.
5. **Only then: what scores best?**

If a member's final justification is "it had the highest R²", they have skipped the deliverable.

**Working the code.** Run cells in order — later sections depend on earlier variables. Inspect
the result of every transform (`.head()`, `.shape`, `.value_counts()`), because most bugs in data
code are silent. Some cells are slow (Random Forest on 240k rows, 5-fold CV); **subsampling and
noting it in the PR is a normal engineering decision, not cheating.** Commit per section.

**Working as a team.** Each member completes their **own full copy** of the notebook — the work
is individual because the learning is individual. What the team does **together** is agree on the
conclusions and write them into `TEAM_REPORT.md` on `team-##/main`. Beyond that, how a team
organises itself is the team's call; they should decide in their first meeting and record it
where their PM can see it. The one habit worth enforcing: **clear outputs before every commit**,
which prevents the large majority of notebook merge conflicts.

### References

All links below were checked against the live pages and point at the specific section needed,
not a landing page or a search result. The same list, organised by notebook section, is in the
notebook itself.

**Start here**

- [scikit-learn: Common pitfalls and recommended practices](https://scikit-learn.org/stable/common_pitfalls.html) — the most useful single page for this deliverable; shows a "Wrong" and a "Right" version of each mistake
- [An Introduction to Statistical Learning](https://www.statlearning.com/) — the standard undergraduate text, **free PDF**. Ch. 6 is Ridge/Lasso, Ch. 8 is trees and boosting. A Python edition (ISLP) is also available.
- [Dataset — Flight Price Prediction (EaseMyTrip)](https://www.kaggle.com/datasets/shubhambathwal/flight-price-prediction)

**Leakage & validation (§3)**

- [Data leakage](https://scikit-learn.org/stable/common_pitfalls.html#data-leakage) — worked example of a score collapsing from 0.76 to chance once the leak is fixed
- [Inconsistent preprocessing](https://scikit-learn.org/stable/common_pitfalls.html#inconsistent-preprocessing)
- [Pipelines and composite estimators](https://scikit-learn.org/stable/modules/compose.html)
- [Cross-validation](https://scikit-learn.org/stable/modules/cross_validation.html)

**Lasso vs Ridge (§4)**

- [Ridge regression](https://scikit-learn.org/stable/modules/linear_model.html#ridge-regression-and-classification) · [Lasso](https://scikit-learn.org/stable/modules/linear_model.html#lasso) · [Elastic-Net](https://scikit-learn.org/stable/modules/linear_model.html#elastic-net)
- [Common pitfalls in interpreting linear coefficients](https://scikit-learn.org/stable/auto_examples/inspection/plot_linear_model_coefficient_interpretation.html) — why scale matters, why coefficients are conditional rather than marginal, and how correlated features destabilise them

**Loss functions (§5)**

- [XGBoost learning-task parameters](https://xgboost.readthedocs.io/en/stable/parameter.html#learning-task-parameters) — the full `objective` list. Note the version floors: `reg:absoluteerror` was added in **1.7.0** and `reg:quantileerror` in **2.0.0**.
- [Prediction intervals for gradient boosting regression](https://scikit-learn.org/stable/auto_examples/ensemble/plot_gradient_boosting_quantile.html) — the reference implementation for §5's quantile models and coverage check; its own conclusion is that *each model performs best on the metric it was trained for*
- [Regression metrics](https://scikit-learn.org/stable/modules/model_evaluation.html#regression-metrics) — MAE, RMSE, R², `mean_pinball_loss`

**Model families (§6)**

- [XGBoost: Introduction to Boosted Trees](https://xgboost.readthedocs.io/en/stable/tutorials/model.html) — the clearest explanation of additive training available, and it explicitly notes that random forests and boosted trees share the same model form and differ in *how they are trained*
- [Random forests](https://scikit-learn.org/stable/modules/ensemble.html#random-forests) · [Gradient-boosted trees](https://scikit-learn.org/stable/modules/ensemble.html#gradient-boosted-trees)
- [statsmodels ARIMA API](https://www.statsmodels.org/stable/generated/statsmodels.tsa.arima.model.ARIMA.html) — the `order=(p, d, q)` signature
- [Forecasting: Principles and Practice, Ch. 9 — ARIMA](https://otexts.com/fpp3/arima.html) · [§9.1 Stationarity and differencing](https://otexts.com/fpp3/arima.html#sec-stationarity)

### New dependency

`statsmodels`, for ARIMA. Everything else is what Deliverable 1 already required:

```sh
pip install pandas numpy matplotlib seaborn scikit-learn xgboost statsmodels
```

### GitHub steps

The branch model changed. Deliverable 1 was solo PRs into `main`, which does not scale to 70+ members.

1. **One member per team** creates the team trunk, once:
   ```sh
   git switch main && git pull origin main
   git switch -c team-06/main          # your team's number, zero-padded
   git push -u origin team-06/main
   ```
   If that name is refused with `cannot lock ref`, a plain `team-##` branch already occupies
   the name — use `team-##-main` instead and tell your team. Nothing needs deleting.
2. **Every member** branches off the trunk — not off `main`:
   ```sh
   git fetch origin
   git switch -c team-06/alex-chen origin/team-06/main
   git push -u origin team-06/alex-chen
   ```
3. **Copy the dataset** from Deliverable 1 to `deliverable2/data/Clean_Dataset.csv`. That path is git-ignored — **do not commit the CSV**.
4. **Work through the notebook**, completing every `# TODO` cell and every markdown answer prompt. Commit per section, not in one lump.
5. **Clear outputs** before committing (Kernel → Restart & Clear Output). This is what keeps notebook merges survivable.
6. **Open a PR** from `team-##/<your-name>` into **`team-##/main`**. GitHub defaults the base to `main` — change it by hand.
7. **Review your assigned teammate's PR.** The pairing is recorded in the team report.
8. **Team merges up.** Once every member's PR has landed in `team-##/main`, the team fills in [`deliverable2/TEAM_REPORT.md`](deliverable2/TEAM_REPORT.md) together and opens **one** PR from `team-##/main` → `main`, tagging a PM.

### Acceptance criteria — per member

- [ ] Branch is `team-##/<your-name>`; the PR targets `team-##/main`, not `main`
- [ ] The notebook runs top-to-bottom with no errors (Kernel → Restart & Run All)
- [ ] All `# TODO` cells are completed
- [ ] The **EDA decisions table (2e)** is filled in — each observation paired with a modelling consequence
- [ ] The Section 3 scoreboard shows the leak-removal sequence and the honest pipeline score
- [ ] Section 4 includes the **alpha sweep plot** and a count of the features Lasso zeroed out
- [ ] Section 5 compares at least **three** objectives, plus the quantile interval and its **coverage check**
- [ ] Section 6 has the **RF vs XGBoost** head-to-head with fit times, and ARIMA scored **against the naive baseline**
- [ ] The **model-selection decision table (6c)** is filled in, with a stated choice and three non-accuracy reasons
- [ ] Every markdown answer prompt is answered; no `*(Write here)*` remains
- [ ] Outputs cleared; `git status` shows no `.csv`
- [ ] One substantive review left on the assigned teammate's PR

### Acceptance criteria — per team

- [ ] Every member's PR is merged into `team-##/main`
- [ ] `deliverable2/TEAM_REPORT.md` is filled in, including the honest number, the model the team would ship with its three reasons, the ARIMA verdict, and the process retrospective
- [ ] One PR open from `team-##/main` → `main` with a PM tagged

---

## Deliverable 3 — Tuning & Error Analysis *(preview — details at the start of Week 5)*

> **Target:** Weeks 5–6

Take the model each team chose in Deliverable 2 and make it better, deliberately: hyperparameter search (`RandomizedSearchCV` over the pipeline, so the search itself stays leakage-safe), learning curves to tell a data problem from a model problem, and segment-level error analysis — metrics broken down by cabin, airline, route and booking window, so the team can say *where* the model fails rather than just how much. Nothing here works without Deliverable 2's pipeline and the model choice behind it, which is why they are sequenced this way.

---

## Branch conventions

| Deliverable | Member branch | Merges into | Then into `main` via |
|---|---|---|---|
| 1 | `deliverable1/<name>` | — | the member's own PR into `main` |
| 2–4 | `team-##/<name>` | `team-##/main` | one PR from `team-##/main`, merged by a PM |

**The names above are a suggestion; the structure is the requirement.** What every team must
have is **one trunk branch** that collects the team's work and **one branch per member** that
merges into it. What those branches are called is up to the team — several teams already use
different shapes, and that is fine.

Three things to keep straight:

- **A team's trunk is not the repo's `main`.** Only the trunk opens a PR into the real `main`,
  and only a PM merges that PR. Members never commit to the trunk directly; work arrives there
  by merge.
- **A branch name cannot be a prefix of another.** Git stores refs as file paths, so if a plain
  branch `team-07` exists, then `team-07/main` is **impossible** — git refuses with
  `cannot lock ref ... 'team-07' exists`. Teams in that position should simply use a free name
  such as `team-07-main`. **No existing branch needs to be deleted to make room**, and no work
  is at risk either way. Same rule inside a namespace: `team-06/alex-chen-v2` is fine,
  `team-06/alex-chen/fix` is not. See [`CONTRIBUTING.md`](CONTRIBUTING.md#branch-naming).
- **Zero-padding is a nicety, not a rule.** `team-06` sorts better than `team-6` in a long
  branch list, but a team already using the unpadded form should not churn its branches over it.

---

## Working agreements

- **Each deliverable maps to one or more GitHub Issues.** The board is the source of truth; this file is the high-level narrative.
- **Dates are estimates.** When reality diverges, update the issue and, if the shift is material, this file.
- **"Done" is defined per issue** via the acceptance criteria above — not by a date passing.
- **Reprioritize openly.** If a deliverable changes, a PM notes why in the issue so the decision is auditable.
- **Teams are stable across Deliverables 2–4.** A team keeps its number and its trunk branch for the rest of the term, so the branch tree stays readable and reviewers keep their context.
