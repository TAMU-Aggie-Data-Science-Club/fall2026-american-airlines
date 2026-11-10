# Deliverables & Timeline

## Milestones at a glance

| # | Deliverable | Description | Owner | Branches | Target |
|---|-------------|-------------|-------|----------|--------|
| 1 | GitHub Fundamentals + ML Starter | Git workflow practice + intro XGBoost notebook on flight price prediction | All members, individually | `deliverable1/<name>` → `main` | ✅ Week 1 (Oct 6–12, 2026) |
| 2 | EDA, Data Leakage & Choosing the Right Model | Deeper EDA tied to modelling decisions; data leakage and what it cost; Lasso vs Ridge; loss functions (MSE / MAE / Huber / quantile); XGBoost vs Random Forest vs ARIMA. *Absorbs the former "EDA & Data Cleaning" and "Model Development & Evaluation" milestones — members cannot evaluate a model honestly until they can spot leakage, so the two were being taught in the wrong order.* | All members, in teams | `team-##/<name>` → `team-##/main` → `main` | ✅ Week 3–4 (due Nov 2, 2026) |
| 3 | Real Data: Ingestion & EDA | Move off the practice Kaggle dataset onto the real, NDA-covered American Airlines dataset. Pull it from Google Drive (link via Discord/team lead), set up a `team-##-modeling/` folder on the team branch, and run EDA that ends in a ready-to-go feature set for a specific model or model subset — no mock notebook. *New milestone — pushes the former Deliverable 3 ("Tuning & Error Analysis") to 4 and "Final Findings" to 5, since teams need real features before tuning a real model.* | Teams | `team-##/<name>` → `team-##/main` → `main` | Week 5 (due Nov 9, 2026) |
| 4 | Tuning & Error Analysis | Hyperparameter search over the chosen model, learning curves, regularisation, segment-level error analysis | Teams | `team-##/<name>` → `team-##/main` → `main` | Weeks 6–7 |
| 5 | Final Findings & Presentation | Report / dashboard / model artifact for American Airlines | Teams + PM | `team-##/<name>` → `team-##/main` → `main` | Week 8 |

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

## Deliverable 3 — Real Data: Ingestion & EDA

> **Due:** Monday, November 9, 2026
> **Owner:** every member, working inside their assigned team
> **Branch convention:** same tree as Deliverable 2 — `team-##/<your-name>` → PR into `team-##/main` → PM merges into `main`

### What this is

Deliverables 1 and 2 ran on the public Kaggle flight-price dataset — useful for learning the
workflow, but it was never the project's real data. This deliverable is the handoff: teams
pull down the **actual American Airlines dataset**, which is covered by an NDA, get it set up
correctly (git-ignored, never shared outside the team), and run EDA on it for real.

The deliverable is **not** "produce a notebook that runs." It is: **by the end of the week, have
a complete, ready-to-go set of features that the team knows works for a specific model or model
family** — chosen using the same reasoning Deliverable 2 taught (data shape, loss, stakeholder
need, cost, and only then score). Every feature kept should have a reason; every feature dropped
should have a reason too.

**No mock or template notebook for this one.** Deliverable 1 and 2 shipped with a shared starter
notebook; this deliverable does not, by design — the real dataset's quirks (whatever they turn
out to be) are the point, and a template would paper over exactly the EDA work this deliverable
exists to teach. Teams build their EDA from a blank notebook under their own
`team-##-modeling/` folder (see below).

### Why it is shaped this way

Deliverable 2 taught leakage and model selection on a clean, well-understood public dataset
specifically so teams would hit those lessons before the real data — which is messier, has its
own leakage traps, and has a client (American Airlines) who will actually read the conclusions.
Doing ingestion + EDA as its own deliverable, instead of folding it into tuning, means teams stop
and look at the real data honestly before any model is fit to it.

### 1. Get the data

The American Airlines dataset is under **NDA**. Treat it accordingly:

- It lives in a **Google Drive folder** shared with the project. It is **not linked from this
  repo** on purpose — ask your **team lead** for the link, or check the project's **Discord**
  (the link/announcement was posted there). If neither gets you access, ask a PM before doing
  anything else.
- Download it locally into the **git-ignored** [`data/`](data/) folder — see [`DATA.md`](DATA.md)
  for the full entry. **Never** commit the raw file, a cleaned copy, a sample of rows, or a
  screenshot of it into this repo, an issue, a PR, or a chat message outside the team.
- Do not re-upload it anywhere else (your own Drive, a personal repo, a DM to someone outside
  the team). If you're unsure whether sharing something crosses that line, ask a PM before doing
  it, not after.
- `git status` should never show the data file as untracked-and-about-to-be-added or staged.

### 2. Set up your team's modeling folder

Each team works inside its own `team-##-modeling/` folder, committed to the **team's branch
tree** (your member branches → your team's trunk), not to a new top-level folder shared across
teams. This is where all of your Deliverable 3 work lives, so every team's models and findings
merge cleanly into `main` side by side.

```
team-##-modeling/
├── notebooks/        # or scripts/ — the Jupyter notebooks you're building to explore and train
├── models/           # saved model files and your prediction outputs
└── results/          # or visuals/ — plots, figures, summary tables
```

**Organization inside the folder is your team's call** — the structure above is a suggestion,
not a requirement. What matters is that your team agrees on a layout before everyone starts
committing. Decide it in your first team meeting for this deliverable.

```sh
git switch team-06/<your-name>          # your existing member branch, or cut a new one
mkdir -p team-06-modeling/notebooks team-06-modeling/models team-06-modeling/results
```

Work through the usual flow: commit on your member branch, open a PR into `team-##/main`, get it
reviewed by a teammate, merge up. Once everyone's EDA has landed in `team-##/main`, the team
opens one PR from `team-##/main` → `main`, tagging a PM — same as Deliverable 2.

### 3. Do the EDA

There's no notebook handed to you this time, but "basic EDA" on a new dataset should at minimum
cover:

- **Shape and types.** Rows, columns, what each column actually represents (ask a PM/team lead
  if a column name is ambiguous — don't guess at a real dataset's schema).
- **Missingness.** Which columns have gaps, how much, and whether the gap is random or tied to
  something else in the data (e.g. missing only for a particular route or date range).
- **Distributions and outliers.** For the target and for the features you expect to matter —
  same instinct as Deliverable 2's price-by-cabin split, applied to whatever this dataset's
  target turns out to be.
- **Leakage risk, specific to this dataset.** Deliverable 2 taught you what leakage looks like
  on the Kaggle data; this dataset will have its own version of that problem, likely not the
  same columns. Look for anything that could only be known *after* the outcome you're trying to
  predict.
- **Relationships that bear on model choice.** Linear vs. non-linear relationships, categorical
  cardinality, whether there's a genuine time axis this time (unlike the Kaggle dataset) — each
  of these should point toward or away from a specific model family.

### 4. Land on a feature set

Finish the week with:

- A **named model or model family** (e.g. "gradient-boosted trees" or "Ridge regression"), with
  the EDA observations that led there — not "we'll decide after we see the scores."
- A **feature set** — the actual transformed columns, produced by code in your notebook/scripts,
  not just a bullet list — that you're confident is appropriate input for that model/family.
- A short **written rationale** per feature you engineered or dropped. This is the part that
  makes the feature set "ready-to-go" instead of just "a dataframe we made."

This feature set is what Deliverable 4 (tuning) builds on, so it needs to actually run, not just
exist in someone's head.

### Acceptance criteria — per member

- [ ] Your branch is `team-##/<your-name>`; your PR targets `team-##/main`, not `main`
- [ ] Your notebook/scripts run top-to-bottom with no errors
- [ ] No data file (raw, cleaned, or sampled) is committed anywhere in your PR
- [ ] Your EDA addresses shape/types, missingness, distributions/outliers, and leakage risk for
      the real dataset — not a repeat of the Kaggle findings from Deliverable 2
- [ ] You can state, in writing, which model/model family your EDA points to and why
- [ ] Outputs are cleared before committing (Kernel → Restart & Clear Output)
- [ ] One substantive review left on your assigned teammate's PR

### Acceptance criteria — per team

- [ ] `team-##-modeling/` exists on the team's trunk with every member's EDA merged in
- [ ] The AA dataset is in a git-ignored `data/` location only — `git status` / `git log` show no committed data files
- [ ] The team has agreed on and documented one feature set, with the per-feature rationale,
      aimed at one named model or model family
- [ ] `DATA.md` row for the AA dataset is accurate for how the team actually accessed it
- [ ] One PR open from `team-##/main` → `main`, tagging a PM

> **A reminder on the NDA.** This is the first deliverable working with data that is not yours
> to post, publish, or discuss outside the team and the club's PMs. If you're ever unsure
> whether something is safe to share — a chart with real numbers on it, a snippet of raw rows
> pasted into a Discord message to ask for help, a screenshot in a PR description — **don't**,
> and ask a PM first. It's much cheaper to ask than to un-share something.

---

## Deliverable 4 — Tuning & Error Analysis *(preview — details at the start of Week 6)*

> **Target:** Weeks 6–7

Take the model each team chose and the feature set each team built in Deliverable 3, on the
real American Airlines data, and make it better, deliberately: hyperparameter search
(`RandomizedSearchCV` over the pipeline, so the search itself stays leakage-safe), learning
curves to tell a data problem from a model problem, and segment-level error analysis — metrics
broken down by whatever segments the real data supports, so the team can say *where* the model
fails rather than just how much. Nothing here works without Deliverable 3's feature set and the
model choice behind it, which is why they are sequenced this way.

---

## Deliverable 5 — Final Findings & Presentation *(preview)*

> **Target:** Week 8

Report, dashboard, or model artifact delivered to American Airlines, built on the tuned model
from Deliverable 4.

---

## Branch conventions

| Deliverable | Member branch | Merges into | Then into `main` via |
|---|---|---|---|
| 1 | `deliverable1/<name>` | — | the member's own PR into `main` |
| 2–5 | `team-##/<name>` | `team-##/main` | one PR from `team-##/main`, merged by a PM |

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
- **Teams are stable across Deliverables 2–5.** A team keeps its number and its trunk branch for the rest of the term, so the branch tree stays readable and reviewers keep their context.
