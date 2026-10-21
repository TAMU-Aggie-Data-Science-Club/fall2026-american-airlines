# Deliverable 2 — EDA, Data Leakage & Choosing the Right Model

> **Due:** Monday, November 2, 2026
> **Owner:** every member, working inside an assigned team
> **Branches:** you work on your own member branch → PR into your team's trunk branch → a PM merges the trunk into `main`
> **Notebook:** [`flight_price_modeling.ipynb`](flight_price_modeling.ipynb)

---

## What this is

Deliverable 1 handed you one model — `XGBRegressor` — and you trained it. Two questions were never asked: **was that 0.98 R² real**, and **was XGBoost the right choice?**

This deliverable answers both. Deeper EDA, data leakage, and then a real model-selection exercise: **Lasso vs Ridge**, **Random Forest vs XGBoost**, **ARIMA**, and the choice sitting underneath all of them that nobody teaches you to make deliberately — your **loss function**.

The thesis: **model selection is not a leaderboard.** It is a set of choices — model family, loss function, regularisation — each of which should follow from something you saw in the data or something the business told you. By the end, members should be able to justify every choice without saying "it scored the best".

### Learning goals

By the end you should be able to:

- run EDA that **changes a modelling decision**, not EDA that just describes the data
- define data leakage, name three forms of it, and spot them in code you did not write
- explain why `fit` belongs inside the train/test split, and use `Pipeline` + `ColumnTransformer` to guarantee it
- state the difference between the **L1 and L2 penalties**, explain why Lasso produces exact zeros and Ridge does not, and tune `alpha` by validation rather than by taste
- distinguish a **loss function** from an **evaluation metric**, and choose between MSE, MAE, Huber and quantile loss on purpose
- explain why **ReLU is an activation function, not a loss** — and where `max(0, ·)` genuinely does appear inside a loss
- explain **bagging vs boosting**, and predict the failure mode of each
- say what **ARIMA** needs from a dataset, and recognise when a model family does not fit the problem at all
- read `mean ± std` from cross-validation instead of trusting one split

---

## Section map

| Section | Topic | The question it answers |
|---|---|---|
| 1 | Load | — |
| **2** | **Deeper EDA** | What does the data look like, and what does that imply about which model to reach for? |
| **3** | **Data leakage** | Why was the Deliverable 1 score too good, and how do I stop it recurring? |
| **4** | **Lasso vs Ridge** | Two penalties. When do I want shrinkage, and when do I want feature selection? |
| **5** | **Loss functions** | MSE, MAE, Huber, quantile — what am I actually asking the model to minimise? |
| **6** | **XGBoost vs RF vs ARIMA** | Three families. Which fits *this* problem, and why does one answer a different question? |
| **7** | **Defend your choice** | Given all of it, what would you ship to American Airlines? |

> **A note on ARIMA, for PMs and the curious.** This dataset has **no dates** — no departure date, no booking date, no timestamp. It is a cross-section, so a time-series model has almost nothing to index. We include ARIMA anyway, applied to the one real sequence in the data: the **booking curve** (mean price by `days_left`, ~49 ordered points). The section is deliberately framed around the honest conclusion — ARIMA forecasts an aggregate curve and **cannot price an individual ticket**, and recognising that a model family does not match your data shape is a more valuable lesson than another tuned score. Members are asked to beat a naive last-value baseline and to report honestly if they cannot.

---

## The git workflow: team branches

**This changed from Deliverable 1.** Last time everyone PR'd into `main` directly. With 70+ members that produces an unreviewable queue. Now each team owns a branch tree, and only the team's trunk touches `main`.

```
main                          <- protected. Only a PM merges here.
 └── team-06/main             <- your team's trunk branch
      ├── team-06/alex-chen
      ├── team-06/priya-r
      └── team-06/sam-lee
```

**The structure is the requirement; the names are a suggestion.** Your team needs one trunk
branch and one branch per member that merges into it. What you call them is up to your team —
several teams already use different shapes, and that is fine.

| | |
|---|---|
| **Your team trunk** | suggested: `team-##/main` — zero-padded, so team 6 is `team-06` |
| **Your personal branch** | suggested: `team-##/<your-name>` — e.g. `team-06/alex-chen` |
| **Your PR target** | your team's trunk — **not** `main` |
| **Who merges to `main`** | a PM, reviewing the trunk once every member has merged in |

### 1. One person per team creates the trunk (once)

Coordinate so this happens exactly once. If it already exists, skip to step 2.

```sh
git switch main
git pull origin main
git switch -c team-06/main          # your team number
git push -u origin team-06/main
```

**If that is refused with `cannot lock ref ... exists`**, a plain branch named `team-##` already
occupies the name. Not a problem, and nothing needs deleting — just use a free name:

```sh
git switch -c team-06-main          # a dash instead of a slash always works
git push -u origin team-06-main
```

Then tell your team in your Discord thread that the trunk exists.

### 2. Everyone else branches off the trunk

Note the branch point: off your team's trunk, **not** off `main`.

```sh
git fetch origin
git switch -c team-06/alex-chen origin/team-06/main
git push -u origin team-06/alex-chen
```

### 3. Get the data and packages in place

```sh
cp deliverable1/data/Clean_Dataset.csv deliverable2/data/Clean_Dataset.csv
pip install pandas numpy matplotlib seaborn scikit-learn xgboost statsmodels
```

`statsmodels` is the one new package — it provides ARIMA. `deliverable2/data/` is git-ignored; **never commit the CSV.** If you no longer have it, the Kaggle link is in the notebook's Resources table.

### 4. Work, commit, push

```sh
git add deliverable2/flight_price_modeling.ipynb
git commit -m "deliverable2: EDA, leakage fixes, model selection"
git push
```

Commit as you finish each section rather than in one lump — small commits are the whole reason the history is useful to a reviewer.

### 5. Open a PR into your team trunk

On GitHub, open a PR from your member branch → your team's trunk. **Check the base branch.** GitHub defaults it to `main`, and you have to change it by hand every time.

### 6. Review a teammate

Each member reviews one other member's PR. Your team decides the pairing in your first meeting and records it in [`TEAM_REPORT.md`](TEAM_REPORT.md) — a simple ring works (A reviews B, B reviews C, C reviews A).

Reviewing is not a formality and it is not a style check. Ask about the *decisions*:

- "You trained on MAE instead of MSE — what in the data convinced you?"
- "Your Lasso zeroed out four airline dummies. Does that match your EDA?"
- "Your ARIMA lost to the naive baseline. Why keep the section in?"

### 7. Team merges up

Once every member's PR is merged into `team-##/main`, the team fills in [`TEAM_REPORT.md`](TEAM_REPORT.md) together, commits it to `team-##/main`, and opens **one** PR from `team-##/main` → `main`. Tag a PM. That PR is the team's deliverable.

---

## Surviving notebook merge conflicts

A `.ipynb` is JSON. Git conflicts inside it are genuinely unpleasant, and most are avoidable.

**Prevention, in order of effectiveness:**

1. **Clear outputs before every commit.** Kernel → Restart & Clear Output. The large majority of notebook conflicts are rendered output and base64 image blobs, not code.
2. **Do not split this notebook across teammates.** Each member completes their own full copy on their own branch. The notebook is individual learning; `TEAM_REPORT.md` is the shared artifact. Four people on four branches editing four copies never conflict with each other — the only merge is into the trunk.
3. **Pull the trunk before you push.** `git fetch origin && git merge origin/team-06/main`.

**If you do hit a conflict in a notebook:**

```sh
# Option A -- your branch's version is the one you want (usually true here,
# since each member owns their own copy):
git checkout --ours deliverable2/flight_price_modeling.ipynb
git add deliverable2/flight_price_modeling.ipynb

# Option B -- back out and ask for help. A completely fine choice.
git merge --abort
```

Then post in your team thread. Do **not** hand-edit notebook JSON to resolve a conflict; you will corrupt the file, and `git merge --abort` costs nothing.

> `nbdime` (`pip install nbdime`, then `nbdime config-git --enable --global`) gives git a notebook-aware diff and merge driver. Worth installing if you expect to work in notebooks past this semester.

---

## Acceptance criteria

Your PR into `team-##/main` is done when:

- [ ] Your branch is `team-##/<your-name>` and the PR targets `team-##/main`
- [ ] The notebook runs top-to-bottom with no errors (**Kernel → Restart & Run All**)
- [ ] Every `# TODO` cell is completed
- [ ] The **EDA decisions table (2e)** is filled in — each observation paired with a modelling consequence
- [ ] The leak sequence in Section 3 is on the scoreboard: D1 baseline, each fix, and the honest pipeline score
- [ ] Section 4 includes the **alpha sweep plot** and a count of how many features Lasso zeroed
- [ ] Section 5 compares at least **three** objectives, plus the quantile interval with its **coverage check**
- [ ] Section 6 has the **RF vs XGBoost** head-to-head with fit times, and ARIMA scored **against the naive baseline**
- [ ] The **model-selection decision table (6c)** is filled in, with a stated choice and three non-accuracy reasons
- [ ] Every markdown answer prompt is answered. No `*(Write here)*` left anywhere
- [ ] Outputs cleared and `git status` shows no `.csv`
- [ ] You have reviewed your assigned teammate's PR with at least one substantive question

---

## If you get stuck

| Symptom | Likely cause |
|---|---|
| `FileNotFoundError` on the CSV | `DATA_PATH` is relative to where you launched Jupyter. From the repo root it is `deliverable2/data/Clean_Dataset.csv`; from inside `deliverable2/` it is `data/Clean_Dataset.csv`. |
| `ModuleNotFoundError: statsmodels` | `pip install statsmodels`, then **restart the kernel**. Installing without restarting is the classic version of this. |
| `OrdinalEncoder`: "Found unknown categories" | Your hard-coded `categories=` list does not match the data. Print `sorted(clean[col].unique())` and copy the real spellings. |
| `XGBoostError` on `reg:absoluteerror` | Needs xgboost ≥ 1.7. Check `xgboost.__version__`; the notebook gives a `GradientBoostingRegressor` fallback. Note the substitution in your PR. |
| Lasso `ConvergenceWarning` | It is iterative and stopped early. Raise `max_iter`. The warning is information, not noise. |
| Random Forest takes forever / eats RAM | Expected: an unbounded forest on ~240k rows grows huge trees. Set `max_depth=20` or subsample, and **report it** — "RF needed 10× the fit time" is a legitimate finding. |
| ARIMA convergence warnings | Expected. 40 training points is a very small sample. Say so in your write-up rather than hiding it. |
| Cross-validation takes forever | 5 folds × 300 trees × 300k rows. Subsample to 80,000 for the CV cells and note it. Keep `n_jobs=1` in `cross_validate` — XGBoost already parallelises, and nesting them fights over cores. |
| `cannot lock ref ... exists` | A branch name cannot be a *prefix* of another, because git stores refs as file paths. If a plain `team-07` branch exists then `team-07/main` is impossible — use `team-07-main` instead. Nothing needs deleting. |
| `refname conflict` during `git fetch` | Harmless, and local to your clone: a branch was renamed on GitHub and your clone still holds the old tracking ref. It does not affect pulling `main`, and it does not affect anyone else. `git remote prune origin` clears it — that only removes your clone's bookkeeping, never a real branch. |
| Notebook won't open after a merge | Conflict markers in the JSON. `git merge --abort`, then see above. |

Ask in the project Discord thread before you lose an evening. Describe what you tried — a question with a traceback and an attempt in it gets a much better answer than "it's broken".

---

## References

Every link was checked against the live page and points at the specific section you need. The same list, organised by notebook section, is in the notebook itself — keep the ones for your current section open in browser tabs.

**Start here**

- [scikit-learn: Common pitfalls and recommended practices](https://scikit-learn.org/stable/common_pitfalls.html) — the most useful single page for this deliverable. Shows a "Wrong" and a "Right" version of each mistake.
- [An Introduction to Statistical Learning](https://www.statlearning.com/) — the standard undergraduate text, **free PDF**. Ch. 6 covers Ridge/Lasso, Ch. 8 covers trees and boosting. A Python edition (ISLP) is also available.
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

- [XGBoost learning-task parameters](https://xgboost.readthedocs.io/en/stable/parameter.html#learning-task-parameters) — the full `objective` list. Version floors: `reg:absoluteerror` was added in **1.7.0**, `reg:quantileerror` in **2.0.0**.
- [Prediction intervals for gradient boosting regression](https://scikit-learn.org/stable/auto_examples/ensemble/plot_gradient_boosting_quantile.html) — the reference implementation for §5's quantile models and coverage check
- [Regression metrics](https://scikit-learn.org/stable/modules/model_evaluation.html#regression-metrics) — MAE, RMSE, R², `mean_pinball_loss`

**Model families (§6)**

- [XGBoost: Introduction to Boosted Trees](https://xgboost.readthedocs.io/en/stable/tutorials/model.html) — the clearest explanation of additive training available. Read it before writing your bagging-vs-boosting answer.
- [Random forests](https://scikit-learn.org/stable/modules/ensemble.html#random-forests) · [Gradient-boosted trees](https://scikit-learn.org/stable/modules/ensemble.html#gradient-boosted-trees)
- [statsmodels ARIMA API](https://www.statsmodels.org/stable/generated/statsmodels.tsa.arima.model.ARIMA.html) — the `order=(p, d, q)` signature
- [Forecasting: Principles and Practice, Ch. 9 — ARIMA](https://otexts.com/fpp3/arima.html) · [§9.1 Stationarity and differencing](https://otexts.com/fpp3/arima.html#sec-stationarity) — read before choosing your `d`

---

## Files

| File | What it is |
|---|---|
| [`flight_price_modeling.ipynb`](flight_price_modeling.ipynb) | The deliverable. Each member completes their own copy. Opens with a **"How to approach this deliverable"** section — read it before you start. |
| [`TEAM_REPORT.md`](TEAM_REPORT.md) | Filled in by the team together on `team-##/main`. |
| `data/` | Git-ignored. Put `Clean_Dataset.csv` here. |

See [`../DELIVERABLES.md`](../DELIVERABLES.md) for where this sits in the timeline, and [`../CONTRIBUTING.md`](../CONTRIBUTING.md) for the general workflow.
