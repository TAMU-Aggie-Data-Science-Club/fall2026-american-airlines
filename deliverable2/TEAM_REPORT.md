# Deliverable 2 — Team Report

> **Team:** `team-##`
> **Filled in on:** `team-##/main`, together, after every member's PR has merged
> **This file is the team's deliverable.** It is what the PM reads when reviewing `team-##/main` → `main`.

Keep it short. The notebooks hold the work; this holds the team's conclusions. Where members disagreed, **record the disagreement** rather than smoothing it over — two defensible answers is a more useful result than a false consensus.

---

## 1. Team roster & review ring

| Member | GitHub handle | Branch | Reviewed whose PR |
|---|---|---|---|
| | | `team-##/` | |
| | | `team-##/` | |
| | | `team-##/` | |
| | | `team-##/` | |

---

## 2. What the EDA told us

From Section 2e. One line each — the observation **and** the modelling decision it drove.

| Observation | Therefore we... |
|---|---|
| Target distribution | |
| Strongest predictor | |
| The `days_left` relationship | |
| Ordered vs unordered categories | |

---

## 3. The leak ledger

From the Section 3 scoreboards. Use one member's numbers as the reference and note if anyone's differed materially.

| Configuration | R² | MAE (INR) | MAPE % |
|---|---|---|---|
| Deliverable 1 baseline (3 leaks) | | | |
| Row counter removed | | | |
| ...and deduplicated | | | |
| Leakage-safe pipeline | | | |
| 5-fold CV (mean ± std) | | | |

**The honest number we would report, and why:**

*(2–3 sentences. Name the single number and the validation scheme behind it.)*

---

## 4. Lasso vs Ridge

| | Best alpha | R² | MAE (INR) | Features kept / zeroed |
|---|---|---|---|---|
| Ridge | | | | |
| Lasso | | | | |
| ElasticNet *(optional)* | | | | |

**Features Lasso discarded, and whether that matched the EDA:**

**Did regularisation actually matter on this dataset?** *(With ~300k rows and ~30 one-hot columns, the honest answer may be "barely". If so, say so and explain why — that is a real finding about the data, not a failure.)*

---

## 5. Loss functions

| Objective trained on | MAE | RMSE | R² | MAPE % |
|---|---|---|---|---|
| Squared error (MSE) | | | | |
| Absolute error (MAE) | | | | |
| Huber | | | | |

**The pattern we observed:** *(one sentence — which loss won on which metric, and why that is not a coincidence)*

**Quantile models:** 80% interval coverage was ____% (target ~80%).

**Which loss we would train on, and which metric we would report:**

**Is it worse to quote a fare too high or too low?** *(Pick a side and name the `α` you would ship. The symmetric default is a choice too — just an unexamined one.)*

---

## 6. Model selection

| Model | MAE (INR) | R² | Fit time (s) | Interpretable? |
|---|---|---|---|---|
| Dummy (mean) | | | | — |
| Ridge | | | | yes |
| Lasso | | | | yes |
| Random Forest | | | | no |
| XGBoost | | | | no |

**Bagging vs boosting — what we observed in practice:** *(Did RF or XGBoost win, by how much, and at what cost in fit time?)*

### ARIMA

| | MAE on held-out tail |
|---|---|
| ARIMA(p,d,q) = ( , , ) | |
| Naive last-value baseline | |

**Did ARIMA beat naive?** *(If not, say so plainly. A negative result reported honestly is worth more than a flattering one.)*

**What question did ARIMA actually answer, and what did we give up to ask it?** *(Be precise: it forecasts an aggregate curve. It cannot price a ticket, and it ignores airline, class, route and stops entirely.)*

**Would we include it in the deliverable to American Airlines, and in what role?**

---

## 7. The model we would ship

**Our choice:**

**Three reasons that are not "best score":**

1.
2.
3.

**The strongest argument against our choice:**

---

## 8. For American Airlines

**What this model could support today:** *(one concrete decision, and the error tolerance it needs)*

**What we would need before anyone should act on it:** *(the top one or two gaps from Section 7b)*

**The limitation we would state first, unprompted, in a client meeting:**

---

## 9. Process retrospective

Three honest lines. These feed into how the PMs run Deliverable 3, so vague answers waste your own time.

- **What worked in the team branch workflow:**
- **What did not:**
- **One thing we want changed for Deliverable 3:**

---

## 10. Sign-off

- [ ] Every member's PR has merged into `team-##/main`
- [ ] Every member's notebook runs top-to-bottom with cleared outputs
- [ ] No `.csv` anywhere in the diff
- [ ] This report is filled in — no empty cells, no placeholder `##`
- [ ] One PR open from `team-##/main` → `main`, with a PM tagged
