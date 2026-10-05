# Deliverables & Timeline

## Milestones at a glance

| # | Deliverable | Description | Owner | Target |
|---|-------------|-------------|-------|--------|
| 1 | GitHub Fundamentals + ML Starter | Git workflow practice + intro XGBoost notebook on flight price prediction | All members | Week 1 (Oct 6–12, 2026) |
| 2 | EDA & Data Cleaning | Deep-dive exploratory analysis, surface quality issues, clean and prep the dataset | All members | Week 2 (Oct 13–19, 2026) |
| 3 | Feature Engineering | Build a reproducible preprocessing pipeline; engineer new features | Members | Weeks 3–4 |
| 4 | Model Development & Evaluation | Baseline → improved model; agree on evaluation metrics | Members | Weeks 4–5 |
| 5 | Iteration & Tuning | Hyperparameter search, ablation studies, error analysis | Members + PM | Weeks 5–7 |
| 6 | Final Findings & Presentation | Report / dashboard / model artifact for American Airlines | Members + PM | Week 8 |

---

## Deliverable 1 — GitHub Fundamentals & ML Starter

> **Due:** October 12, 2026  
> **Owner:** Every team member individually  
> **Branch convention:** `deliverable1/<your-name>` (e.g. `deliverable1/alex-chen`)

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

5. **Open and work through** `deliverable1/flight_price_prediction.ipynb`. Fill in every `# TODO` cell and answer the reflection questions at the end.

6. **Commit your work** with a clear message:
   ```sh
   git add deliverable1/flight_price_prediction.ipynb
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

## Deliverable 2 — EDA & Data Cleaning *(preview — details coming Week 2)*

> **Target:** October 19, 2026

Building on Deliverable 1, this deliverable goes deeper: thorough exploratory data analysis, handling missing values and outliers at scale, and building a reproducible cleaning pipeline we can run reliably as we iterate. More details will be posted as a GitHub Issue at the start of Week 2.

---

## Working agreements

- **Each deliverable maps to one or more GitHub Issues.** The board is the source of truth; this file is the high-level narrative.
- **Dates are estimates.** When reality diverges, update the issue and, if the shift is material, this file.
- **"Done" is defined per issue** via the acceptance criteria above — not by a date passing.
- **Reprioritize openly.** If a deliverable changes, a PM notes why in the issue so the decision is auditable.
