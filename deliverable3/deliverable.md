# Deliverable 3 — Real Data: Ingestion & EDA

> **Due:** Monday, November 9, 2026
> **Owner:** every member, working inside their assigned team
> **Branches:** you work on your own member branch → PR into your team's trunk branch → a PM merges the trunk into `main`

---

## What this is

Deliverables 1 and 2 ran on the public Kaggle flight-price dataset. That was practice — a clean,
well-documented stand-in so everyone could learn the git workflow, EDA, leakage, and model
selection without the extra friction of a real, messy dataset under NDA.

This deliverable is the handoff. Your team now gets the **actual American Airlines dataset**.
From here on, this is the dataset the project ships findings about.

**The deliverable is getting the data and doing basic EDA.** That's it — this week is not about
training a tuned model. But "basic EDA" has a specific, checkable end state:

> By the end of this week, your team has a **complete, ready-to-go set of features** that you
> know works for a specific model, or a specific small family of models.

That means: you've looked at the real data closely enough to know what you're working with, you
picked a model (or model family) based on that — using the same reasoning Deliverable 2 taught
(data shape first, then loss, then stakeholder need, then cost, and *only then* score) — and you
engineered a feature set aimed at that model, with the reasoning for every kept and dropped
feature written down.

**Do not write a mock or template notebook.** Deliverables 1 and 2 gave everyone a shared
starter notebook to fill in. This deliverable deliberately does not. A template notebook, by
definition, pre-decides what the EDA looks for — and finding out what the real dataset actually
contains, what's missing, what's leaking, and what's usable is the entire point of this week.
If your notebook could have been handed to you with blanks already in it, you have not done the
deliverable.

---

## 1. Get the data

The American Airlines dataset is under **NDA**. Treat it accordingly:

- The dataset lives in a **Google Drive folder** shared with the project. It is **not linked
  from this repo** on purpose — ask your **team lead** for the link, or check the project's
  **Discord** (the link/announcement was posted there). If neither gets you access, ask a PM
  before doing anything else.
- Download it locally into the **git-ignored** `data/` folder (or `deliverable3/data/` — both
  are git-ignored; see [`DATA.md`](../DATA.md) for the full entry). **Never** commit the raw
  file, a cleaned copy, a sample of rows, or a screenshot of it into this repo, an issue, a PR,
  or a chat message outside the team.
- Do not re-upload it anywhere else (your own Drive, a personal repo, a Slack/Discord DM to
  someone outside the team). If you're unsure whether sharing something crosses that line, ask a
  PM before doing it, not after.
- `git status` should never show the data file as untracked-and-about-to-be-added or staged. If
  it does, check your `.gitignore` entry before you touch `git add`.

## 2. Set up your team's modeling folder

Each team works inside its own `team-##-modeling/` folder, committed to the **team's branch
tree** (your member branches → your team's trunk), not to a new top-level folder shared across
teams. This is where all of your Deliverable 3 work lives, so that every team's models and
findings merge cleanly into `main` side by side.

```
team-##-modeling/
├── notebooks/        # or scripts/ — the Jupyter notebooks you're building to explore and train
├── models/           # saved model files and your prediction outputs
└── results/          # or visuals/ — plots, figures, summary tables
```

**Organization inside the folder is your team's call** — the structure above is a suggestion,
not a requirement. What matters is that your team agrees on a layout before everyone starts
committing, so you don't end up with five different conventions in one folder. Decide it in your
first team meeting for this deliverable and record the decision somewhere your PM can see (e.g.
a short note at the top of your team's `team-##-modeling/README.md`, if you want one).

```sh
git switch team-06/<your-name>          # your existing member branch, or cut a new one
mkdir -p team-06-modeling/notebooks team-06-modeling/models team-06-modeling/results
```

Work through the usual flow: commit on your member branch, open a PR into `team-##/main`, get it
reviewed by a teammate, merge up. Once everyone's EDA has landed in `team-##/main`, the team
opens one PR from `team-##/main` → `main`, tagging a PM — same as Deliverable 2.

## 3. Do the EDA

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

## 4. Land on a feature set

Finish the week with:

- A **named model or model family** (e.g. "gradient-boosted trees" or "Ridge regression"), with
  the EDA observations that led there — not "we'll decide after we see the scores."
- A **feature set** — the actual transformed columns, produced by code in your notebook/scripts,
  not just a bullet list — that you're confident is appropriate input for that model/family.
- A short **written rationale** per feature you engineered or dropped. This is the part that
  makes the feature set "ready-to-go" instead of just "a dataframe we made."

This feature set is what Deliverable 4 (tuning) builds on, so it needs to actually run, not just
exist in someone's head.

---

## Acceptance criteria — per member

- [ ] Your branch is `team-##/<your-name>`; your PR targets `team-##/main`, not `main`
- [ ] Your notebook/scripts run top-to-bottom with no errors
- [ ] No data file (raw, cleaned, or sampled) is committed anywhere in your PR
- [ ] Your EDA addresses shape/types, missingness, distributions/outliers, and leakage risk for
      the real dataset — not a repeat of the Kaggle findings from Deliverable 2
- [ ] You can state, in writing, which model/model family your EDA points to and why
- [ ] Outputs are cleared before committing (Kernel → Restart & Clear Output)
- [ ] One substantive review left on your assigned teammate's PR

## Acceptance criteria — per team

- [ ] `team-##-modeling/` exists on the team's trunk with every member's EDA merged in
- [ ] The team has agreed on and documented one feature set, with the per-feature rationale,
      aimed at one named model or model family
- [ ] `DATA.md` in the root repo reflects how the team actually accessed the dataset (update the
      row if your access path differed from the default)
- [ ] One PR open from `team-##/main` → `main`, tagging a PM

---

## A reminder on the NDA

This is the first deliverable working with data that is not yours to post, publish, or discuss
outside the team and the club's PMs. If you're ever unsure whether something is safe to share —
a chart with real numbers on it, a snippet of raw rows pasted into a Discord message to ask for
help, a screenshot in a PR description — **don't**, and ask a PM first. It's much cheaper to ask
than to un-share something.
