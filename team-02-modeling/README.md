# Team 02 — Deliverable 3: Real Data, Ingestion & EDA

**Team lead:** Akshith Reddy Sheri · **Trunk branch:** `team-02/main` · **Due:** Monday, November 9, 2026

**Team deadline:** tonight — at the very latest **before our team meeting tomorrow (Thursday, Nov 12)**. Your PR into `team-02/main` must be open by then.

Goal (from `DELIVERABLES.md`): look at the real American Airlines data honestly, find its leakage traps, and finish
with **one feature set built by code**, aimed at **one named model family**, with a reason for every feature kept or dropped.

> **There is no template notebook — on purpose.** The brief says each of us builds our EDA from a **blank notebook**,
> because the real data's quirks are the point. This README tells you *what* your notebook must cover, not how.

---

## 🔒 NDA rules — read first

- The AA file goes **only** in the repo's `data/raw/` folder on your own laptop. That folder is git-ignored.
- **Never** commit, upload, screenshot, or paste the data, rows from it, or charts/tables with real numbers —
  not in a PR, not in Discord, not into an AI tool.
- Always **Kernel → Restart & Clear Output** before committing a notebook.
- `team-02-modeling/.gitignore` blocks common data files (`.csv`, `.parquet`, `.xlsx`, `.pkl` …) as a safety net.
  If `git status` ever shows a data file, **stop** and ask the lead.

---

## Who does what

**Everyone** builds their own EDA notebook covering all five required areas below (each of us is graded on that).
**Then** each person goes deeper on one assigned area and writes a short hand-off for the team feature set.

| Member | Branch | Deep-dive area | Hand-off to the lead |
|---|---|---|---|
| **Akshith Reddy Sheri** (lead) | `team-02/akshithreddy-sheri` | Team setup, reviews and merges, `FEATURE_SET.md`, `DATA.md`, final PR | Assembles everyone's hand-offs |
| **Sahith Yamani** | `team-02/sahith-yamani` | **Leakage** — decide every suspect column: leak, ID, or keep | Leak / ID column list, with a reason each |
| **Aidan Chance** | `team-02-aidan-chance` | **Target & outliers** — what the target is, its shape, outlier policy, which loss | Target definition, outlier handling, recommended loss |
| **Chau Ha** | `team-02/chau-ha-d3` *(new branch)* | **Missingness** — random, or tied to a route/date/category? How to handle each column | Per-column handling plan |
| **Kimberly Maxwell** | `team-02/kimberly-maxwell` | **Time axis & date features** — is there a real time axis? validation scheme, date features | Validation scheme + date features to build |
| **Aditi Walvekar** | `team-02/aditi_walvekar` | **Categoricals & relationships** — cardinality, encoding, linear vs non-linear, interactions. **Also writes the team's final feature-set code** (`notebooks/feature_set.ipynb`) from everyone's hand-offs | Encoding plan + the feature-set notebook |

**Review ring** (each person leaves one substantive review on this person's PR):
Akshith → Sahith → Aidan → Chau → Kimberly → Aditi → Akshith

---

## What your notebook must cover (from the brief)

Build this yourself in a **new, blank** notebook. Write a markdown conclusion under each part — the sentence
*"therefore we should ___"* is the deliverable, not the plot.

1. **Shape and types** — rows, columns, what each column actually represents. If a column name is ambiguous,
   **ask the lead / a PM — don't guess.**
2. **Missingness** — which columns have gaps, how much, and whether the gap is random or tied to something
   (a particular route, date range, category).
3. **Distributions and outliers** — for the target and for the features you expect to matter.
4. **Leakage risk, specific to this dataset** — anything that could only be known *after* the outcome we predict.
   Ask of every column: *"would I have this value at the moment I need a prediction?"*
5. **Relationships that bear on model choice** — linear vs non-linear, categorical cardinality, whether there is
   a genuine time axis — each should point toward or away from a model family.
6. **Your deep-dive** (your row in the table above), ending with your **hand-off**:

   | Column / feature | Decision (keep / drop / engineer / leak / id) | Reason |
   |---|---|---|

7. **Your model-family view** — in writing: which model or family your EDA points to, and why.

---

## Step by step (everyone)

**1. Get the data** from the Google Drive link the lead shared. Put the file in `data/raw/` inside the repo
(create the folder if needed). Don't rename or edit it.

**2. Get the latest team branch and make your own branch from it**
```sh
cd ~/fall2026-american-airlines        # wherever your clone is
git fetch origin --prune
git switch -c team-02/your-name origin/team-02/main
```
- Already have a `team-02/your-name` branch whose Deliverable 2 work is already merged (Aidan, Aditi, Kimberly)?
  Reuse it instead: `git switch <your branch>` then `git pull origin team-02/main`
- **Chau:** make a fresh branch, because your old one has unmerged D2 changes:
  `git switch -c team-02/chau-ha-d3 origin/team-02/main`
- Error `cannot lock ref`? You still have an old local `team-02` branch: `git branch -m team-02 old-team-02`, then retry.

**3. Create your blank notebook**
- Run `jupyter notebook` from the repo folder
- Go into `team-02-modeling/notebooks/` → **New → Python 3**
- Rename it `eda_your-name.ipynb` (click the title at the top)
- Load the file from `../../data/raw/<file name>` (the notebook sits two folders below the repo root)

**4. Do the EDA** — everything in "What your notebook must cover" above.

**5. Send your hand-off to the lead** in the team chat — **words only, no numbers or rows from the data**
(e.g. "`<column>` is a leak: only known after travel").

**6. Clear, commit only your notebook, push**
```sh
# in Jupyter: Kernel → Restart & Clear Output, then save
git status                      # must NOT show anything under data/
git add team-02-modeling/notebooks/eda_your-name.ipynb
git commit -m "deliverable3: EDA (your-name)"
git push -u origin HEAD
```

**7. Open a PR on GitHub**: base **`team-02/main`** (GitHub defaults to `main` — change it!) ← compare your branch.
Request your review-ring reviewer. Don't merge your own PR; the lead merges.

**8. Review your assigned teammate's PR** with at least one real question about their decisions.

---

## Folder layout (agreed)

```
team-02-modeling/
├── README.md          ← this plan
├── FEATURE_SET.md     ← the team's agreed feature set (lead assembles from hand-offs)
├── .gitignore         ← blocks data files from being committed
├── notebooks/
│   ├── eda_<your-name>.ipynb   ← one per member, built from blank
│   └── feature_set.ipynb       ← the team's final feature-set code (Aditi)
├── models/            ← saved models (later deliverables)
└── results/           ← names-only outputs; real-number charts stay local
```

## Lead's checklist (Akshith)

- [ ] Share the Drive link with the team privately
- [ ] Merge this setup into `team-02/main` so everyone can branch from it
- [ ] Review Sahith's PR; merge each member's PR into `team-02/main` after its review
- [ ] Fill `FEATURE_SET.md` from the hand-offs; agree on the model family with the team
- [ ] Confirm the AA row in the root `DATA.md` matches how we accessed the data
- [ ] Open the PR `team-02/main` → `main`, tag a PM
