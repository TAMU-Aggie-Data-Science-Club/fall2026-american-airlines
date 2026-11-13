# Team 02 — Deliverable 3 Feature Set

> Assembled by the lead on `team-02/main` from each member's Section 12 hand-off.
> **NDA:** describe columns and decisions only — no raw values, sample rows, or charts with real numbers.

| Section | Owner |
|---|---|
| Target and loss | Aidan Chance |
| Leakage decisions | Sahith Yamani |
| Missingness handling | Chau Ha |
| Validation scheme and date features | Akshith Reddy Sheri |
| Encoding, relationships, feature-set code | Aditi Walvekar |
| Model family, feature register, data access | Akshith Reddy Sheri |

## Target and model family

- **Target:** `<column>` — *(what it means; regression or classification)* — **Aidan**
- **Outlier handling and loss to train on:** — **Aidan**
- **Model family we are aiming at:** *(e.g. gradient-boosted trees)* — **team decision, written by Akshith**
- **Why (EDA evidence, at least three, citing members' notebooks):**
  1. 
  2. 
  3. 
- **Validation scheme:** *(time-ordered split on `<date column>` if there is a real time axis, otherwise K-fold — and why)* — **Akshith**

## Leakage decisions — Sahith

| Column | Decision (leak / id / keep) | Why — would we have it at prediction time? |
|---|---|---|
| | | |

## Missingness handling — Chau

| Column | % missing (rounded) | Random or tied to something? | Handling |
|---|---|---|---|
| | | | |

## Date features — Akshith

| Feature | Built from | Why |
|---|---|---|
| | | |

## Encoding and relationships — Aditi

| Column | Cardinality (rounded) | Encoding | Why |
|---|---|---|---|
| | | | |

## Feature register — Akshith (from everyone's hand-offs)

Produced by code in `notebooks/feature_set.ipynb` (Aditi), built from everyone's hand-offs.
One row per column kept, engineered or dropped.

| Feature | Kept / engineered / dropped | Encoding | Reason (owner + notebook section) |
|---|---|---|---|
| | | | |

## Data access (for `DATA.md`) — Akshith

- How the team accessed the AA dataset: *(Google Drive link from <PM/Discord>, downloaded to `data/raw/`, never committed)*
- File name(s) and date downloaded:

## Open questions for the PMs

- *(any ambiguous columns we could not interpret without guessing)*
