# Team 02 — How we finished (procedure)

**Deadlines from the PMs:** final project/model **Saturday, Nov 21**; slides (2–3) **Sunday, Nov 22**.
**NDA:** notebooks print summaries only; outputs cleared before commit; hand-offs in words; never commit `data/`.

## Approach

Instead of finishing deliverables one at a time, we built **one pipeline** that every later step reuses:

```
data/raw (AA files, local only)
   └─ notebooks/feature_set.ipynb ──► data/processed/team02_features.csv (local only)
         ├─ notebooks/model_comparison.ipynb   D4: baseline vs Ridge vs XGBoost (2 losses), last 12 months held out
         │      └─ results/model-comparison.md, results/baseline-decision.md
         └─ notebooks/two_year_forecast.ipynb  D5: 24 months held out, forecast all at once, intervals, diagnostics,
                └─ results/final-model-report.md, results/forecast-evaluation.md, charts in data/visuals/ (local only)
```

Supporting EDA: `eda_akshithreddy-sheri.ipynb` (time axis, validation), `eda_aidan-chance.ipynb` (target),
`eda_sahith-yamani.ipynb` (leakage), `eda_categoricals_akshithreddy-sheri.ipynb` (encoding). Decisions: `FEATURE_SET.md`.

## Shared rules
- Time split only, never shuffled. D4 test = last 12 months; D5 test = last 24 months.
- D5 uses only information known at the forecast date: "same month two years ago", not "last year".
- Metrics: MAE, RMSE, **WAPE** (headline). Baselines: same month last year (D4), same month two years ago (D5).
- Encoders and early stopping only ever see training months.

## Who did what

| Person | Part |
|---|---|
| Akshith Reddy Sheri (lead) | pipeline, feature set, categoricals takeover, baseline, comparison + decision docs, merges, final PR, slides |
| Aidan Chance | XGBoost + loss decision, reproducibility run, `final-model-report.md` |
| Sahith Yamani | Ridge, reproducibility run, `forecast-evaluation.md`, business section |
| Chau Ha | missingness (D3), "models considered but not run" paragraph |
| Aditi Walvekar | slide text drafts (problem approach, data aggregation) |
