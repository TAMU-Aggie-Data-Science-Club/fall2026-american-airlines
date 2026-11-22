# Baseline decision — Team 10 (Deliverable 4)

## The rule (fixed before any score was computed)

The **seasonal naive** is the default forecast. A fitted candidate replaces it only if all three
of these hold:

1. at the **select** origin, its route-month WAPE is lower than the naive's in **year 1**;
2. at the **select** origin, its route-month WAPE is lower than the naive's in **year 2**;
3. at the **stability** origin, its 24-month route-month WAPE is **no worse** than the naive's.

If several candidates pass, the one with the lowest select-origin 24-month WAPE is carried. A
candidate that cannot be fitted at an origin (too few clean training pairs) fails that origin's
checks. The rule is implemented in `carried_model()` in `src/forecasting.py`. `04_model_comparison.ipynb`
applies it and writes the result to `decision.json`, which `05_two_year_forecast.ipynb` reads. D5
never re-chooses.

### Why this rule

- **Year 2 is checked separately** because the brief asks for two years. A model that only wins
  next year does not answer it.
- **The stability check guards against one lucky window.** It only asks "no worse", because that
  origin's history is short and close to the disruption.
- **The default is the model with no settings.** If the evidence is mixed, the forecast that
  cannot be overfit wins.

## Decision

<!-- FILL:decision -->
| | |
|---|---|
| Carried model | Seasonal naive (all parametric and trend-adjusted candidates failed the selection gate) |
| Margin over the naive, select origin, year 1 / year 2 | Best challenger (Gradient-boosted growth): -0.0077 / -0.0711 WAPE (0.1621 vs 0.1544 in y1; 0.2522 vs 0.1811 in y2) — 0.0000 / 0.0000 (Seasonal naive retained) |
| Stability check | Pass for Seasonal naive (0.2297 24m WAPE); Fail for all growth challengers (best challenger Gradient-boosted growth at 0.2330, -0.0033 behind naive) |
| Decided by / date | Team 10 / 2026-11-22 |

<!-- /FILL -->

## What is given up either way

- **If the naive is carried:** no growth, so a network that is growing is under-forecast in year 2.
  No route-level explanation is available beyond "last year".
- **If a growth model is carried:** growth is capped at 0.8–1.25 per year, so a genuine step change
  (a new hub, a closure) is understated. It also relies on the disruption window being set right.
