# Final model report — Team 10 (Deliverable 5)

**Notebooks:** `01_schema` → `02_eda` → `03_model_selection` → `04_final_model` (all in `notebooks/`)
**Data:** `data/raw/aa_fares_compact_SYNTHETIC.csv`, 200 synthetic AA fare quotes (199 after cleaning)
**Target:** `total_price` (USD) of one quote · **Saved model:** `models/final_ridge.joblib`

## 1. What the model predicts, and over what horizon

Given an itinerary at the moment it is quoted (route distance, cabin, booking window, departure
time, peak flags, load factor, lowest competitor quote), the model predicts the fare AA quotes.

**It is not a two-year demand forecast.** The brief asks for one, but this file has no dates and
no demand variable, so neither a two-year nor a next-step demand forecast can be built or tested
on it. The details are in `forecast-evaluation.md`. What the data *does* support is a
cross-sectional pricing model, whose horizon is the 0–119-day booking window.

## 2. Final training procedure

1. `load_data()`: read and type the 13 columns.
2. `clean()`: drop duplicates, missing targets, and prices that are non-positive or more than 3×
   away from the competitor quote (removes 1 data-entry error).
3. `build_features()`: 16 features (`feature-set.md`).
4. Pipeline: `StandardScaler` → `Ridge`, wrapped to fit on `log1p(price)` and return dollars.
5. Tuning on 159 training rows (repeated 5-fold CV × 3): alpha ∈ [1e-3, 1e3] × {no interactions,
   all pairwise interactions}. **Chosen: no interactions, alpha = 0.316** (CV MAE $19.56; pairwise
   interactions were worse at $28.33, because 152 terms overfit 159 rows).
   Plot: `visuals/d5_ridge_tuning.png`.
6. Score once on the 40 held-out rows, then refit on all 199 rows and save.

## 3. Test results (summary; full detail in `forecast-evaluation.md`)

| model | MAE | RMSE | MAPE |
|---|---|---|---|
| Final Ridge | $21.97 | $37.82 | 6.4% |
| Match-the-competitor rule | $16.75 | $25.59 | 5.4% |
| Cabin median | $159.85 | $221.68 | 77.7% |
| XGBoost, no competitor | $76.83 | $113.64 | 23.9% |

The final model is far better than a cabin-only price (−86% MAE) and than any model without the
competitor quote. **But it does not beat the plain competitor rule**, either here or reliably in CV.
80% interval coverage is 82.5% and 90% coverage is 85.0%.

## 4. How the model behaves

**Coefficients** (log price per 1 SD; `visuals/d5_coefficients.png`):

| feature | multiplier per +1 SD |
|---|---|
| log competitor price | × 1.79 |
| log distance | × 1.066 |
| cabin rank | × 1.046 |
| last-minute (≤ 14 days) | × 1.027 |
| nonstop | × 1.017 |
| holiday period | × 1.015 |
| everything else | within ± 1.1% |

**Permutation importance on test** (MAE increase when shuffled): competitor price +$205,
distance +$13, cabin +$7, nonstop +$2, all others under $1. Three inputs do the work.

These coefficients are conditional on the competitor quote. Cabin and the 14-day step look small
here because the competitor quote *already* reflects them: a last-minute Business fare is expensive
at every airline. Without the competitor quote, those features carry the model (XGBoost, MAE $53 in CV).

**Learning curve** (`visuals/d5_learning_curve.png`): validation MAE falls from $28 at 25 rows to
$18.7 at 127 rows, while training MAE stays around $16–17. The gap is small and closing, so more
rows would help a little. The floor (~$16, ~6%) looks like noise the features cannot explain.

**Segment errors** (out-of-fold, all 199 rows; `visuals/d5_segment_errors.png`):

| segment | n | MAE | MAPE |
|---|---|---|---|
| Basic Economy | 57 | $15.8 | 7.6% |
| Main Cabin | 91 | $16.0 | 6.4% |
| Premium Economy | 33 | $21.5 | 5.1% |
| Business/First | 18 | $55.5 | 5.7% |
| booked 0–14 days out | 28 | $44.5 | 7.7% |
| booked 15+ days out | 171 | $16.5 | 6.2% |
| long-haul (> 2,200 mi) | 30 | $33.9 | 6.6% |

In relative terms the error is flat at ~5–8% everywhere. Dollar errors are larger on expensive
fares simply because those fares are larger. The weakest segment is **last-minute bookings**
(7.7%), where fares are most volatile and only 28 rows exist.

**Observed vs predicted:** `visuals/d5_observed_vs_predicted.png`. Points follow the diagonal
across a 30× price range. Residuals sit within about ±15%, apart from one cheap fare at +23%.

## 5. Business interpretation for American Airlines

**What the result says.** In this data, AA's quoted fare is the lowest competitor quote ± about 6%.
Distance, cabin and booking window explain only a few percent beyond that. Either AA's pricing
(as simulated here) tracks the market closely, or the competitor quote is captured in a way that
already reflects AA's own price. **The second possibility is a leakage risk to confirm before any
real use.**

**Uses this supports:**

- **Fare-audit / anomaly flagging.** Compare every filed fare with the model's interval (about
  ×0.88–1.13 of the prediction). A fare outside the band is either a filing error, like the $3,804
  quote removed in D3 (11.7× the competitor), or a deliberate deviation worth a second look.
  That catches costly mistakes cheaply.
- **Competitive-position monitoring.** The model makes "AA vs market" explicit per segment. A
  sustained shift in the AA/competitor ratio on a route, cabin or booking window is a signal for
  revenue management.
- **Continuity when the competitor feed is down.** The no-competitor XGBoost gets within about
  15–24% (MAPE). That is enough for rough guidance, not for setting fares.

**Uses it does not support:**

- Capacity, schedule, staffing or fleet planning. There is no demand variable.
- Anything two years out. There is no time axis, so nothing about trend or drift can be learned
  or tested.
- Setting fares autonomously. The model learns what AA *did* price, not what it *should* price;
  there is no booking or revenue outcome to optimise.

**Risks of acting on it:** the dependence on one input (a broken or late competitor feed degrades
MAE from ~$20 to ~$53–77); the possibility that the competitor quote leaks AA's own price; and
synthetic data. None of these numbers describe real AA pricing.

## 6. Recommended follow-up data and monitoring

1. **Dates** (`search_date`, `departure_date`) to enable out-of-time validation, drift monitoring
   and booking-curve modelling.
2. **A demand target** (bookings or passengers per route-day/month, e.g. the PAX forecasting file)
   over ≥ 3 years. This is the minimum for a two-year forecast tested on a full 24-month
   rolling-origin hold-out.
3. **Competitor-quote provenance:** capture time relative to AA's quote, and cabin matching.
4. **Monitoring once live:** monthly MAPE and interval coverage by cabin and booking window;
   alert if 80% coverage drops below ~75% or the AA/competitor ratio drifts by more than 5%.

## 7. Decision summary

- **Model shipped from this work:** Ridge on log price (D4 baseline, tuned), with
  split-conformal intervals. XGBoost without the competitor price is the fallback.
- **Honest headline:** ~6% typical error, but no better than "price at the competitor quote".
  Its value is in the explicit adjustments, the intervals, and the anomaly flagging, not in
  extra accuracy.
- **Not delivered:** a two-year demand forecast, which this dataset cannot support. The data
  needed to build one is listed in §6.
