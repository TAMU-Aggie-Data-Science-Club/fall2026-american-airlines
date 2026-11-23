# Forecast evaluation (Deliverable 5)

Test year 2025. Reproduce with `scripts/origin_check.py`. Cells show WAPE % (bias %).

| Segment | Seasonal naive | A: trained 2022–2024 | B: trained 2022–2023 | C: B without calendar-year features |
|---|---|---|---|---|
| All | 17.49 (−1.29) | 15.69 (−0.44) | **16.73 (−4.79)** | 17.80 (−4.95) |
| Domestic | 16.06 (+0.63) | 14.31 (+0.35) | 15.08 (−3.88) | 16.10 (−3.76) |
| Short/medium international | 21.47 (−4.11) | 20.10 (−2.85) | 21.79 (−7.50) | 22.42 (−8.58) |
| Long-haul international (3,000+ mi) | 23.07 (−12.07) | 20.27 (−3.26) | 22.49 (−8.03) | 24.50 (−9.11) |
| High volume | 14.55 (−0.61) | 12.92 (−0.66) | 14.09 (−5.56) | 14.93 (−4.70) |
| Mid volume | 24.51 (−3.93) | 22.34 (−0.95) | 22.89 (−4.34) | 24.76 (−6.34) |
| Thin | 38.88 (+2.31) | 35.77 (+9.96) | 37.19 (+12.77) | 37.78 (+0.12) |

## How to read it

- **A is the notebook's design.** Its inputs are lagged, but it learns from 2024 outcomes, which would not exist when
  forecasting 2025 from 2023. Do not quote 15.69% as two-year-ahead accuracy.
- **B is our headline.** It approximates one forecast made at the end of 2023, 13 to 24 months ahead. It beats naive
  overall and in five of six segments, but under-forecasts more.
- **C is a harsher check.** Some features summarize the whole calendar year two years earlier, reaching to within 13
  months of the target month. Without them the model loses to naive overall. C still trains through 2023, so it
  remains generous to a strict 24-month horizon for every month.
- External data adds little: B with AA data only scores 16.77%.

## Limitations

- One test year and no rolling-origin backtest or prediction intervals.
- FAA and Census files are joined with a two-year lag on the data year, but are published later than that and revised.
- The notebook's fare, revenue and 36-month figures also train on 2024 outcomes and were not re-tested, so they are
  not claimed here.
- The notebook's "Unadjusted HGB" row is typed in, not computed.
- Volume segments are defined by actual 2025 volume. Routes with no traffic 24 months earlier are not forecast.
