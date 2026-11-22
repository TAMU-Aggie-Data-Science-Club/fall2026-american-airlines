# Forecast evaluation

Release run `20261122T062603Z`, source revision `7b5618c`. Horizons 12–24. A horizon is scored only when its target month is already in the extract, which ends in July 2026. Origins are January, April, July, and October 2024, plus January 2025. October 2024 and January 2025 have no 24-month actual yet. This is a retrospective evaluation of periods that had already been inspected.

957,727 matched market cases. WAPE is total absolute error divided by total actual passengers.

| Method | Market WAPE | Network WAPE |
| --- | ---: | ---: |
| Anchor and Adjust | 19.6% | 7.0% |
| Seasonal anchor only | 19.6% | 7.0% |
| Same month last year | 20.2% | 6.8% |
| Last year × trend | 26.6% | 15.3% |

Every scored fold selected a mix weight of zero from earlier matured outcomes, so the evaluated Anchor and Adjust forecast is the seasonal anchor. Market WAPE is 15.9% at 12 months, 19.2% at 18 months, and 22.1% at 24 months. At 24 months, same-month-last-year is 21.8%.

Nominal 80% intervals covered 82.8% of January 2024 cases and 69.0% of January 2025 cases. Volume-weighted coverage on that last origin is 60.9%.

The chart of market WAPE by horizon is `visuals/wape_by_horizon.png`. The six review answers are in `docs/EVALUATION.md`.
