# Baseline decision (Deliverable 4)

**Selected:** 70% `HistGradientBoostingRegressor` + 30% seasonal naive. The benchmark to beat is seasonal naive.

- It has the lowest WAPE of the candidates tested on a common split ([comparison](model-comparison.md)).
- Trained only through 2023 it still beats naive overall, 16.73% against 17.49%.
- The model learns each route's change from two years earlier after removing the year's average change, so the COVID
  rebound is not learned as normal growth.

**Reservations:** trained through 2023, bias is −4.79% against −1.29% for naive, short/medium international gets
worse, and removing features built from the whole earlier calendar year loses the gain. Keep naive as a live challenger.
