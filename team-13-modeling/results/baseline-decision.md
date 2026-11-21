# Team 13 baseline decision

The completed local decision record is `data/interim/team-13-d4/baseline-decision.md`. It names the candidate to carry into D5, gives its measured advantage over the runner-up, and states its main limitation. Regenerate it by running [the D4 notebook](../../team-13/deliverable4/notebooks/01-model-comparison.ipynb).

The decision uses the lowest pooled 24-month validation MAE, breaking ties by RMSE and then candidate name. Equal weighting reflects the objective of reducing the typical route-month error. Evaluate the selected specification on the reserved final 24 months in D5. Team review should confirm the intended route coverage and whether high-volume markets require different business weights.
