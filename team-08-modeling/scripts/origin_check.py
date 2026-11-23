"""Re-score the final notebook's 24-month model under different training cut-offs.

Usage (from the repo root): python team-08-modeling/scripts/origin_check.py [data_dir]
Reuses the notebook's own setup and feature cells. Prints percentages only.
"""
import json
import os
import sys
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
HERE = Path(__file__).resolve().parent
NB = HERE.parent / "notebooks" / "AA_02_modeling_and_evaluation_v3_final.ipynb"
data_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else HERE.parents[1] / "data" / "raw"
code = ["".join(c["source"]) for c in json.loads(NB.read_text(encoding="utf-8"))["cells"] if c["cell_type"] == "code"]
os.chdir(data_dir)
g, feats = {}, {}
exec(code[0], g)  # imports and helpers
exec(code[1], g)  # feature table
exec(code[3].split("trva_mask")[0], feats)  # feature lists
df, wape = g["df"], g["wape"]
CORE, EXT = feats["INV_CORE"], feats["INV_EXT"]
# Built from the whole calendar year two years earlier (up to 13 months before the target month).
CAL = {"lag24_pct_rank", "lag24_zscore", "season_ratio_lag24", "yoy_growth_z", "fare_lag24_z"}

has_lag = df["pax_lag_24"].notna()
test = df[has_lag & (df["DEP_YEAR"] == 2025)]
y, naive = test["OD_PAX"].values, test["pax_lag_24"].values


def ensemble(years, cols):
    tr = df[has_lag & df["DEP_YEAR"].isin(years)]
    target = g["weighted_debias_target"](tr)
    w = np.sqrt(np.clip(tr["pax_lag_24"].values, 1, None))
    m = g["HistGradientBoostingRegressor"](max_iter=200, learning_rate=0.05, max_depth=7, min_samples_leaf=80,
                                           l2_regularization=5.0, random_state=42)
    m.fit(tr[cols], target, sample_weight=w)
    smear = np.average(np.exp(target - m.predict(tr[cols])), weights=w)
    trees = np.expm1(np.clip(test["log_pax_lag_24"].values + m.predict(test[cols]), 0, 15)) * smear
    return 0.7 * trees + 0.3 * naive


def score(p, k):
    return f"{wape(y[k], p[k]):.2f} ({(p[k].sum() - y[k].sum()) / y[k].sum() * 100:+.2f})"


runs = {
    "naive": naive,
    "A trained 2022-24": ensemble([2022, 2023, 2024], EXT),
    "B trained 2022-23": ensemble([2022, 2023], EXT),
    "B, AA data only": ensemble([2022, 2023], CORE),
    "C = B without calendar-year features": ensemble([2022, 2023], [c for c in EXT if c not in CAL]),
}
pax, dom, dist = test["OD_PAX"], test["is_domestic"] == 1, test["distance_mi"]
segments = {
    "All": pax >= 0, "Domestic": dom, "Short/med intl": ~dom & (dist < 3000), "Long-haul intl": ~dom & (dist >= 3000),
    "High volume": pax >= 5000, "Mid volume": (pax >= 500) & (pax < 5000), "Thin": pax < 500,
}
print("WAPE % (bias %), test year 2025")
for run, pred in runs.items():
    print(f"\n{run}")
    for seg, k in segments.items():
        print(f"  {seg:16s} {score(pred, k.values)}")
